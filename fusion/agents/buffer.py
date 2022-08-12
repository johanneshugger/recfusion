import torch
import scipy
import numpy as np
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from torch_geometric.data import Batch, Data

from bbaselines.utils.mpi_tools import mpi_statistics_scalar

from fusion.utils.data_utils import ObservationBatch


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class RolloutBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, action_dim, buffer_size, num_envs, gamma=0.99, lam=0.95):
        self.action_dim = action_dim[0]
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.gamma, self.lam = gamma, lam
        self.reset()

    def reset(self) -> None:
        self.ptr, self.path_start_idx = 0, 0
        self.observations = [[] * self.num_envs]
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.buffer_size     # buffer has to have room so you can store
        # self.observations[self.ptr] = obs
        self.observations.append(obs)
        if hasattr(obs, 'out'):
            print(obs)
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.values[self.ptr] = val
        self.log_probs[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_value=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_value" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.returns[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]
        self.path_start_idx = self.ptr
        

    def get(self, batch_size=5):
        assert self.ptr == self.buffer_size    # buffer has to be full before you can get
        # if not isinstance(self.observations, Batch):
        #     self.observations = ObservationBatch.from_data_list(self.observations)
        indices = np.random.permutation(self.buffer_size)

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        advantages = self.advantages[batch_inds]
        # advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(advantages)
        advantages = (advantages - adv_mean) / adv_std
 
        data = dict(
            obs=[self.observations[i] for i in batch_inds],
            act=self.actions[batch_inds],
            # self.values[batch_inds].flatten(),
            logp=self.log_probs[batch_inds],
            adv=advantages,
            ret=self.returns[batch_inds],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) \
                if k != 'obs' else ObservationBatch.from_data_list(v) \
                for k, v in data.items()}


#class RolloutBuffer1:
#    """
#    Rollout buffer used in on-policy algorithms like A2C/PPO.
#    It corresponds to ``buffer_size`` transitions collected
#    using the current policy.
#    This experience will be discarded after the policy update.
#    In order to use PPO objective, we also store the current value of each state
#    and the log probability of each taken action.
#    The term rollout here refers to the model-free notion and should not
#    be used with the concept of rollout used in model-based RL or planning.
#    Hence, it is only involved in policy and value function training but not action selection.
#    :param buffer_size: Max number of element in the buffer
#    :param observation_space: Observation space
#    :param action_space: Action space
#    :param device:
#    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
#        Equivalent to classic advantage when set to 1.
#    :param gamma: Discount factor
#    :param n_envs: Number of parallel environments
#    """
#
#    def __init__(
#        self,
#        buffer_size: int,
#        observation_space: spaces.Space,
#        action_space: spaces.Space,
#        device: Union[th.device, str] = "cpu",
#        gae_lambda: float = 1,
#        gamma: float = 0.99,
#        n_envs: int = 1,
#    ):
#
#        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
#        self.gae_lambda = gae_lambda
#        self.gamma = gamma
#        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
#        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
#        self.generator_ready = False
#        self.reset()
#
#    def reset(self) -> None:
#
#        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
#        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
#        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#        self.generator_ready = False
#        super().reset()
#
#    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
#        """
#        Post-processing step: compute the lambda-return (TD(lambda) estimate)
#        and GAE(lambda) advantage.
#        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
#        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
#        where R is the sum of discounted reward with value bootstrap
#        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.
#        The TD(lambda) estimator has also two special cases:
#        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
#        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
#        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.
#        :param last_values: state value estimation for the last step (one for each env)
#        :param dones: if the last step was a terminal step (one bool for each env).
#        """
#        # Convert to numpy
#        last_values = last_values.clone().cpu().numpy().flatten()
#
#        last_gae_lam = 0
#        for step in reversed(range(self.buffer_size)):
#            if step == self.buffer_size - 1:
#                next_non_terminal = 1.0 - dones
#                next_values = last_values
#            else:
#                next_non_terminal = 1.0 - self.episode_starts[step + 1]
#                next_values = self.values[step + 1]
#            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
#            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
#            self.advantages[step] = last_gae_lam
#        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
#        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
#        self.returns = self.advantages + self.values
#
#    def add(
#        self,
#        obs: np.ndarray,
#        action: np.ndarray,
#        reward: np.ndarray,
#        episode_start: np.ndarray,
#        value: th.Tensor,
#        log_prob: th.Tensor,
#    ) -> None:
#        """
#        :param obs: Observation
#        :param action: Action
#        :param reward:
#        :param episode_start: Start of episode signal.
#        :param value: estimated value of the current state
#            following the current policy.
#        :param log_prob: log probability of the action
#            following the current policy.
#        """
#        if len(log_prob.shape) == 0:
#            # Reshape 0-d tensor to avoid error
#            log_prob = log_prob.reshape(-1, 1)
#
#        # Reshape needed when using multiple envs with discrete observations
#        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
#        if isinstance(self.observation_space, spaces.Discrete):
#            obs = obs.reshape((self.n_envs,) + self.obs_shape)
#
#        # Same reshape, for actions
#        action = action.reshape((self.n_envs, self.action_dim))
#
#        self.observations[self.pos] = np.array(obs).copy()
#        self.actions[self.pos] = np.array(action).copy()
#        self.rewards[self.pos] = np.array(reward).copy()
#        self.episode_starts[self.pos] = np.array(episode_start).copy()
#        self.values[self.pos] = value.clone().cpu().numpy().flatten()
#        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
#        self.pos += 1
#        if self.pos == self.buffer_size:
#            self.full = True
#
#    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
#        assert self.full, ""
#        indices = np.random.permutation(self.buffer_size * self.n_envs)
#        # Prepare the data
#        if not self.generator_ready:
#
#            _tensor_names = [
#                "observations",
#                "actions",
#                "values",
#                "log_probs",
#                "advantages",
#                "returns",
#            ]
#
#            for tensor in _tensor_names:
#                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
#            self.generator_ready = True
#
#        # Return everything, don't create minibatches
#        if batch_size is None:
#            batch_size = self.buffer_size * self.n_envs
#
#        start_idx = 0
#        while start_idx < self.buffer_size * self.n_envs:
#            yield self._get_samples(indices[start_idx : start_idx + batch_size])
#            start_idx += batch_size
#
#    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
#        data = (
#            self.observations[batch_inds],
#            self.actions[batch_inds],
#            self.values[batch_inds].flatten(),
#            self.log_probs[batch_inds].flatten(),
#            self.advantages[batch_inds].flatten(),
#            self.returns[batch_inds].flatten(),
#        )
#        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
