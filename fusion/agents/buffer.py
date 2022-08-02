import torch
import scipy
import numpy as np
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from torch_geometric.data import Batch

from bbaselines.utils.mpi_tools import mpi_statistics_scalar


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

    def __init__(self, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = []
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        # self.obs_buf[self.ptr] = obs
        self.obs_buf.append(obs)
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
        self.obs_buf = Batch.from_data_list(self.obs_buf)

    def get(self, batch_size=2):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        indices = np.random.permutation(self.max_size)

        start_idx = 0
        while start_idx < self.max_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        data = dict(
            obs=self.obs_buf[batch_inds],
            act=self.act_buf[batch_inds],
            # self.values[batch_inds].flatten(),
            logp=self.logp_buf[batch_inds],
            adv=self.adv_buf[batch_inds],
            ret=self.ret_buf[batch_inds],
        )
        import ipdb; ipdb.set_trace()

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,  # self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) \
                if k != 'obs' else Batch.from_data_list(v) \
                for k, v in data.items()}


# class RolloutBuffer:
#     """
#     Rollout buffer for on-policy algorithms A2C/PPO.
#     It corresponds to ``buffer_size`` transitions collected
#     using the current policy.
#     This experience will be discarded after the policy update.
#     In order to use PPO objective, we also store the current value of each state
#     and the log probability of each taken action.
#     The term rollout here refers to the model-free notion and should not
#     be used with the concept of rollout used in model-based RL or planning.
#     Hence, it is only involved in policy and value function training but not action selection.
#     :param buffer_size: Max number of element in the buffer
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param device:
#     :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
#         Equivalent to classic advantage when set to 1.
#     :param gamma: Discount factor
#     :param n_envs: Number of parallel environments
#     """
# 
#     def __init__(
#         self,
#         buffer_size: int,
#         action_dim: int,
#         gae_lambda: float = 1,
#         gamma: float = 0.99,
#     ):
# 
#         super().__init__()
#         self.buffer_size = buffer_size
# 
#         self.action_dim = action_dim
#         self.gae_lambda = gae_lambda
#         self.gamma = gamma
#         self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
#         self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
#         self.generator_ready = False
#         self.reset()
# 
#     def reset(self) -> None:
#         self.pos = 0
#         self.full = False
#         self.generator_ready = False
#         self.observations = []
#         self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
#         self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
#         self.returns = np.zeros((self.buffer_size), dtype=np.float32)
#         self.episode_starts = np.zeros((self.buffer_size), dtype=np.float32)
#         self.values = np.zeros((self.buffer_size), dtype=np.float32)
#         self.log_probs = np.zeros((self.buffer_size), dtype=np.float32)
#         self.advantages = np.zeros((self.buffer_size), dtype=np.float32)
# 
#     def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
#         """
#         Post-processing step: compute the lambda-return (TD(lambda) estimate)
#         and GAE(lambda) advantage.
#         Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
#         to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
#         where R is the sum of discounted reward with value bootstrap
#         (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.
#         The TD(lambda) estimator has also two special cases:
#         - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
#         - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
#         For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.
#         :param last_values: state value estimation for the last step (one for each env)
#         :param dones: if the last step was a terminal step (one bool for each env).
#         """
#         # Convert to numpy
#         last_values = last_values.clone().cpu().numpy().flatten()
# 
#         last_gae_lam = 0
#         for step in reversed(range(self.buffer_size)):
#             if step == self.buffer_size - 1:
#                 next_non_terminal = 1.0 - dones
#                 next_values = last_values
#             else:
#                 next_non_terminal = 1.0 - self.episode_starts[step + 1]
#                 next_values = self.values[step + 1]
#             delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
#             last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
#             self.advantages[step] = last_gae_lam
#         # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
#         # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
#         self.returns = self.advantages + self.values
# 
#     def add(
#         self,
#         obs: np.ndarray,
#         action: np.ndarray,
#         reward: np.ndarray,
#         episode_start: np.ndarray,
#         value: th.Tensor,
#         log_prob: th.Tensor,
#     ) -> None:
#         """
#         :param obs: Observation
#         :param action: Action
#         :param reward:
#         :param episode_start: Start of episode signal.
#         :param value: estimated value of the current state
#             following the current policy.
#         :param log_prob: log probability of the action
#             following the current policy.
#         """
#         if len(log_prob.shape) == 0:
#             # Reshape 0-d tensor to avoid error
#             log_prob = log_prob.reshape(-1, 1)
# 
#         # Reshape needed when using multiple envs with discrete observations
#         # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
#         if isinstance(self.observation_space, spaces.Discrete):
#             obs = obs.reshape((self.n_envs,) + self.obs_shape)
# 
#         self.observations[self.pos] = np.array(obs).copy()
#         self.actions[self.pos] = np.array(action).copy()
#         self.rewards[self.pos] = np.array(reward).copy()
#         self.episode_starts[self.pos] = np.array(episode_start).copy()
#         self.values[self.pos] = value.clone().cpu().numpy().flatten()
#         self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
#         self.pos += 1
#         if self.pos == self.buffer_size:
#             self.full = True
# 
#     def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
#         assert self.full, ""
#         indices = np.random.permutation(self.buffer_size * self.n_envs)
#         # Prepare the data
#         if not self.generator_ready:
# 
#             _tensor_names = [
#                 "observations",
#                 "actions",
#                 "values",
#                 "log_probs",
#                 "advantages",
#                 "returns",
#             ]
# 
#             for tensor in _tensor_names:
#                 self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
#             self.generator_ready = True
# 
#         # Return everything, don't create minibatches
#         if batch_size is None:
#             batch_size = self.buffer_size * self.n_envs
# 
#         start_idx = 0
#         while start_idx < self.buffer_size * self.n_envs:
#             yield self._get_samples(indices[start_idx : start_idx + batch_size])
#             start_idx += batch_size
# 
#     def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
#         data = (
#             self.observations[batch_inds],
#             self.actions[batch_inds],
#             self.values[batch_inds].flatten(),
#             self.log_probs[batch_inds].flatten(),
#             self.advantages[batch_inds].flatten(),
#             self.returns[batch_inds].flatten(),
#         )
#         return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

