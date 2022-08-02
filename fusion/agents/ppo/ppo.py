import time
import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.distributions.transforms import AffineTransform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from torch_geometric.data import Data

from bbaselines.utils.mpi_tools import proc_id, setup_torch_for_mpi, sync_params, num_procs, mpi_avg_grads, mpi_avg
from bbaselines.utils.logx import EpochLogger

import fusion.io
from fusion.networks import *
from fusion.agents.core import GaussianActor, Critic
from fusion.agents.buffer import RolloutBuffer, count_vars


class Critic(nn.Module):

    def __init__(self, net, act_dim):
        super().__init__()
        self.v_net = net
        self.mlp = MLP(
            input_dim=act_dim,
            hidden_dims=[2048, 1024],
            output_dim=1
        )

    def forward(self, obs):
        vals = self.v_net(obs)
        if isinstance(vals, Data):
            vals = vals.out
        vals = vals.flatten() if vals.ndim == 2 else vals
        vals = self.mlp(vals)
        return torch.squeeze(vals, -1) # Critical to ensure v has right shape.


class SquashedGaussianActor(nn.Module):

    def __init__(self, network, act_dim):
        super().__init__()
        # State-independent log standard deviation
        # TODO: implement optional state-dependent log std
        log_std = torch.zeros(act_dim, dtype=torch.float32) # -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(log_std)
        self.mu_net = network

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        if isinstance(mu, Data):
            mu = mu.out
        mu = mu.flatten() if mu.ndim == 2 else mu
        # log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = self.log_std
        std = torch.exp(log_std)
        base_distribution = dist.Normal(mu, std)
        transformed_dist = TransformedDistribution(
            base_distribution, [TanhTransform(), AffineTransform(loc=0.5, scale=-0.5)]
        )
        return transformed_dist

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class PPOAgent(nn.Module):

    def __init__(
        self,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        buf,
        logger=None,    
        train_critic_iters=80,
        train_policy_iters=80,
        target_kl=0.01,
        clip_ratio=0.2
    ):
        """Args:
            action_space_type: str {'continous', 'discrete'}
        """
        super().__init__()
        self.pi = actor
        self.v = critic

        self.logger = logger
        self.logger.setup_torch_saver(
            {'policy_net': self.pi, 'value_net': self.v} 
        )
        self.buf = buf

        self.pi_optimizer = actor_optimizer
        self.v_optimizer = critic_optimizer
        self.train_critic_iters = train_critic_iters
        self.train_policy_iters = train_policy_iters
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def update(self):
        # Set up function for computing PPO policy loss
        def compute_loss_pi(data):
            # obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
            next(data)
            import ipdb; ipdb.set_trace()

            # Policy loss
            pi, logp = self.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(
                ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
            ) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

            return loss_pi, pi_info

        # Set up function for computing value loss
        def compute_loss_v(data):
            obs, ret = data['obs'], data['ret']
            return ((self.v(obs) - ret)**2).mean()

        data = self.buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a multiple steps of gradient descent
        for i in range(self.train_policy_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_critic_iters):
            self.v_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.v)    # average grads across MPI processes
            self.v_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                          KL=kl, Entropy=ent, ClipFrac=cf,
                          DeltaLossPi=(loss_pi.item() - pi_l_old),
                          DeltaLossV=(loss_v.item() - v_l_old))


def environment_loop(
    agent,
    env,
    epochs,
    logger,
    local_steps_per_epoch,
    steps_per_epoch,
    max_ep_len,
    save_freq
):
    # Prepare for interaction with environment
    start_time = time.time()
    timestep = env.reset()
    observation, ep_return, ep_len = timestep.observation, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    from tqdm import tqdm
    for epoch in range(epochs):
        for t in tqdm(range(local_steps_per_epoch)):
            action, value, log_prob = agent.step(observation)
            next_timestep = env.step(action)
            next_observation = next_timestep.observation
            reward = next_timestep.reward
            done = next_timestep.done

            ep_return += reward
            ep_len += 1

            agent.buf.store(
                next_observation,
                action,
                reward,
                value,
                log_prob
            )

            # save and log
            logger.store(VVals=value)

            observation = next_observation

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, value, _ = agent.step(observation)
                else:
                    value = 0
                agent.buf.finish_path(value)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_return, EpLen=ep_len)
                timestep = env.reset()
                observation, ep_return, ep_len = timestep.observation, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, epoch)

        # Perform VPG update!
        agent.update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


def build_ppo_agent(
    agent,
    policy_network,
    critic_network,
    policy_optimizer,
    critic_optimizer,
    steps_per_epoch,
    local_steps_per_epoch,
    # obs_dim,
    act_dim,
    discount,
    lam,
    logger
):
    policy_network = eval(policy_network['name'])(**policy_network['kwargs'])
    critic_network = eval(critic_network['name'])(**critic_network['kwargs'])

    # Set up optimizers for policy and value function
    policy_optimizer = getattr(optim, policy_optimizer['name'])(
        policy_network.parameters(), **policy_optimizer['kwargs'] 
    )
    critic_optimizer = getattr(optim, critic_optimizer['name'])(
        critic_network.parameters(), **critic_optimizer['kwargs'] 
    )

    # Set up actor and critic
    actor = SquashedGaussianActor(policy_network, act_dim)
    critic = Critic(critic_network, act_dim)

    # Set up experience buffer
    buf = RolloutBuffer(act_dim, local_steps_per_epoch, discount, lam)

    agent = PPOAgent(
        actor=actor,
        critic=critic,
        actor_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        buf=buf,
        logger=logger,
        **agent
    )
    return agent


def ppo(
    env_fn,
    seed=0, 
    epochs=50,
    steps_per_epoch=4000,
    max_ep_len=1000,
    save_freq=10,
    agent=dict(),
    rollout_buffer=dict(),
    critic_network=dict(),
    policy_network=dict(),
    critic_optimizer=dict(),
    policy_optimizer=dict(),
    logger_kwargs=dict(),
):
    # Set variables
    clip_ratio = agent['clip_ratio']
    target_kl = agent['target_kl']
    train_critic_iters = agent['train_critic_iters']
    train_policy_iters = agent['train_policy_iters']
    discount = rollout_buffer['discount']
    lam = rollout_buffer['lam']

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_torch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    act_dim = env.action_space_dim()

    local_steps_per_epoch = int(steps_per_epoch / num_procs())

    # Build actor-critic module
    agent = build_ppo_agent(
        agent,
        policy_network,
        critic_network,
        policy_optimizer,
        critic_optimizer,
        steps_per_epoch,
        local_steps_per_epoch,
        # obs_dim,
        act_dim,
        discount,
        lam,
        logger
    )

    # Sync params across processes
    sync_params(agent)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [agent.pi, agent.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    environment_loop(
        agent,
        env,
        epochs,
        logger,
        local_steps_per_epoch,
        steps_per_epoch,
        max_ep_len,
        save_freq
    )
