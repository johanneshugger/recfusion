import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import cloudpickle

import numpy as np
from fusion.environments.fusion_env import make_env
from fusion.utils.environment_utils import TimeStep


# def to_timestep_batch(timestep_list):
#     import ipdb; ipdb.set_trace()
#     done, reward, discount, observation = [], [], []
#     for timestep in timestep_list:
#         if done != None:
#             done.append(timestep.done)
#         if reward != None:
#             reward.append(timestep.reward)
#         if discount != None:
#             discount.append(timestep.discount)
#         if observation != None:
#             observation.append(timestep.observation)


def _worker(
    remote,
    parent_remote,
    env_kwargs
    # env_fn_wrapper: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import
    parent_remote.close()
    env = make_env(**env_kwargs)
    # env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                timestep = env.step(data)
                remote.send(timestep)
                # observation, reward, done, info = env.step(data)
                # if done:
                #     # save final observation where user can get it, then reset
                #     info["terminal_observation"] = observation
                #     observation = env.reset()
                # remote.send((observation, reward, done, info))
            elif cmd == "reset":
                timestep = env.reset()
                remote.send(timestep)
                # observation = env.reset()
                # remote.send(observation)
            elif cmd == "action_dim":
                act_dim = env.action_dim()
                remote.send(act_dim)
            elif cmd == "close":
                # env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocessEnvWrapper:
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.
    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.
    .. warning::
        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.
    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_kwargs_list, start_method=None):
        """
        Args: 
          env_kwargs_list: list of env_kwargs
        """
        self.waiting = False
        self.closed = False
        n_envs = len(env_kwargs_list)
        self.n_envs = n_envs

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_kwargs in zip(self.work_remotes, self.remotes, env_kwargs_list):
            args = (work_remote, remote, env_kwargs)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

    def action_dim(self):
        for remote in self.remotes:
            remote.send(("action_dim", None))
        action_dims = [remote.recv() for remote in self.remotes]
        return action_dims

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        timesteps = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return timesteps

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        timesteps = [remote.recv() for remote in self.remotes]
        return timesteps

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    from fusion.environments.fusion_env import RecursiveFusionEnv, make_env
    from fusion.io.opengm_benchmark import load_single_opengm_benchmark_graph
    from fusion.utils import pyg_to_nifty

    num_processes = 6
    env_kwargs = dict(
        benchmark='opengm',
        filepath='data/seg-3d-300/graphs/gm_knott_3d_074.npy',
        proposal_solver='multicut_gaec',
        subroutine_solver='multicut_gaec',
        num_steps=10
    )

    env = SubprocessEnvWrapper([env_kwargs for _ in range(num_processes)], 'spawn')

    import time
    t0 = time.time()
    timesteps = env.reset()
    N = timesteps[-1].observation.x.shape[0]

    for t in tqdm(range(20)):
        env.step_async(np.random.uniform(0.0, 1.0, (num_processes, N)))
        timesteps = env.step_wait()

    env.close()

    print('time: ', time.time() - t0)

# class CloudpickleWrapper:
#     """
#     Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
#     :param var: the variable you wish to wrap for pickling with cloudpickle
#     """
# 
#     def __init__(self, var: Any):
#         self.var = var
# 
#     def __getstate__(self) -> Any:
#         return cloudpickle.dumps(self.var)
# 
#     def __setstate__(self, var: Any) -> None:
#         self.var = cloudpickle.loads(var)



    # def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
    #     if seed is None:
    #         seed = np.random.randint(0, 2**32 - 1)
    #     for idx, remote in enumerate(self.remotes):
    #         remote.send(("seed", seed + idx))
    #     return [remote.recv() for remote in self.remotes]

    # def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
    #     """Return attribute from vectorized environment (see base class)."""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(("get_attr", attr_name))
    #     return [remote.recv() for remote in target_remotes]

    # def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
    #     """Set attribute inside vectorized environments (see base class)."""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(("set_attr", (attr_name, value)))
    #     for remote in target_remotes:
    #         remote.recv()

    # def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
    #     """Call instance methods of vectorized environments."""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(("env_method", (method_name, method_args, method_kwargs)))
    #     return [remote.recv() for remote in target_remotes]

    # def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
    #     """Check if worker environments are wrapped with a given wrapper"""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(("is_wrapped", wrapper_class))
    #     return [remote.recv() for remote in target_remotes]

    # def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
    #     """
    #     Get the connection object needed to communicate with the wanted
    #     envs that are in subprocesses.
    #     :param indices: refers to indices of envs.
    #     :return: Connection object to communicate between processes.
    #     """
    #     indices = self._get_indices(indices)
    #     return [self.remotes[i] for i in indices]


# def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: gym.spaces.Space) -> VecEnvObs:
#     """
#     Flatten observations, depending on the observation space.
#     :param obs: observations.
#                 A list or tuple of observations, one per environment.
#                 Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
#     :return: flattened observations.
#             A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
#             Each NumPy array has the environment index as its first axis.
#     """
#     assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
#     assert len(obs) > 0, "need observations from at least one environment"
# 
#     if isinstance(space, gym.spaces.Dict):
#         assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
#         assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
#         return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
#     elif isinstance(space, gym.spaces.Tuple):
#         assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
#         obs_len = len(space.spaces)
#         return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))
#     else:
#         return np.stack(obs)
