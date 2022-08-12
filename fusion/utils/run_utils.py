import psutil
import json

from bbaselines.utils.logx import colorize
from bbaselines.utils.serialization_utils import convert_json
from bbaselines.utils.mpi_tools import mpi_fork
from bbaselines.utils.experiment_utils import setup_logger_kwargs

from fusion.io import *
from fusion.environments import RecursiveFusionEnv, SubprocessEnvWrapper
from fusion.utils.conversion_utils import pyg_to_nifty


def run_fusion_experiment(
    exp_name,
    func,
    seed=0,
    num_cpu=1,
    data_dir=None,
    datestamp=False,
    **kwargs
):
    # Determine number of CPU cores to run on
    num_cpu = psutil.cpu_count(logical=False) if num_cpu=='auto' else num_cpu

    # Send random seed to func
    kwargs['seed'] = seed

    # Be friendly and print out your kwargs, so we all know what's up
    print(colorize('Running experiment:\n', color='cyan', bold=True))
    print(exp_name + '\n')
    print(colorize('with kwargs:\n', color='cyan', bold=True))
    kwargs_json = convert_json(kwargs)
    print(json.dumps(kwargs_json, separators=(',',':\t'), indent=4, sort_keys=True))
    print('\n')

    # Set up logger output directory
    kwargs['logger_kwargs'] = setup_logger_kwargs(exp_name, seed, data_dir, datestamp)

    # Set up environment
    # io_fn = kwargs['env_kwargs']['io_function']
    # filepath = kwargs['env_kwargs']['filepath']
    # pggraph = eval(io_fn)(filepath)
    # ngraph, edge_weights = pyg_to_nifty(pggraph)
    # env_kwargs = {
    #     'ngraph': ngraph,
    #     'edge_weights': edge_weights,
    #     'pggraph': pggraph,
    #     'proposal_solver': kwargs['env_kwargs']['proposal_solver'],
    #     'subroutine_solver': kwargs['env_kwargs']['subroutine_solver'],
    #     'num_steps': kwargs['env_kwargs']['num_steps']
    # }
    # kwargs['env_fn'] = lambda : RecursiveFusionEnv(**env_kwargs)
    env_kwargs = kwargs['env_kwargs']
    kwargs['env_fn'] = lambda : SubprocessEnvWrapper(env_kwargs)
    del kwargs['env_kwargs']

    # Fork into multiple processes
    # mpi_fork(num_cpu)

    # Run func
    func(**kwargs)
