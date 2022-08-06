import argparse
import sys
import yaml
import fusion.agents as agents
from bbaselines.utils.experiment_utils import VariantGenerator
from fusion.utils.run_utils import run_fusion_experiment


BASELINE_ALGORITHMS = ['a2c', 'ppo']
RUN_KEYS = ['num_cpu', 'data_dir', 'datestamp']


def setup_and_run(baseline_fn, config):
    run_kwargs = dict()
    for key in RUN_KEYS:
        assert key in config, \
            "Key %s is not specified in the configuration file." % key
        val = config[key]
        run_kwargs[key] = val

    vg = VariantGenerator()
    for key, vals in config.items():
        if key in RUN_KEYS:
            continue
        vg.add(key, vals)
    vg.run(baseline_fn, call_experiment=run_fusion_experiment, **run_kwargs)


def parse_args(args):
    arg_dict = dict()
    for i, arg in enumerate(args):
        if '--' in arg:
            arg_key = arg.lstrip('-')
            arg_dict[arg_key] = []
        else:
            try:
                arg = eval(arg)
            except:
                pass
            arg_dict[arg_key].append(arg)
    return arg_dict


if __name__ == '__main__':
    baseline = sys.argv[1]
    assert baseline in BASELINE_ALGORITHMS
    baseline_fn = eval('agents.' + baseline)
    arg_dict = parse_args(sys.argv[2:])

    if 'config' in arg_dict.keys():
        config_path = arg_dict['config'][0]
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise NotImplementedError

    setup_and_run(baseline_fn, config)
