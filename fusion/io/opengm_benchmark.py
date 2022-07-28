import os
import torch
import numpy as np
from torch_geometric.data import Data


def load_opengm_benchmark(data_dir='data/seg-3d-300/graphs'):
    """ Loads all instances of a dataset in the opengm benchmark
    Args:
        data_dir: path to directory containing opengm benchmark graphs
    Retruns:
        a list of toch_geometric.data.Data graphs
    """
    graphs = []
    files = [f for f in os.listdir(data_dir) if '.npy' in f]
    for f in files:
        filepath = os.path.join(data_dir, f)
        np_graph = np.load(filepath)
        edges = torch.from_numpy(np_graph[:, :2].astype('int').T)
        weights = torch.from_numpy(np_graph[:, 2])
        graph = Data(edge_index=edges, edge_attr=weights)
        graphs += [graph]
    return graphs


if __name__ == '__main__':
    graphs = load_opengm_benchmark()
    import ipdb; ipdb.set_trace()
