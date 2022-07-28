import torch
import torch_geometric
import numpy as np
import nifty
from torch_geometric.utils import coalesce, to_networkx, from_networkx, remove_self_loops
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from tqdm import tqdm


def load_snap_graph(filepath, normalize=False):
    """
    """
    u_v_weights = torch.from_numpy(np.loadtxt(filepath, dtype=int))
    u = u_v_weights[:, 0] 
    v = u_v_weights[:, 1]
    weights = u_v_weights[:, 2]
    edge_index = torch.stack([u, v])
    edge_attr = weights

    if normalize:
        out_degrees = torch_geometric.utils.degree(edge_index[0])
        edge_attr = (1. / out_degrees)[edge_index[0]] * edge_attr

    graph = torch_geometric.data.Data(
        edge_index=edge_index, edge_attr=edge_attr
    )
 
    edge_index = torch.stack(
        [torch.cat([u, v]), torch.cat([v, u])]
    )
    edge_attr = torch.cat([edge_attr, torch.zeros_like(edge_attr)])
    edge_index, edge_attr = coalesce(edge_index, edge_attr)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    graph = torch_geometric.data.Data(
        edge_index=edge_index, edge_attr=edge_attr
    )

    nx_digraph = to_networkx(graph, edge_attrs=['edge_attr'], to_undirected=False)
    nx_graph = nx_digraph.to_undirected()
    nifty_graph = nifty.graph.undirectedGraph(nx_graph.number_of_nodes())
    weights = np.zeros(len(nx_graph.edges))
    for e in nx_graph.edges:
        u, v = e
        assert u != v
        w1 = nx_digraph.edges[(u, v)]['edge_attr']
        w2 = nx_digraph.edges[(v, u)]['edge_attr']
        nx_graph.edges[e]['edge_attr'] = w1 + w2
        nifty_graph.insertEdge(u, v)
        i = nifty_graph.findEdge(u, v)
        assert i != -1
        weights[i] = w1 + w2

    edge_index = []
    edge_attr = []
    for e in nifty_graph.edges():
        u, v = nifty_graph.uv(e)
        assert u != v
        edge_index += [[u, v]]
        i = nifty_graph.findEdge(u, v)
        edge_attr += [weights[i]]
    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr)
    graph = Data(edge_index=edge_index, edge_attr=edge_attr)
    # graph = from_networkx(nx_graph, group_edge_attrs=['edge_attr'])
    # Line graph transform
    # linegraph = LineGraph()(from_networkx(nx_graph, group_edge_attrs=['edge_attr']))
    return nifty_graph, weights, graph # linegraph


if __name__ == '__main__':
    epinions = load_snap_graph(
        filepath="data/snap/soc-sign-epinions.txt", normalize=True
    )
    print(epinions)

    slashdot = load_snap_graph(
        filepath="data/snap/soc-sign-Slashdot081106.txt", normalize=True 
    )
    print(slashdot)
