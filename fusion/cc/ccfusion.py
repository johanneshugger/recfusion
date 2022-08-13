import dgl
import networkx as nx
import numpy as np
import torch
import copy
import nifty
import torch_geometric
import pylab
import elf
from elf.segmentation.multicut import multicut_kernighan_lin, multicut_gaec
from fusion.utils import pyg_to_nifty


def ccfusion(graph, weights, proposals):
    """Args:
        graph: nifty graph
        weights: 1d numpy array
        proposals: numpy array of shape [num_proposals, num_nodes]
    """
    ufd = nifty.ufd.ufd(size=graph.numberOfNodes)
    ufd.reset()
    for edge in graph.edges():
        u, v = graph.uv(edge)
        merge = all(proposals[:, u] == proposals[:, v])
        if merge:
            ufd.merge(u, v)

    # fuseImpl
    relabeling_set = {ufd.find(n) for n in graph.nodes()}
    fm_graph = nifty.graph.undirectedGraph(len(relabeling_set))

    node_to_dense = np.arange(0, graph.numberOfNodes)
    for i, sparse in enumerate(relabeling_set):
        node_to_dense[sparse] = i

    for edge in graph.edges():
        u, v = graph.uv(edge)
        lu = node_to_dense[ufd.find(u)]
        lv = node_to_dense[ufd.find(v)]
        if lu != lv:
            fm_graph.insertEdge(lu, lv)

    fm_edges = fm_graph.numberOfEdges

    if fm_edges == 0:
        result = np.asarray([ufd.find(n) for n in graph.nodes()])
    else:
        fm_weights = np.zeros(fm_edges)
        for edge in graph.edges():
            u, v = graph.uv(edge)
            lu = node_to_dense[ufd.find(u)]
            lv = node_to_dense[ufd.find(v)]
            assert lu < fm_graph.numberOfNodes
            assert lv < fm_graph.numberOfNodes
            if lu != lv:
                e = fm_graph.findEdge(lu, lv)
                assert e != -1
                fm_weights[e] += weights[edge]

        # fm_labels = multicut_kernighan_lin(fm_graph, fm_weights)
        fm_labels = multicut_gaec(fm_graph, fm_weights)
        for edge in graph.edges():
            u, v = graph.uv(edge)
            lu = node_to_dense[ufd.find(u)]
            lv = node_to_dense[ufd.find(v)]
            if lu != lv:
                if fm_labels[lu] == fm_labels[lv]:
                    ufd.merge(u, v)
        result = np.asarray([ufd.find(n) for n in graph.nodes()])

    return result



class ccFusionMove:
    """
        Fusion move for correlation clustering.
    """
    def __init__(self, graph):
        """Args:
        graph: pytorch geometric graph with 'edge_attr' as weights
        """
        self.graph, _ = pyg_to_nifty(graph)
        self.ufd = nifty.ufd.ufd(size=self.graph.numberOfNodes)
        self.weights = self._get_weights(graph)

    def _get_weights(self, graph):
        weights = []
        edge_list = graph.edge_index.T.tolist()
        for e in self.graph.edges():
            u, v = self.graph.uv(e)
            i = edge_list.index([u, v])
            weights += [float(graph.edge_attr[i])]
        return np.array(weights)

    def fuse(self, proposals):
        self.ufd.reset()
        for edge in self.graph.edges():
            u, v = self.graph.uv(edge)
            merge = all(proposals[:, u] == proposals[:, v])
            if merge:
                self.ufd.merge(u, v)

        # fuseImpl
        relabeling_set = {self.ufd.find(n) for n in self.graph.nodes()}
        fm_graph = nifty.graph.undirectedGraph(len(relabeling_set))

        node_to_dense = np.arange(0, self.graph.numberOfNodes)
        for i, sparse in enumerate(relabeling_set):
            node_to_dense[sparse] = i

        for edge in self.graph.edges():
            u, v = self.graph.uv(edge)
            lu = node_to_dense[self.ufd.find(u)]
            lv = node_to_dense[self.ufd.find(v)]
            if lu != lv:
                fm_graph.insertEdge(lu, lv)

        fm_edges = fm_graph.numberOfEdges

        if fm_edges == 0:
            result = np.asarray([self.ufd.find(n) for n in self.graph.nodes()])
        else:
            fm_weights = np.zeros(fm_edges)
            for edge in self.graph.edges():
                u, v = self.graph.uv(edge)
                lu = node_to_dense[self.ufd.find(u)]
                lv = node_to_dense[self.ufd.find(v)]
                assert lu < fm_graph.numberOfNodes
                assert lv < fm_graph.numberOfNodes
                if lu != lv:
                    e = fm_graph.findEdge(lu, lv)
                    assert e != -1
                    fm_weights[e] += self.weights[edge]

            fm_labels = multicut_kernighan_lin(fm_graph, fm_weights)
            for edge in self.graph.edges():
                u, v = self.graph.uv(edge)
                lu = node_to_dense[self.ufd.find(u)]
                lv = node_to_dense[self.ufd.find(v)]
                if lu != lv:
                    if fm_labels[lu] == fm_labels[lv]:
                        self.ufd.merge(u, v)
            result = np.asarray([self.ufd.find(n) for n in self.graph.nodes()])

        return result

def test():
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8],
         [3, 1, 0, 4, 2, 1, 5, 0, 6, 4, 1, 3, 7, 5, 2, 4, 8, 3, 7, 4, 6, 8, 5, 7]]
    )
    edge_attr = torch.tensor(
         [1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
    )
    pyg_graph = torch_geometric.data.Data(
        edge_index=edge_index, edge_attr=edge_attr
    )
    fusion = ccFusionMove(pyg_graph)
    current_solution = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0]).numpy()
    proposal = torch.tensor([1, 0, 2, 1, 0, 0, 1, 1, 1]).numpy()
    proposals = np.stack([current_solution, proposal])
    r1 = fusion.fuse(proposals)

    ngraph, weights = pyg_to_nifty(pyg_graph)
    r2 = ccfusion(ngraph, weights, proposals)
    assert all(r1 == r2)

if __name__ == '__main__':
    test()
