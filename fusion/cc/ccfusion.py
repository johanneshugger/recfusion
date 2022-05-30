import dgl
import networkx as nx
import numpy as np
import torch
import copy
import nifty
import torch_geometric
import pylab
import elf
from elf.segmentation.multicut import multicut_kernighan_lin


def pyg_to_nifty(pyg_graph):
    ngraph = nifty.graph.undirectedGraph(pyg_graph.num_nodes)
    uv_ids = pyg_graph.edge_index.T.numpy()
    ngraph.insertEdges(uv_ids)
    return ngraph
    # nifty.graph.drawGraph(ngraph)
    # pylab.show() 


class ccFusionMove:
    """
        Fusion move for correlation clustering.
    """
    def __init__(self, graph):
        """Args:
        graph: pytorch geometric graph with 'edge_attr' as weights
        """
        self.graph = pyg_to_nifty(graph)
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

        # proposals = torch.stack(proposals)
        # contracted_graph = dgl.to_networkx(self.graph, edge_attrs=['w']).to_undirected()
        # uv = [e for e in contracted_graph.edges()]
        # node_mapping = dict(
        #     zip(sorted(contracted_graph), sorted(contracted_graph))
        # )
        # print(uv)
        # for u, v in uv:
        #     print(u, v)
        #     merge = proposals[:, u] == proposals[:, v]
        #     print(proposals)
        #     print(merge)
        #     u, v = node_mapping[u], node_mapping[v]
        #     if all(merge):
        #         print(u, v)
        #         contracted_graph = nx.contracted_edge(
        #             contracted_graph,
        #             (u, v),
        #             self_loops=False,
        #         )
        #         node_mapping[v] = u
        # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
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
    fusion.fuse(proposals)

    # graph = nx.grid_2d_graph(3, 3)
    # for e in graph.edges():
    #     if e in {((0, 0), (1, 0)), ((0, 1), (1, 1)), ((1, 1), (1, 2)), ((2, 1), (2, 2))}:
    #         graph[e[0]][e[1]]['w'] = np.array([-1])
    #     else:
    #         graph[e[0]][e[1]]['w'] = np.array([1])
    # for n in graph.nodes():
    #     graph.nodes[n]['pos'] = n
    # nx.draw(graph, with_labels=True, font_weight='bold')
    # labels = nx.get_edge_attributes(graph, 'w')
    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos, with_labels=True)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    # plt.show()
    # g = torch_geometric.utils.from_networkx(graph)
    # fusion = ccFusionMove(g)

    # graph = dgl.from_networkx(graph)
    # graph.edata['w'] = torch.ones(24)
    # graph.edata['w'][1] = -1
    # graph.edata['w'][2] = -1
    # graph.edata['w'][9] = -1
    # graph.edata['w'][11] = -1
    # graph.edata['w'][12] = -1
    # graph.edata['w'][16] = -1
    # graph.edata['w'][19] = -1
    # graph.edata['w'][22] = -1
    # t = graph.edges(form='all')
    # for a, b, c in zip(t[0], t[1], t[2]):
    #     print('src', a, 'dst', b, 'num', c, 'w', graph.edata['w'][c])
    # fusion = ccFusionMove(graph)
