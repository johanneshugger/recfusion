from ctg_benchmark.loaders import get_cross_validation_loaders


def ctg_graph_to_nifty(pyg_graph):
    import ipdb; ipdb.set_trace()
    nifty_graph = nifty.graph.undirectedGraph(pyg_graph.num_nodes)
    weights = np.zeros(len(pyg_graph.edges))
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
    # graph = from_networkx(nx_graph, group_edge_attrs=['edge_attr'])
    # Line graph transform
 
