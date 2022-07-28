import numpy as np
import nifty


def pyg_to_nifty(pyg_graph):
    """ Converts a torch_geometric graph into a nifty graph
    """
    ngraph = nifty.graph.undirectedGraph(pyg_graph.num_nodes)
    uv_ids = pyg_graph.edge_index.T.numpy()
    ngraph.insertEdges(uv_ids)
    weights = np.zeros(ngraph.numberOfEdges)
    for e in ngraph.edges():
        assert e != -1
        weights[e] = pyg_graph.edge_attr[e]
    return ngraph, weights
    # nifty.graph.drawGraph(ngraph)
    # pylab.show() 

def nx_to_nifty(nx_graph):
    ngraph = nifty.graph.undirectedGraph(nx_graph.number_of_nodes())
    uv_ids = nx_graph.edge_index.T.numpy()
    ngraph.insertEdges(uv_ids)
    return ngraph

# def celltypegraph_to_nifty(pyg_graph):
#     nifty_graph = pyg_to_nifty(pyg_graph)
#     weights = np.zeros(nifty_graph.numberOfEdges)
#     for e in nifty_graph.edges():
#         assert e != -1
#         weights[e] = pyg_graph.edge_attr[e]
#     return nifty_graph, weights
 
