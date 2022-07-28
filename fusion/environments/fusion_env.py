import torch
import numpy as np
import nifty
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from elf.segmentation.multicut import multicut_kernighan_lin, multicut_gaec, multicut_fusion_moves

from fusion.utils import pyg_to_nifty, TimeStep
from fusion.cc.ccfusion import ccfusion


def energy(ngraph, weights, labels):
    energy = 0.0
    for e in ngraph.edges():
        u, v = ngraph.uv(e)
        if labels[u] != labels[v]:
            energy += weights[e]
    return energy


class RecursiveFusionEnv:

    def __init__(
        self,
        ngraph,
        edge_weights,
        pggraph,
        proposal_solver=multicut_gaec,
        subroutine_solver=multicut_gaec,
        num_steps=100
    ):
        self.ngraph = ngraph
        self.weights = edge_weights
        self.pggraph = pggraph
        self.proposal_solver = proposal_solver
        self.subroutine_solver = subroutine_solver
        self.num_steps = num_steps

    def reset(self):
        self._step = 0
        # Generate initial proposal
        proposal = self.proposal_solver(
            self.ngraph, self.weights, # time_limit=10
        )
        self.current_solution = proposal
        self.energy = energy(self.ngraph, self.weights, proposal)
        observation = self._observation(proposal, reset_env=True)
        timestep = TimeStep(
            done=False, reward=None, observation=observation, discount=None
        )
        return timestep

    def step(self, actions=None):
        import time
        t0 = time.time()
        self._step += 1
        proposal = self.proposal_solver(
            self.ngraph, actions 
        )
        proposals = np.stack([self.current_solution, proposal])
        labels = ccfusion(self.ngraph, self.weights, proposals)
        self.current_solution = labels
        reward = energy(self.ngraph, self.weights, labels) - self.energy
        # print('Energy: ', reward + self.energy)
        # print('Reward: ', reward)
        observation = self._observation(labels)
        timestep = TimeStep(
            done=False if self._step < self.num_steps else True,
            reward=reward,
            observation=observation,
            discount=None
        )
        print('time', time.time() - t0)
        return timestep

    def _observation(self, labels, reset_env=False):
        # Node labels to binary edge labeling
        edge_labels = - torch.ones(self.ngraph.numberOfEdges)
        for e in self.ngraph.edges():
            u, v = self.ngraph.uv(e)
            edge_labels[e] = 1 if labels[u] != labels[v] else 0

        # Concatenate edge weights with edge labeling
        edge_attr = torch.stack(
            [self.pggraph.edge_attr.clone(), edge_labels], dim=1
        )

        if reset_env:
            edge_index = self.pggraph.edge_index.clone()
            graph = Data(edge_index=edge_index, edge_attr=edge_attr)
            # Create linegraph
            line_graph = LineGraph()(graph)
            self.line_graph = Data(
                edge_index=line_graph.edge_index 
            )
        else:
            line_graph = Data(
                x=edge_attr,
                edge_index=self.line_graph.edge_index.clone() 
            )
 
        return line_graph


if __name__ == '__main__':
    import random
    from fusion.io import load_opengm_benchmark
    from fusion.agents.random_agent import RandomAgent
    from fusion.utils import pyg_to_nifty
    from fusion.environments import EnvironmentLoop
    # from ctg_benchmark.loaders import get_cross_validation_loaders
    # from torch_geometric.data import Data

    # loader = get_cross_validation_loaders(root='data/ctg_data/')[0]['train']
    # for data in loader:
    #     if data[0].edge_index.shape[1] < 15000:
    #         continue
    #     pggraph = Data(edge_index=data[0].edge_index, edge_attr=data[0].edge_y)
    #     print(pggraph)
    #     ngraph, edge_weights = celltypegraph_to_nifty(pggraph)
    #     break

    # line_graph = LineGraph()(Data(edge_index=pggraph.edge_index, edge_attr=pggraph.edge_y))
    # ngraph, edge_weights, pggraph = load_snap_graph(
    #     filepath="data/snap/soc-sign-epinions.txt", normalize=False
    # )

    pggraph = load_opengm_benchmark()[0]
    ngraph, edge_weights = pyg_to_nifty(pggraph)

    env = RecursiveFusionEnv(ngraph, edge_weights, pggraph)
    agent = RandomAgent()
    env_loop = EnvironmentLoop(env, agent).run_episode(num_steps=10)
