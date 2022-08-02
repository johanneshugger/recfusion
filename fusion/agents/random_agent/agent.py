import torch
from torch_geometric.nn.models import GCN


class RandomAgent:

    def __init__(self):
        self.network = GCN(
            in_channels=2, hidden_channels=10, num_layers=5, out_channels=1
        )

    def select_action(self, observation):
        actions = self.network(
            observation.x.to(torch.float), observation.edge_index
        ).detach().squeeze(1).numpy()
        return actions

    def observe_first(self, timestep):
        pass

    def observe(self, action, next_timestep):
        pass

    def update(self):
        pass
