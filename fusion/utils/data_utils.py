import torch
import numpy as np
from torch_geometric.data import Batch


class ObservationBatch(Batch):
    def get_batched_attr(self, attr):
        slices = [
            slice(self.ptr[i].item(), self.ptr[i+1].item())
            for i in range(len(self.ptr) - 1)
        ]
        X = [getattr(self, attr)[s] for s in slices]
        # print(X[0] == X[1])
        X = [x.squeeze(-1) for x in X if x.shape[-1] == 1]
        return torch.stack(X)
