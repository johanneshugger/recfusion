import torch.nn as nn


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(
            self,
            input_dim,
            hidden_dims,
            output_dim,
            activation='ReLU',
            final_activation='Identity'
    ):
        super().__init__()
        layers = []
        for n, k in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]):
            layers.append(nn.Linear(n, k))
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(self.layers)
        self.activation = getattr(nn, activation)()
        self.final_activation = getattr(nn, final_activation)()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = self.activation(layer(x))
            else:
                x = self.final_activation(layer(x))
        return x


if __name__ == '__main__':
    import torch

    mlp = MLP(
        input_dim=5, 
        hidden_dims=[16, 32, 64, 32, 16],
        output_dim=2,
        activation=nn.ReLU,
        final_activation=nn.Tanh
    )
    x = torch.randn(20, 5)
    out = mlp(x)
    import ipdb; ipdb.set_trace()
