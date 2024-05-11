import torch  
import torch.nn as nn

class MLP(nn.Module): 
    def __init__(self, mlp_spec, bn=True, layer_norm=False, leaky_relu=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec)-1):
            if bn:
                self.mlp.add_module(f"bn{i}", nn.BatchNorm1d(mlp_spec[i]))
            elif layer_norm:
                self.mlp.add_module(f"ln{i}", nn.LayerNorm(mlp_spec[i]))
            self.mlp.add_module(f"linear{i}", nn.Linear(mlp_spec[i], mlp_spec[i+1]))
            if i < len(mlp_spec)-2:
                if leaky_relu:
                    self.mlp.add_module(f"act{i}", nn.LeakyReLU())
                else:
                    self.mlp.add_module(f"act{i}", nn.ReLU())

    def forward(self, x):
        return self.mlp(x)