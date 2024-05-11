import torch.nn as nn
from torch.nn import functional as F


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class PointNet(nn.Module):
    ''' PointNet-based encoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, 2 * hidden_dim, 1)
        self.conv3 = nn.Conv1d(2 * hidden_dim, 4 * hidden_dim, 1)
        self.conv4 = nn.Conv1d(4 * hidden_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn3 = nn.BatchNorm1d(4 * hidden_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.permute(0, 2, 1)

        return x, self.pool(x, dim=1)