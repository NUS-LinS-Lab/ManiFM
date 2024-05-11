import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class PointNet2(nn.Module):
    ''' PointNet++-based encoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''

    def __init__(self, in_dim=6, hidden_dim=128, out_dim=3):
        super().__init__()
        # assert in_dim == 6, "input (xyz, norm), channel=6"
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_dim,
                                          mlp=[hidden_dim, hidden_dim * 2], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=hidden_dim * 2 + 3,
                                          mlp=[hidden_dim * 2, hidden_dim * 4], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=hidden_dim * 4 + 3,
                                          mlp=[hidden_dim * 4, hidden_dim * 8], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=hidden_dim * 8 + hidden_dim * 4,
                                              mlp=[hidden_dim * 8, hidden_dim * 4])
        self.fp2 = PointNetFeaturePropagation(in_channel=hidden_dim * 4 + hidden_dim * 2,
                                              mlp=[hidden_dim * 4, hidden_dim * 2])
        self.fp1 = PointNetFeaturePropagation(in_channel=hidden_dim * 2 + in_dim,
                                              mlp=[hidden_dim * 2, hidden_dim])
        self.conv1 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(out_dim)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz[:, 3:, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
 
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        return feat.permute(0, 2, 1)


class PointNet2Cls(nn.Module):
    def __init__(self, input_feature_dim, output_dim):
        super(PointNet2Cls, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.08, nsample=32, in_channel=input_feature_dim + 3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.16, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 512], group_all=True)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, xyz):
        bs, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1) # xyz to shape [bs, D, point_num]
        l0_points = xyz 
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(bs, 512)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = torch.clamp(x, min=0., max=1.)


        return x
