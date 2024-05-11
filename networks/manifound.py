import sys 
sys.path.append('.')
import torch
from torch.nn import functional as F
import torch.nn as nn
from networks import MLP, PointNet2Cls, PointNet2, PointNet
import numpy as np 
from omegaconf import OmegaConf


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, final_nl=False):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        x_out = x_s + dx
        if final_nl:
            return F.leaky_relu(x_out, negative_slope=0.2)
        return x_out
    
    
class LatentEncoder(nn.Module):
    def __init__(self, in_dim, dim, out_dim):
        super().__init__()
        self.block = ResnetBlockFC(size_in=in_dim, size_out=dim, size_h=dim)
        self.fc_mean = nn.Linear(dim, out_dim)
        self.fc_logstd = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.block(x, final_nl=True)
        return self.fc_mean(x), self.fc_logstd(x)
    
    
class ManiFM(nn.Module):
    def __init__(
        self,
        cfg,
        device,
    ):  # cfg: hand_feature_dim, obj_feature_dim, encoder_out_dim, object_pn_hidden_dim
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.physical_mlps = [
            MLP([1, *[cfg.physical_mlp_hidden for _ in range(cfg.physical_mlp_layer-2)], cfg.physical_feature_dim], bn=False).to(self.device) for p in range(cfg.physcial_property_num)
        ]

        self.obj_feature_net = PointNet2(cfg.obj_in_dim + cfg.physical_feature_dim, hidden_dim=cfg.object_pn_hidden_dim, out_dim=cfg.obj_feature_dim)
        self.hand_feature_net = PointNet(in_dim=cfg.hand_in_dim, hidden_dim=cfg.object_pn_hidden_dim, out_dim=cfg.hand_feature_dim)
                
        self.contact_encoder = PointNet2Cls(3+cfg.obj_feature_dim+cfg.hand_feature_dim + 1 , cfg.encoder_out_dim)
        self.contact_latent = LatentEncoder(in_dim=cfg.encoder_out_dim, dim=cfg.latent_neurons, out_dim=cfg.d_z)
        self.contact_decoder = PointNet(in_dim=3+cfg.obj_feature_dim+cfg.hand_feature_dim + cfg.d_z, hidden_dim=cfg.object_pn_hidden_dim, out_dim=1)
        
        self.force_decoder = PointNet(in_dim=3+cfg.obj_feature_dim+cfg.hand_feature_dim + cfg.d_z +1 , hidden_dim=cfg.object_pn_hidden_dim, out_dim=3)
        

    def get_condition(self, hand, obj):
        physcial_input = obj[:, :, -2*self.cfg.physcial_property_num:-self.cfg.physcial_property_num]
        physcial_features = [p(physcial_input[:, :, i:i+1]) for i, p in enumerate(self.physical_mlps)]
        physcial_feature = torch.zeros_like(physcial_features[0])

        for i in range(self.cfg.physcial_property_num):
            physcial_feature += physcial_features[i] * obj[:, :, -self.cfg.physcial_property_num+i].unsqueeze(-1) # mask

        physcial_feature = physcial_feature / obj[:, :, -self.cfg.physcial_property_num:].sum(dim=-1, keepdim=True) # mean pooling

        obj = obj[:, :, 0:-2*self.cfg.physcial_property_num]

        _, hand_feature = self.hand_feature_net(hand)
        obj_feature = self.obj_feature_net(torch.cat([obj, physcial_feature], dim=-1))

        obj_feature = torch.cat([obj[:, :, :3], obj_feature, hand_feature.unsqueeze(1).repeat(1, obj_feature.shape[1], 1)], dim=-1)
        
        return obj_feature
    
    
    def encode(self, obj_cond, contacts_object):
        contact_latent = self.contact_encoder(torch.cat([obj_cond, contacts_object], -1))  #  (c+x) -> z_posterior
        contact_mu, contact_std = self.contact_latent(contact_latent)  # z_posterior -> mu, std
        z_contact = torch.distributions.normal.Normal(contact_mu, torch.exp(contact_std))  # 
        z_s_contact = z_contact.rsample()
        
        return z_contact, z_s_contact
    
    def decode(self, z_contact, obj_cond):
        z_contact = z_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        contacts_object, _ = self.contact_decoder(torch.cat([obj_cond, z_contact], -1))
        
        forces_object, _ = self.force_decoder(torch.cat([obj_cond, z_contact, contacts_object], -1))
        return contacts_object, forces_object
    

    def infer(self, hand, obj):
        
        with torch.no_grad():
            
            bs = obj.shape[0]
            dtype = obj.dtype
            device = obj.device
            self.eval()
            with torch.no_grad():
                c = self.get_condition(hand, obj)
                z = np.random.normal(0., 1., size=(bs, self.cfg.d_z))
                z = torch.tensor(z,dtype=dtype).to(device)
                
                return self.decode(z, c)


        
    def forward(self, hand, obj, gt_heatmap):
        '''
        obj_cond: shape=(bs, n_points, pointnet_hc) -> (bs, 2048, 64)
        z_contact: shape=(bs, latentD) -> (bs, 16)
        ---
        contacts_pred: shape=(bs, n_points, 1)
        forces_pred: shape=(bs, n_points, 3)
        '''
        c = self.get_condition(hand, obj)
        z_contact, z_s_contact = self.encode(c, gt_heatmap)
        results = {'mean_contact': z_contact.mean, 'std_contact': z_contact.scale}
        contacts_pred, forces_pred = self.decode(z_s_contact, c)
        results.update({'contacts_object': contacts_pred,
                        'forces_object': forces_pred})
        return results
    