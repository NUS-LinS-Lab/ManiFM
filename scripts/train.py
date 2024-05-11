import torch
import torch.nn as nn
import os
import sys
sys.path.append(".")
from networks import ManiFM
from dataset import RigidBodyDataset, ClothDataset, MPMDataset
import numpy as np
import random
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import count_parameters

from accelerate import Accelerator

def build_work_dir(cfg):
    if cfg.dir.work_dir is None:
        work_dir = "./work_dir"
    else:
        work_dir = cfg.dir.work_dir

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "log"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "models"), exist_ok=True)
    cfg.dir.log_dir = os.path.join(work_dir, "log")
    cfg.dir.model_dir = os.path.join(work_dir, "models")

    return cfg

def set_seed(cfg, accelerator):
    random.seed(cfg.train.seed + accelerator.process_index)
    np.random.seed(cfg.train.seed + accelerator.process_index)
    torch.manual_seed(cfg.train.seed + accelerator.process_index)
    torch.cuda.manual_seed_all(cfg.train.seed + accelerator.process_index)

def load_dataset_list(cfg):
    '''
    remember to add new args if update the config file'''
    dataset_list = []
    info_list = []
    if OmegaConf.is_list(cfg.dir.data_dir.rigid_body):
        # try:
        if len(cfg.dir.data_dir.rigid_body) > 0:
            rigid_body_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.num_object, num_palm_pose=cfg.dir.num_palm_pose, num_motion=cfg.dir.num_motion, use_scale=cfg.dir.use_scale,  remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
            dataset_list.append(rigid_body_dataset)
            info_list.append(f"[info] Rigid body dataset size: {len(rigid_body_dataset)}")
        # except:
        #     info_list.append(f"Missing rigid_body dataset")
    if OmegaConf.is_dict(cfg.dir.data_dir.rigid_body):
        try:
            if len(cfg.dir.data_dir.rigid_body.force_closure.path) > 0:
                force_closure_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.force_closure.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=False, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.force_closure.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.force_closure.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.force_closure.num_motion, use_scale=cfg.dir.use_scale,  remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                dataset_list.append(force_closure_dataset)
                info_list.append(f"[info] Rigid body/Force closure dataset size: {len(force_closure_dataset)}")
        except:
            info_list.append(f"Missing force_closure dataset")
            
        try:
            if len(cfg.dir.data_dir.rigid_body.leap_hand.path) > 0:
                leap_hand_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.leap_hand.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.leap_hand.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.leap_hand.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.leap_hand.num_motion, use_scale=cfg.dir.use_scale, remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                dataset_list.append(leap_hand_dataset)
                info_list.append(f"[info] Rigid body/Leap Hand dataset size: {len(leap_hand_dataset)}")
        except:
            info_list.append(f"Missing leap_hand dataset")
        
        try:
            if len(cfg.dir.data_dir.rigid_body.kinova3f_hand.path) > 0:
                kinova3f_hand_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.kinova3f_hand.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.kinova3f_hand.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.kinova3f_hand.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.kinova3f_hand.num_motion, use_scale=cfg.dir.use_scale,  remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                dataset_list.append(kinova3f_hand_dataset)
                info_list.append(f"[info] Rigid body/Kinova3f hand dataset size: {len(kinova3f_hand_dataset)}")
        except:
            info_list.append(f"Missing kinova3f_hand dataset")
            
        try:
            if len(cfg.dir.data_dir.rigid_body.panda_hand.path) > 0:
                panda_hand_dataset = RigidBodyDataset(pkl_file_directory=cfg.dir.data_dir.rigid_body.panda_hand.path, load_ratio=cfg.dir.load_ratio, generate_new_wrench=True, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, num_object=cfg.dir.data_dir.rigid_body.panda_hand.num_object, num_palm_pose=cfg.dir.data_dir.rigid_body.panda_hand.num_palm_pose, num_motion=cfg.dir.data_dir.rigid_body.panda_hand.num_motion, use_scale=cfg.dir.use_scale, remove_pc_num=cfg.dir.remove_pc_num, remove_pc_prob=cfg.dir.remove_pc_prob, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
                dataset_list.append(panda_hand_dataset)
                info_list.append(f"[info] Rigid body/Panda hand dataset size: {len(panda_hand_dataset)}")
        except:
            info_list.append(f"Missing panda_hand dataset")
    
    # try:
    if len(cfg.dir.data_dir.cloth) > 0:
        cloth_dataset = ClothDataset(cloth_file_directory=cfg.dir.data_dir.cloth, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, use_scale=cfg.dir.use_scale, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
        dataset_list.append(cloth_dataset)
        info_list.append(f"[info] Cloth dataset size: {len(cloth_dataset)}")
    # except:
    #     info_list.append(f"Missing cloth dataset")
        
    # try:
    if len(cfg.dir.data_dir.mpm) > 0:
        mpm_dataset = MPMDataset(pkl_file_directory=cfg.dir.data_dir.mpm, load_ratio=1.0, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, use_scale=cfg.dir.use_scale, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
        dataset_list.append(mpm_dataset)
        info_list.append(f"[info] Mpm dataset size: {len(mpm_dataset)}")
    # except:
    #     info_list.append(f"Missing mpm dataset")

    try:
        if len(cfg.dir.data_dir.rope) > 0:
            rope_dataset = MPMDataset(pkl_file_directory=cfg.dir.data_dir.rope, load_ratio=1.0, use_gaussian_map_ratio=cfg.train.use_gaussian_map_ratio, augment_part_seg_indicator_ratio=cfg.train.augment_part_seg_indicator_ratio, use_region=cfg.dir.use_region, use_physics=cfg.dir.use_physics, use_scale=cfg.dir.use_scale, noisy_upper_level=cfg.dir.noisy_upper_level, random_flip_normal_upper_level=cfg.dir.random_flip_normal_upper_level)
            dataset_list.append(rope_dataset)
            info_list.append(f"[info] Rope dataset size: {len(rope_dataset)}")
    except:
        info_list.append(f"Missing Rope dataset")
    return dataset_list, info_list

def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_coeff):
        return max(min(min_kl_coeff + (max_kl_coeff - min_kl_coeff) * (step - constant_step) / total_step, max_kl_coeff), min_kl_coeff)
    
def train():
    cfg = OmegaConf.load("configs/train.json")

    # For accelerate: initialize accelerator wo step_scheduler_with_optimizer
    accelerator = Accelerator(step_scheduler_with_optimizer=False)

    set_seed(cfg, accelerator)
    cfg = build_work_dir(cfg)
    device = accelerator.device

    # For accelerate: initialize writer only on main process
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=cfg.dir.log_dir)

    assert (cfg.train.use_mix_heatmap and not cfg.train.use_k_heatmap and not cfg.train.use_cross_entropy) or (not cfg.train.use_mix_heatmap and cfg.train.use_k_heatmap and not cfg.train.use_cross_entropy) or (not cfg.train.use_mix_heatmap and not cfg.train.use_k_heatmap and cfg.train.use_cross_entropy), "only one of them can be true" 

    # model
    model = ManiFM(cfg.model, device)
    # load_model(model, "/home/gchongkai/backup/server4_lzixuan/test_notebooks/025_add_1d/log/models/model_84/model.safetensors")
    
    # For accelerate: printing statements you only want executed once per machine
    accelerator.print(f"[info] Model parameters: {count_parameters(model)}")

    dataset_list, info_list = load_dataset_list(cfg)
    for info in info_list:
        print(info) 

    assert len(dataset_list) > 0, "dataset_list is empty"
    train_dataset = ConcatDataset(dataset_list)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True) 
    n_iters = len(train_loader)
    # For accelerate: printing statements you only want executed once per machine 
    accelerator.print(f"[info] Dataset size: {len(train_dataset)}")

    # opt
    backbone_optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    # For accelerate: prepare model, backbone_optimizer, train_loader and lr_scheduler with accelerator. 
    # For accelerate: You should always pass the the learning rate scheduler to prepare(), however if the scheduler should not be stepped at each optimization step, pass step_with_optimizer=False to the Accelerator init.
    lr_scheduler = optim.lr_scheduler.StepLR(backbone_optimizer, step_size=30)
    model, backbone_optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, backbone_optimizer, train_loader, lr_scheduler
    )
    total_step = 0
    # For accelerate: avoid having multiple progress bars in your output by adding disable=not accelerator.is_local_main_process
    progress_bar = tqdm(range(cfg.train.epoch), desc='Training', disable=not accelerator.is_local_main_process)
    for epoch in progress_bar:
        model.train()
        for batch_idx, data in enumerate(train_loader):
            backbone_optimizer.zero_grad()

            results = model(data["input_hand_point_cloud"], data["input_object_point_cloud"], data["mix_heatmap"].unsqueeze(-1))

            mu  = results['mean_contact']
            std = results['std_contact']
            contacts_pred = results['contacts_object']
            forces_pred = results['forces_object']
            
            contact_loss = nn.MSELoss()(contacts_pred.squeeze(-1), data["mix_heatmap"])
            force_loss =  nn.MSELoss()(forces_pred, data["forcemap"]) 
            
        
            ''' kl loss '''
            global_step = epoch * n_iters + batch_idx
            num_total_iter = cfg.train.epoch * n_iters
            weight_kl = kl_coeff(step=global_step,
                        total_step=num_total_iter,
                        constant_step=0,
                        min_kl_coeff=1e-7,
                        max_kl_coeff=cfg.loss.kl_coef)
            
            batch_size = data["input_hand_point_cloud"].shape[0]
            dtype = data["input_hand_point_cloud"].dtype
            p_z = torch.distributions.normal.Normal(
                loc=torch.tensor(np.zeros([batch_size, cfg.model.d_z]), requires_grad=False).to(device).type(dtype),
                scale=torch.tensor(np.ones([batch_size, cfg.model.d_z]), requires_grad=False).to(device).type(dtype))
            q_z_contact = torch.distributions.normal.Normal(mu,std)
            loss_kl_contact = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z_contact, p_z), dim=[1]))
            # robust kl 
            # if cfg.robustkl:
            loss_kl_contact = torch.sqrt(1 + loss_kl_contact ** 2) - 1
                           
            loss_dict = {
                'loss_contact_rec': contact_loss,
                'loss_force_rec': force_loss,
                'loss_kl_contact': loss_kl_contact,
            }
            
            loss_total = weight_kl * loss_kl_contact + cfg.loss.contact_point_loss_coef * contact_loss + cfg.loss.force_loss_coef * force_loss
            loss_dict['loss_total'] = loss_total

            # For accelerate: Replace the loss.backward() line with accelerator.backward(loss).
            accelerator.backward(loss_total)
            backbone_optimizer.step()

            # For accelerate: log loss only on main process
            if accelerator.is_main_process:
                writer.add_scalar("loss/train_loss_contact_rec", loss_dict['loss_contact_rec'].detach(), total_step)
                writer.add_scalar("loss/train_loss_force_rec", loss_dict['loss_force_rec'].detach(), total_step)
                writer.add_scalar("loss/train_loss_kl_contact", loss_dict['loss_kl_contact'].detach(), total_step)
                writer.add_scalar("loss/train_loss_total", loss_dict['loss_total'].detach(), total_step)
                
            if batch_idx % 10 == 0:
                # For accelerate: printing statements you only want executed once per machine
                accelerator.print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, batch_idx, loss_total.item()))

            total_step += 1
        # For accelerate: wait for all processes to reach this point
        if epoch % cfg.train.save_epoch == 0:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, os.path.join(cfg.dir.model_dir, f"model_{epoch}"))


def main():
    train()

if __name__ == "__main__":
    main()
