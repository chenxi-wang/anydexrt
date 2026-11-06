"""Retargeting objectives.
Author: Chenxi Wang, Ying Feng
"""
import numpy as np
import torch

def world_to_local_torch(v, q):
    """
    v: direction vector in world frame, Tensor of shape (..., 3)
    q: local frame, Tensor of shape (..., 4) in (w, x, y, z) format
    """
    w = q[..., 0]
    u = -q[..., 1:] # Negate x,y,z for the conjugate (World -> Local)

    uv = torch.cross(u, v, dim=-1)
    uuv = torch.cross(u, uv, dim=-1)
    
    return v + 2.0 * (w.unsqueeze(-1) * uv + uuv)

def compute_partial_chamfer_loss(pred, target):
    """pred: [B, 3], target: [N, 3], N > B"""
    dists, _ = torch.linalg.norm(pred[:,None]-target[None], dim=-1).min(dim=-1) # [B]
    loss = dists.mean()
    return loss

def compute_local_motion_loss(input_glove_data, input_glove_data_delta, pred_hand_data, pred_hand_data_delta, target_hand_data):
    """input_glove_data: [B, 7], input_glove_data_delta: [B, 3], pred_hand_data: [B, 3], pred_hand_data_delta: [B, 3], target_hand_data: [N, 7]
    """
    dists = torch.linalg.norm(pred_hand_data[:, None] - target_hand_data[None, :, :3], dim=-1) # [B, N]
    nn_idx = torch.argmin(dists, dim=-1) # [B]

    glove_motion = world_to_local_torch(input_glove_data_delta, input_glove_data[:, 3:]) # [B, 3]
    hand_motion = world_to_local_torch(pred_hand_data_delta, target_hand_data[nn_idx, 3:]) # [B, 3]
    loss = 1 - torch.cosine_similarity(glove_motion, hand_motion, dim=-1).mean()
    return loss

def compute_distance_loss(input_glove_data, pred_hand_data, glove_scale, hand_scale):
    """input_glove_data: [B, 7], pred_hand_data_delta: [B, 3]"""
    B = input_glove_data.size(0)
    glove_unnormalized = input_glove_data[...,:3] * glove_scale / 2
    hand_unnormalized = pred_hand_data[...,:3] * hand_scale / 2
    glove_dists = torch.linalg.norm(glove_unnormalized[:, np.newaxis] - glove_unnormalized[np.newaxis], dim=-1)
    hand_dists = torch.linalg.norm(hand_unnormalized[:, np.newaxis] - hand_unnormalized[np.newaxis], dim=-1)
    loss = torch.abs(glove_dists - hand_dists).sum() / (B * (B - 1))
    return loss

def compute_supervised_contrastive_loss(input_data, feats, pos_thresh=0.01, tau=0.1):
    """input_data: [B, 3], feats: [B, 128]"""
    dists = torch.linalg.norm(input_data[:, None] - input_data[None], dim=-1) # [B, B]
    self_mask = torch.eye(len(input_data), dtype=bool).to(input_data.device)
    
    positive_mask = (dists < pos_thresh) & (~self_mask) # [B, B]
    pos_cnt = positive_mask.float().sum(dim=1) # [B]
    training_mask = torch.any(positive_mask, dim=1) # [B]
    
    feats = feats / (torch.linalg.norm(feats, dim=-1, keepdim=True) + 1e-8)
    cos_similarity = torch.matmul(feats, feats.t()) # [B, B]
    cos_similarity_max, _ = cos_similarity.max(dim=1, keepdim=True) # [B, 1]
    cos_similarity = (cos_similarity - cos_similarity_max.detach()) / tau # [B, B]
    all_exp = torch.sum(torch.exp(cos_similarity) * (~self_mask).float(), dim=1, keepdim=True) # [B, 1]
    log_exp = torch.log(all_exp + 1e-8) - cos_similarity # [B, B]
    loss = torch.sum(log_exp * positive_mask.float(), dim=1) / (pos_cnt + 1e-8) # [B]
    loss = torch.sum(loss * training_mask.float()) / (training_mask.float().sum() + 1e-8)

    return loss

def compute_alignment_loss(anchor_pred, anchor_target):
    """anchor_pred: [B, 5, 3/7], anchor_target: [B, 5, 3/7]"""
    return torch.linalg.norm(anchor_pred[..., :3] - anchor_target[..., :3], dim=-1).mean()

def criterion(input_glove_data, input_glove_data_delta, pred_hand_data, pred_hand_delta, target_hand_data, anchor_pred, anchor_target, feats, glove_scales, hand_scales):
    """input_glove_data: [B, 5, 7], input_glove_data_delta: [B, 5, 3], pred_hand_data: [B, 5, 3], pred_hand_delta: [B, 5, 3], target_hand_data: (5 x [N, 7]), anchor_pred: [B, 5, 3/7], anchor_target: [B, 5, 3/7], feats: [B, 5, 128]"""
    partial_chamfer_loss = 0
    contrastive_loss = 0
    local_motion_loss = 0
    distance_loss = 0
    alignment_loss = 0
    ret_dict = {}
    for i in range(pred_hand_data.shape[1]):
        partial_chamfer_loss += compute_partial_chamfer_loss(pred_hand_data[:,i], target_hand_data[i][:,:3])
        contrastive_loss += compute_supervised_contrastive_loss(input_glove_data[:,i,:3], feats[:,i], pos_thresh=0.02/glove_scales[i])
        local_motion_loss += compute_local_motion_loss(input_glove_data[:,i], input_glove_data_delta[:,i], pred_hand_data[:,i], pred_hand_delta[:,i], target_hand_data[i])
        distance_loss += compute_distance_loss(input_glove_data[:,i], pred_hand_data[:,i], glove_scales[i], hand_scales[i])
        alignment_loss += compute_alignment_loss(anchor_pred[:,i], anchor_target[:,i])

    loss = partial_chamfer_loss + contrastive_loss + local_motion_loss + distance_loss + alignment_loss 
    ret_dict["loss"] = loss
    ret_dict["partial_chamfer_loss"] = partial_chamfer_loss
    ret_dict["contrastive_loss"] = contrastive_loss
    ret_dict["local_motion_loss"] = local_motion_loss
    ret_dict["distance_loss"] = distance_loss
    ret_dict["alignment_loss"] = alignment_loss
    return loss, ret_dict
