"""Retargeting model training.
Author: Chenxi Wang, Ying Feng
"""
import os
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from .losses import criterion

def train(model, dataset, num_epochs=5, batch_size=2048, save_ckpt="mapping_model.pth", device="cuda", num_workers=0):

    model = model.to(device)
    model.train()
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    if len(dataloader) == 0:
        raise ValueError("Dataset must contain at least batch_size samples")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    target_hand_data = dataset.hand_finger_pose
    target_hand_data = [torch.from_numpy(x).to(device) for x in target_hand_data]

    glove_scales = torch.from_numpy(dataset.glove_scales.astype(np.float32)).to(device) 
    hand_scales = torch.from_numpy(dataset.hand_scales.astype(np.float32)).to(device)

    for epoch in range(num_epochs):
        print("Epoch", epoch)
        for i, glove_data in enumerate(dataloader):
            glove_data = glove_data.to(device)
            glove_data_delta = torch.randn_like(glove_data[..., :3])
            d = 0.001 + torch.rand(glove_data.shape[:2]).to(glove_data.device) * 0.009
            glove_data_delta = glove_data_delta / (torch.linalg.norm(glove_data_delta, dim=-1, keepdim=True) + 1e-8) * d[...,None] * 2 / glove_scales[None, :, None]

            anchor_input, anchor_target = dataset.sample_batch_anchor_data(32)
            anchor_input = torch.from_numpy(anchor_input).to(device)
            anchor_target = torch.from_numpy(anchor_target).to(device)
            input_data = torch.cat([glove_data[..., :3], glove_data[..., :3] + glove_data_delta, anchor_input[..., :3]], dim=0)
            output_data, feats = model(input_data)
            feats = feats[:batch_size]
            pred_hand_data = output_data[:batch_size]
            pred_hand_data_delta = output_data[batch_size:2*batch_size] - pred_hand_data
            anchor_pred = output_data[2*batch_size:2*batch_size+32]
            loss, ret_dict = criterion(glove_data, glove_data_delta, pred_hand_data, pred_hand_data_delta, target_hand_data, anchor_pred, anchor_target, feats, glove_scales, hand_scales)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                ret_dict = {k: v.item() for k, v in ret_dict.items()}
                print(ret_dict)

    os.makedirs(os.path.dirname(os.path.abspath(save_ckpt)), exist_ok=True)
    torch.save(model.state_dict(), save_ckpt)

    return model
