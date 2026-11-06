"""Contact model training.
Author: Chenxi Wang, Ying Feng
"""
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

def train(model, dataset, num_epochs=20, batch_size=2048, save_ckpt="contact.pth", device="cuda", num_workers=0):
    model = model.to(device)
    model.train()
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    if len(dataloader) == 0:
        raise ValueError("Dataset must contain at least batch_size samples")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for i, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            pred = model(data)

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print({"loss": loss.item()})

    os.makedirs(os.path.dirname(os.path.abspath(save_ckpt)), exist_ok=True)
    torch.save(model.state_dict(), save_ckpt)
    return model
