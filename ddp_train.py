from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm

from ddp_dataset import DemoDataset
from ddp_model import VanillaModel


device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

demoDataset = DemoDataset()

train_loader = DataLoader(demoDataset, batch_size=2,shuffle=False)

model = VanillaModel().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)

loss_func = nn.CrossEntropyLoss()

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score

model.train()

num_epochs = 10

for epoch in range(num_epochs):
    loop = tqdm((train_loader), total = len(train_loader))
    for x_batch, y_batch_gt in loop:

        x_batch = x_batch.to(device)
        y_batch_gt = y_batch_gt.to(device)

        y_batch = model(x_batch)

        loss = loss_func(y_batch, y_batch_gt)

        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        loop.set_postfix(loss = loss.item())
