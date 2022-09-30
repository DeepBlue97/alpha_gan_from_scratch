import os

from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm

from dp_dataset import DemoDataset
from dp_model import VanillaModel


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # 在环境中设置可见device，有空格没事哈哈哈
device_ids = [0, 1]

device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")  # 表示从cuda:0开始选择gpu

demoDataset = DemoDataset()

train_loader = DataLoader(demoDataset, batch_size=2, shuffle=False)

model = VanillaModel()
model = torch.nn.DataParallel(model, device_ids=device_ids)  # 数据并行
model = model.to(device)  # 放gpu上

optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
# optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)  # 数据并行，该行可不写，写了反而报错

loss_func = nn.CrossEntropyLoss()

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score

model.train()  # 模型置为训练模式

num_epochs = 100

for epoch in range(num_epochs):
    loop = tqdm((train_loader), total = len(train_loader))
    for x_batch, y_batch_gt in loop:

        x_batch = x_batch.to(device)  # 单卡
        y_batch_gt = y_batch_gt.to(device)  # 单卡
        

        y_batch = model(x_batch)

        loss = loss_func(y_batch, y_batch_gt)

        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        loop.set_postfix(loss = loss.item())
