from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tqdm import tqdm

from ddp_dataset import DemoDataset
from ddp_model import VanillaModel


demoDataset = DemoDataset()

train_loader = DataLoader(demoDataset, batch_size=2,shuffle=False)

model = VanillaModel()

optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)

loss_func = nn.CrossEntropyLoss()

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score

model.train()

for epoch in range(10):
    for x_batch, y_batch_gt in tqdm(train_loader):

        # with tqdm(total=(len(demoDataset)//2+1)*10, desc=f'Epoch {epoch}/{10}', unit='img') as pbar:

        y_batch = model(x_batch)

        loss = loss_func(y_batch, y_batch_gt)
        # print(loss)

        loss.backward()
        optimizer.step()
        # print(f'epoch: {epoch} done!')

