import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from dp_model import VanillaModel
from dp_dataset import DemoDataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description="MNIST TRAINING")
parser.add_argument('--device_ids', type=str, default='0', help="Training Devices")
parser.add_argument('--epochs', type=int, default=10, help="Training Epoch")
parser.add_argument('--log_interval', type=int, default=100, help="Log Interval")
parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")

args = parser.parse_args()

device_ids = list(map(int, args.device_ids.split(',')))
dist.init_process_group(backend='nccl')
device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
print(device)
print(args.local_rank)
torch.cuda.set_device(device)
# model = VanillaModel()
model = VanillaModel().to(device)
model = DistributedDataParallel(model, device_ids=[device_ids[args.local_rank]], output_device=device_ids[args.local_rank], find_unused_parameters=True)

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])

# dataset_train = datasets.MNIST('../data', train=True, transform=transform, download=True)
# dataset_test = datasets.MNIST('../data', train=False, transform=transform, download=True)

# sampler_train = DistributedSampler(dataset_train)

# train_loader = DataLoader(dataset_train, batch_size=8, num_workers=8, sampler=sampler_train)
# test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=8)

demoDataset = DemoDataset()

sampler_train = DistributedSampler(demoDataset)

train_loader = DataLoader(demoDataset, batch_size=2, shuffle=False, sampler=sampler_train)  # DDP时，由于用了sampler，shuffle必须为False

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=1)

# for epoch in range(args.epochs):
#     sampler_train.set_epoch(epoch)
#     train(args, model, device, train_loader, optimizer, epoch)
#     if args.local_rank == 0:
#         test(args, model, device, test_loader)
#     scheduler.step()
#     if args.local_rank == 0:
#         torch.save(model.state_dict(), 'train.pt')

model.train()  # 模型置为训练模式

loss_func = nn.CrossEntropyLoss()

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
