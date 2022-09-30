# alpha_gan_from_scratch
Alpha-GAN from scratch


# 数据并行DataParallel

```bash
python dp_train.py
```

# 分布式数据并行DistributedDataParallel(DDP)
```bash
# 检查端口是否被占用：netstat -ntlp | grep 1234
python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 ddp_train.py

```
