"""
ECE-GY 9143 - Lab 5 Part A
Jayraj Pamnani
jmp10051

This script implements DistributedDataParallel (DDP) training for CIFAR-10
using the ResNet-18 model architecture from Lab 2, with the same default
SGD hyper-parameters (lr=0.1, momentum=0.9, weight_decay=5e-4) and 2
data-loader workers by default.

"""
import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.nn.functional as F


def setup(rank: int, world_size: int, backend: str, init_method: str):
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dataloaders(batch_size: int, world_size: int, rank: int, data_root: str, num_workers: int):
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    train_ds = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=(rank == 0),
        transform=tfm,
    )
    if world_size > 1:
        dist.barrier()  # make sure data is downloaded before others use it

    sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


class BasicBlock(nn.Module):
    """Basic residual block from Lab 2 (with BatchNorm)."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet backbone from Lab 2."""

    def __init__(self, block, num_blocks, num_classes: int = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 architecture matching Lab 2's default model."""

    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def run_epoch(model, loader, opt, scaler, device, epoch: int, sampler=None, profiler=None):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    compute_time = 0.0

    for images, targets in loader:
        use_amp = scaler is not None

        # Measure compute-only time for this batch:
        #  - EXCLUDES: data loading / DataLoader wait time
        #  - INCLUDES: CPU->GPU transfer, forward, loss, backward, optimizer step.
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = nn.CrossEntropyLoss()(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        compute_time += t1 - t0

        total_loss += loss.item()
        total_acc += accuracy(logits, targets)
        total_batches += 1

        if profiler is not None:
            profiler.step()

    return total_loss / total_batches, total_acc / total_batches, compute_time


def profile_comm_compute():
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb-trace"),
    )


def summarize_profiler(prof):
    events = prof.key_averages()

    def get_cuda_time(e):
        if hasattr(e, "self_cuda_time_total") and e.self_cuda_time_total is not None:
            return float(e.self_cuda_time_total)
        if hasattr(e, "cuda_time_total") and e.cuda_time_total is not None:
            return float(e.cuda_time_total)
        return 0.0

    # Treat any all-reduce / NCCL op as communication
    comm = 0.0
    total = 0.0
    for e in events:
        t = get_cuda_time(e)
        total += t
        key = str(e.key).lower()
        if "all_reduce" in key or "nccl" in key:
            comm += t

    compute = max(total - comm, 0.0)
    # return in milliseconds
    return compute / 1e3, comm / 1e3


def train(rank: int, world_size: int, args):
    setup(rank, world_size, args.backend, args.dist_url)
    device = torch.device(f"cuda:{rank}")

    loader, sampler = get_dataloaders(
        args.batch_size,
        world_size,
        rank,
        args.data_root,
        args.workers,
    )

    # Use the Lab 2 ResNet-18 architecture instead of torchvision's model.
    model = ResNet18(num_classes=10).to(device)
    model = DDP(model, device_ids=[rank])

    opt = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    if args.profile and rank == 0:
        prof = profile_comm_compute()
        prof_ctx = prof
    else:
        prof = None
        prof_ctx = nullcontext()

    epoch2_time = None          # full epoch-2 time (incl. data loading) for Q2
    epoch2_compute = None       # compute-only epoch-2 time (excl. data loading) for Q1
    epoch5_loss = None
    epoch5_acc = None

    for epoch in range(1, args.epochs + 1):
        with prof_ctx as p:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss, acc, compute_time = run_epoch(
                model,
                loader,
                opt,
                scaler,
                device,
                epoch,
                sampler=sampler,
                profiler=p if (args.profile and rank == 0) else None,
            )
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            elapsed = t1 - t0  # full epoch time including data loading

        # average across ranks
        metrics = torch.tensor([elapsed, loss, acc, compute_time], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        elapsed_avg, loss_avg, acc_avg, compute_time_avg = metrics.tolist()

        if rank == 0:
            if epoch == 2:
                epoch2_time = elapsed_avg
                epoch2_compute = compute_time_avg
            if epoch == 5:
                epoch5_loss, epoch5_acc = loss_avg, acc_avg

            print(f"[Epoch {epoch}/{args.epochs}] "
                  f"time={elapsed_avg:.3f}s loss={loss_avg:.4f} acc={acc_avg:.4f}")

            if args.profile and prof is not None:
                compute_ms, comm_ms = summarize_profiler(prof)
                pct = 100.0 * comm_ms / (comm_ms + compute_ms + 1e-9)
                print(f"  profiler: compute={compute_ms:.2f} ms, "
                      f"comm={comm_ms:.2f} ms, comm_pct={pct:.1f}%")

    if rank == 0:
        print("\n=== Lab 5 Part A Summary ===")
        if epoch2_time is not None:
            print(f"Epoch 2 full time (includes data loading; use for Q2 tables): "
                  f"{epoch2_time:.3f} s")
        if epoch2_compute is not None:
            print(f"Epoch 2 compute-only time (excludes data loading; use for Q1): "
                  f"{epoch2_compute:.3f} s")
        if epoch5_loss is not None:
            print(f"Epoch 5 loss/acc (Q4.1): {epoch5_loss:.4f} / {epoch5_acc:.4f}")

    cleanup()


def parse_args():
    p = argparse.ArgumentParser(description="Lab 5 DDP timing script")
    p.add_argument("--batch-size", type=int, required=True, help="Batch size per GPU")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--use-amp", action="store_true", help="Use mixed precision")
    p.add_argument("--profile", action="store_true", help="Enable CUDA profiler")
    p.add_argument("--backend", type=str, default="nccl")
    p.add_argument("--dist-url", type=str, default="env://")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # torch.distributed.run / torchrun will set these
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))

    train(rank, world_size, args)

