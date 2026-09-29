"""A small PyTorch CNN trained on MNIST: the test workload for `train_telemetry`.

The script deliberately has no telemetry imports. The collector has to pick up its
hardware usage, run metadata, console output and log records from the outside:

    python -m train_telemetry train_mnist.py --epochs 1

Run it on its own to check the workload:

    python train_mnist.py --epochs 1                          # real MNIST (downloads ~12 MB)
    python train_mnist.py --data fake --limit-batches 50      # offline smoke test
    python train_mnist.py --data fake --crash-at-step 20      # failure path
    python train_mnist.py --num-workers 2                     # DataLoader child processes

It exercises each thing the collector must capture: `print` to stdout, stdlib
`logging` to stderr, a Python warning, CPU/GPU load, optional worker processes,
and (with `--crash-at-step`) an uncaught exception with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

log = logging.getLogger('train_mnist')

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class Net(nn.Module):
    """The classic PyTorch MNIST example: two conv layers, two linear layers (~1.2M parameters)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.dropout2(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)


def load_dataset(kind: str, data_dir: Path, train: bool) -> Dataset[tuple[torch.Tensor, int]]:
    """Load MNIST or deterministic synthetic images."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    if kind == 'fake':
        # Random 28x28 images with MNIST's shape and size, for machines that can't download MNIST.
        return datasets.FakeData(
            size=60_000 if train else 10_000,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=0 if train else 1,
        )
    return datasets.MNIST(str(data_dir), train=train, download=True, transform=transform)


def pick_device(requested: str) -> torch.device:
    """Choose CUDA, MPS, or CPU unless explicitly selected."""
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    global_step: int,
) -> int:
    """Train one epoch and return the cumulative step count."""
    model.train()
    for batch_idx, (images, targets) in enumerate(loader):
        if args.limit_batches and batch_idx >= args.limit_batches:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(images), targets)
        # torch leaves optional backward parameters untyped.
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()
        global_step += 1

        if args.crash_at_step and global_step >= args.crash_at_step:
            raise RuntimeError(f'Simulated training failure at step {global_step}')
        if batch_idx % args.log_every == 0:
            print(f'epoch {epoch} batch {batch_idx}/{len(loader)} loss {loss.item():.4f}', flush=True)
    return global_step


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device, limit_batches: int
) -> tuple[float, float]:
    """Return mean test loss and classification accuracy."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for batch_idx, (images, targets) in enumerate(loader):
        if limit_batches and batch_idx >= limit_batches:
            break
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        total_loss += F.nll_loss(output, targets, reduction='sum').item()
        correct += (output.argmax(dim=1) == targets).sum().item()
        seen += len(targets)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse workload options independently of the telemetry runner."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', choices=['mnist', 'fake'], default='mnist')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1.0, help='Adadelta learning rate')
    parser.add_argument('--device', default='auto', help="'auto', 'cpu', 'cuda', 'cuda:1', 'mps'")
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker processes')
    parser.add_argument('--limit-batches', type=int, default=0, help='Stop each epoch after N batches (0 = all)')
    parser.add_argument('--log-every', type=int, default=100, help='Print progress every N batches')
    parser.add_argument('--crash-at-step', type=int, default=0, help='Raise an error at this step (0 = never)')
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the unchanged training workload."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    # torch leaves the seed parameter untyped.
    torch.manual_seed(args.seed)  # pyright: ignore[reportUnknownMemberType]

    device = pick_device(args.device)
    if device.type == 'cpu':
        warnings.warn('No GPU found, training on CPU', RuntimeWarning, stacklevel=1)
    log.info('Training on %s with torch %s', device, torch.__version__)

    train_loader = DataLoader(
        load_dataset(args.data, args.data_dir, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    test_loader = DataLoader(
        load_dataset(args.data, args.data_dir, train=False),
        batch_size=1000,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    log.info('Model has %d parameters', sum(p.numel() for p in model.parameters()))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        global_step = train_epoch(model, train_loader, optimizer, device, epoch, args, global_step)
        test_loss, accuracy = evaluate(model, test_loader, device, args.limit_batches)
        scheduler.step()
        log.info(
            'Epoch %d done in %.1fs: test loss %.4f, accuracy %.2f%%',
            epoch,
            time.perf_counter() - started,
            test_loss,
            accuracy * 100,
        )
    print(f'Finished {args.epochs} epoch(s), {global_step} steps', file=sys.stderr)


if __name__ == '__main__':
    main()
