
import warnings
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def load_cifar_datasets(
    data_dir: str,
    num_devices: int = 100,
    frac_random: float = 0.2,
    batch_size_train: int = 32,
    batch_size_test: int = 128,
    shuffle_train: bool = True,
):
    # 0) Define exactly the AirComp transformation pipelines
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 1) Load full CIFAR10 via torchvision (handles download & pickles)
    full_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    full_test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    # 2) Build the same non‑IID index splits (rand + global‑sort shard)
    N = len(full_train)  # 50000
    local_per_dev = N // num_devices
    rand_per_dev = int(local_per_dev * frac_random)
    shard_size = local_per_dev - rand_per_dev

    if N % num_devices != 0:
        warnings.warn(f"{N} not divisible by {num_devices}, dropping {N%num_devices}")

    # 2a) shuffle all indices, carve off the random chunk & reshape
    all_idxs = np.arange(N)
    np.random.shuffle(all_idxs)
    rand_idxs = all_idxs[: num_devices * rand_per_dev]
    rem_idxs = all_idxs[num_devices * rand_per_dev :]
    rand_blocks = rand_idxs.reshape(num_devices, rand_per_dev)

    # 2b) global sort of the remainder by label → shard_blocks
    rem_labels = np.array(full_train.targets)[rem_idxs]
    sorted_rem = rem_idxs[np.argsort(rem_labels)]
    shards = sorted_rem[: num_devices * shard_size]
    shard_blocks = shards.reshape(num_devices, shard_size)

    # 3) Create one DataLoader per device via Subset(full_train, idxs)
    device_loaders = []
    for dev in range(num_devices):
        idxs = np.concatenate([rand_blocks[dev], shard_blocks[dev]])
        subset = Subset(full_train, idxs.tolist())

        loader = DataLoader(
            subset,
            batch_size=batch_size_train,
            shuffle=shuffle_train,
            num_workers=2,
            pin_memory=True
        )
        device_loaders.append(loader)

    # 4) Single global test loader
    test_loader = DataLoader(
        full_test,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return device_loaders, test_loader