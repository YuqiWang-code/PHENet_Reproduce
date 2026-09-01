from torch.utils.data import DataLoader

from dataloaders.datasets.CD_dataset_heightmap import CDDataSet


def make_data_loaders(args):
    kwargs = {
        "num_workers": args.workers,
        "pin_memory": bool(args.cuda),
        "persistent_workers": args.workers > 0,
    }
    train_set = CDDataSet(args, split="train")
    val_set = CDDataSet(args, split=args.val_split)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    return train_loader, val_loader, 2
