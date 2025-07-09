from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def setup_mnist_loader(args, val_split=0.2,):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=True)

    val_size = int(val_split * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    return loader