from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split
import glob
import torch
import os


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


def setup_cifar_loader(args, val_split=0.2):

    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )

    # train_transform = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.RandomCrop(32, padding=4),     
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.RandomRotation(15),
    #     transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, transform=train_transform, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, transform=test_transform, download=True)

    train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=val_split, random_state=42)
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, transform=test_transform, download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    return loader


def create_dataloader_from_chunks(data_dir, device, batch_size, max_chunks=None):

    print('chunky cheese * ' * 50)
    datasets = []
    
    if os.path.isfile(data_dir):

        ckpt = torch.load(data_dir, map_location=device)
        activations = ckpt.get("fc1_activations_norm", ckpt.get("fc1"))

        if activations is None:
            raise KeyError("Data file must contain 'fc1' or 'fc1_activations_norm' key.")

        return DataLoader(TensorDataset(activations.reshape(-1, hidden_dim)), batch_size=batch_size, shuffle=True)
    
    chunk_pattern = os.path.join(data_dir, "*batch*.pt") if os.path.isdir(data_dir) else data_dir + "*batch*.pt"
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files:
        chunk_pattern = os.path.join(data_dir, "*.pt") if os.path.isdir(data_dir) else data_dir + "*.pt"
        chunk_files = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found matching pattern")
    
    if max_chunks:
        chunk_files = chunk_files[:max_chunks]
    
    print(f"Creating datasets from {len(chunk_files)} chunks")
    
    for chunk_file in chunk_files:
        try:
            ckpt = torch.load(chunk_file, map_location=device)
            activations = ckpt.get("fc1_activations_norm", ckpt.get("fc1"))
            if activations is not None:
                if activations.ndim == 3:
                    batch_size, seq_len, hidden_dim = activations.shape
                    activations = activations.reshape(-1, hidden_dim)
                datasets.append(TensorDataset(activations))
        except Exception as e:
            print(f"Warning: Skipping {chunk_file}: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets created from chunks")
    
    combined_dataset = ConcatDataset(datasets)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)


def create_dataloader_with_labels(data_dir, device, batch_size, max_chunks=None):

    activation_datasets = []
    label_datasets = []
    
    if os.path.isfile(data_dir):
        ckpt = torch.load(data_dir, map_location=device)
        activations = ckpt.get("fc1_activations_norm", ckpt.get("fc1"))
        labels = ckpt.get("labels")
        
        if activations is None:
            raise KeyError("Data file must contain 'fc1', 'fc1_activations', or 'fc1_activations_norm' key.")
        
        if labels is not None:
            return DataLoader(TensorDataset(activations, labels), batch_size=batch_size, shuffle=True)
        else:
            return DataLoader(TensorDataset(activations), batch_size=batch_size, shuffle=True)
    
    chunk_pattern = os.path.join(data_dir, "*batch*.pt") if os.path.isdir(data_dir) else data_dir + "*batch*.pt"
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files:
        chunk_pattern = os.path.join(data_dir, "*.pt") if os.path.isdir(data_dir) else data_dir + "*.pt"
        chunk_files = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found matching pattern")
    
    if max_chunks:
        chunk_files = chunk_files[:max_chunks]
    
    print(f"Creating datasets from {len(chunk_files)} chunks")
    has_labels = True
    
    for chunk_file in chunk_files:
        try:
            ckpt = torch.load(chunk_file, map_location=device)
            activations = ckpt.get("fc1_activations_norm", ckpt.get("fc1"))
            labels = ckpt.get("labels")
            
            if activations is not None:
                activation_datasets.append(activations)
                if labels is not None:
                    label_datasets.append(labels)
                else:
                    has_labels = False
        except Exception as e:
            print(f"Warning: Skipping {chunk_file}: {e}")
    
    if not activation_datasets:
        raise ValueError("No valid datasets created from chunks")
    
    all_activations = torch.cat(activation_datasets, dim=0)
    
    if has_labels and len(label_datasets) == len(activation_datasets):
        all_labels = torch.cat(label_datasets, dim=0)
        combined_dataset = TensorDataset(all_activations, all_labels)
    else:
        print("Warning: Not all chunks have labels, returning activations only")
        combined_dataset = TensorDataset(all_activations)
    
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

