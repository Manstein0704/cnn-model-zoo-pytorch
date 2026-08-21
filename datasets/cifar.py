from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 


def get_dataloader(batch_size:int, root="./data"):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5))
    ])

    dataset_train = datasets.CIFAR10(root=root, transform=transform_train, download=True)
    dataset_test = datasets.CIFAR10(root=root, train=False, transform=transform_test, download=True)

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader