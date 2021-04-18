import torch
from torchvision import datasets, transforms

NORMALIZATION_MEAN = 0.1307
NORMALIZATION_STD = 0.3081
normalizer = transforms.Normalize((NORMALIZATION_MEAN,), (NORMALIZATION_STD,))
transform = transforms.Compose([
    transforms.ToTensor(),
    normalizer
])
mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('../data', train=False, transform=transform)
raw_mnist_test = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())


def mnist_train_dataloader(batch_size, shuffle=True):
    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    train_loader = torch.utils.data.DataLoader(mnist_train, **train_kwargs)
    return train_loader


def mnist_test_dataloader(batch_size, shuffle=True):
    test_kwargs = {'batch_size': batch_size, 'shuffle': shuffle}
    test_loader = torch.utils.data.DataLoader(mnist_test, **test_kwargs)
    return test_loader


def raw_mnist_test_dataloader(batch_size, shuffle=True):
    raw_test_kwargs = {'batch_size': batch_size, 'shuffle': True}
    raw_test_loader = torch.utils.data.DataLoader(raw_mnist_test, **raw_test_kwargs)
    return raw_test_loader
