from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


# Define transforms for data augmentation
train_transform = transforms.Compose([
    transforms.Resize((448, 448)),  # Resize the image to 448x448
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])


def load_railway(path, train_size, test_size):
    # railway_train = datasets.MNIST(path, train=True, download=True, transform=train_transform)
    # train_loader = torch.utils.data.DataLoader(railway_train, batch_size=train_size, shuffle=True)
    railway_test = ImageFolder(root=path, transform=train_transform)
    test_loader = DataLoader(railway_test, batch_size=test_size, shuffle=False, num_workers=0)
    return test_loader

