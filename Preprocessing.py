import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from PIL import Image


def load_all(path):
    folders = os.listdir(path)
    X = []
    Y = []
    for folder in folders:
        x, y = data_loading(path + "//" + folder, folders.index(folder))
        X.extend(x)
        Y.extend(y)
    return X, Y


def data_loading(path, label):
    """Loads all file paths from a directory and assigns a single label.

    This function iterates through a given directory, collects the
    full file paths of all items within it, and creates a corresponding
    PyTorch tensor of a uniform label.

    Note: This function recursively lists items, it does not
    check if the items are files or sub-directories.

    Args:
        path (str): The string path to the directory (e.g., './data/cats').
        label (int): The single class label to assign to all items in
            this directory (e.g., 0 for 'cats').

    Returns:
        tuple[list[str], torch.Tensor]: A tuple containing two elements:
            - A list of full string paths for each item in the directory.
            - A 1D PyTorch tensor containing the same label repeated
              for each item.
    """
    images = []
    for i in os.listdir(path):
        full_path = os.path.join(path, i)
        images.append(full_path)

    labels = torch.tensor([label] * len(images))
    return images, labels


class CustomDataset(Dataset):
    """A custom dataset for loading images from file paths.

    This dataset expects a list of image file paths and a corresponding
    list or tensor of labels. It reads images using torchvision.

    Attributes:
        images (list[str]): A list of file paths to the images.
        labels (torch.Tensor): A tensor of labels corresponding to the images.
    """

    def __init__(self, images, labels):
        """
        Initializes the dataset.

        Args:
            images (list[str]): A list of file paths to the images.
            labels (torch.Tensor): A tensor of labels for the images.
        """
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),

        ])

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetches one sample (image and label) from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The image as a 3D tensor (C, H, W) of type float32.
                - The corresponding label.
        """
        sample = Image.open(self.images[idx]).convert('RGB')
        transformed_image = self.transform(sample)
        x = transformed_image.float()

        y = self.labels[idx]
        return x, y


def get_dataloader(X, Y, batch_size=8, shuffle=True):
    """Creates a PyTorch DataLoader from image paths and labels.

    This function wraps the CustomDataset with a DataLoader for
    easy batching, shuffling, and iteration during model training
    or evaluation.

    Args:
        X (list[str]): A list of file paths to the images.
        Y (torch.Tensor): A tensor of labels for the images.
        batch_size (int, optional): The number of samples per batch.
            Defaults to 8.
        shuffle (bool, optional): Whether to shuffle the data at the
            beginning of each epoch. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: A DataLoader instance configured
            with the custom dataset.
    """
    dataset = CustomDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def feature_extraction(data_loader, model,device=None):
    model.eval()
    with torch.no_grad():
        extracted_features = []
        progress_bar = tqdm(data_loader, total=len(data_loader), unit='batch', unit_scale=True, desc='Extracting features',
                            dynamic_ncols=True)
        for x,y in progress_bar:
            x=x.to(device)
            extracted_features.extend(model(x).detach().cpu().numpy())
    return extracted_features
