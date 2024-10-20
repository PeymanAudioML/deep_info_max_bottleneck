from typing import Optional, Tuple, Any, Iterator

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class MNISTDataLoader:
    """
    A comprehensive data loader for the MNIST dataset with customizable transformations and batching.

    This class handles downloading the MNIST dataset, applying a series of transformations, and providing
    data loaders for training or evaluation purposes. Each data sample consists of two differently
    transformed versions of the same image along with its label, suitable for contrastive learning tasks.

    Attributes:
        root (str): Root directory where the MNIST dataset will be stored.
        batch_size (int): Number of samples per batch to load.
        train (bool): If True, creates dataset from the training set, otherwise from the test set.
        download (bool): If True, downloads the dataset from the internet if it's not already present.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        transformations (transforms.Compose): A composed transform that applies a series of augmentations and tensor conversion.
        dataloader (DataLoader): The PyTorch DataLoader for batching and shuffling.
    """

    class CustomMNISTDataset(Dataset):
        """
        A custom Dataset for MNIST that returns two differently transformed images along with the label.

        This is particularly useful for tasks like contrastive learning, where multiple views of the same
        image are required.

        Attributes:
            mnist (datasets.MNIST): The MNIST dataset instance.
            transformations (transforms.Compose): The transformation pipeline applied to the images.
        """

        def __init__(self, mnist_dataset: datasets.MNIST, transformations: transforms.Compose) -> None:
            """
            Initializes the CustomMNISTDataset with the MNIST dataset and transformation pipeline.

            Args:
                mnist_dataset (datasets.MNIST): The MNIST dataset instance.
                transformations (transforms.Compose): The transformation pipeline to apply to the images.
            """
            self.mnist = mnist_dataset
            self.transformations = transformations

        def __len__(self) -> int:
            """
            Returns the total number of samples in the dataset.

            Returns:
                int: Number of samples.
            """
            return len(self.mnist)

        def __getitem__(self, idx: int) -> Tuple[Any, Any, int]:
            """
            Retrieves the sample at the specified index, applies transformations twice, and returns both
            transformed images along with the label.

            Args:
                idx (int): Index of the sample to retrieve.

            Returns:
                Tuple[Any, Any, int]: A tuple containing two transformed images and the corresponding label.
            """
            image, label = self.mnist[idx]
            image1 = self.transformations(image)
            image2 = self.transformations(image)
            return image1, image2, label

    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        train: bool = True,
        download: bool = True,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False,
        transform_params: Optional[dict] = None
    ) -> None:
        """
        Initializes the MNISTDataLoader with specified parameters and prepares the DataLoader.

        Args:
            root (str): Root directory for storing the MNIST dataset.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            train (bool, optional): If True, loads the training set; otherwise, the test set. Defaults to True.
            download (bool, optional): If True, downloads the dataset if not present. Defaults to True.
            shuffle (bool, optional): If True, shuffles the data every epoch. Defaults to True.
            num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
            pin_memory (bool, optional): If True, uses pinned memory for faster data transfer to GPU. Defaults to False.
            transform_params (dict, optional): Dictionary of parameters to customize transformations.
                                              Expected keys:
                                                - rotation_degrees (float or int)
                                                - affine_translate (tuple of float)
                                                - perspective_distortion (float)
                                                - erasing_probability (float)
                                              If None, default values are used.
        """
        self.root: str = root
        self.batch_size: int = batch_size
        self.train: bool = train
        self.download: bool = download
        self.shuffle: bool = shuffle
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory

        # Set default transformation parameters if not provided
        if transform_params is None:
            transform_params = {
                'rotation_degrees': 30,
                'affine_translate': (0.1, 0.1),
                'perspective_distortion': 0.2,
                'erasing_probability': 0.5
            }

        # Define the transformation pipeline
        self.transformations: transforms.Compose = transforms.Compose([
            transforms.RandomRotation(degrees=transform_params['rotation_degrees']),
            transforms.RandomAffine(
                degrees=0,
                translate=transform_params['affine_translate']
            ),
            transforms.RandomPerspective(
                distortion_scale=transform_params['perspective_distortion'],
                p=0.5
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=transform_params['erasing_probability']
            )
        ])

        # Initialize the MNIST dataset without any transformations
        self.mnist_dataset: datasets.MNIST = datasets.MNIST(
            root=self.root,
            train=self.train,
            download=self.download,
            transform=None  # Transformations are handled in the custom Dataset
        )

        # Initialize the custom dataset
        self.dataset: Dataset = self.CustomMNISTDataset(
            mnist_dataset=self.mnist_dataset,
            transformations=self.transformations
        )

        # Initialize the DataLoader
        self.dataloader: DataLoader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def get_dataloader(self) -> DataLoader:
        """
        Retrieves the DataLoader for the MNIST dataset.

        Returns:
            DataLoader: The DataLoader instance for batching and shuffling the dataset.
        """
        return self.dataloader

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.

        Returns:
            int: Number of batches per epoch.
        """
        return len(self.dataloader)

    def __iter__(self) -> Iterator[Tuple[Any, Any, int]]:
        """
        Returns an iterator over the DataLoader.

        Returns:
            Iterator[Tuple[Any, Any, int]]: An iterator over the DataLoader yielding batches of data.
        """
        return iter(self.dataloader)


def main() -> None:
    """
    Demonstrates the usage of the MNISTDataLoader class by fetching a batch of data and displaying its shape.
    """
    # Configuration parameters
    data_path: str = "/Users/imanghamarian/Desktop/codes/1_InfoMaxIB/data"  # Replace with your desired path
    batch_size: int = 4
    is_train: bool = True
    should_download: bool = True

    # Initialize the data loader
    mnist_loader = MNISTDataLoader(
        root=data_path,
        batch_size=batch_size,
        train=is_train,
        download=should_download,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Retrieve the DataLoader
    dataloader: DataLoader = mnist_loader.get_dataloader()

    # Fetch a single batch
    try:
        images1, images2, labels = next(iter(dataloader))
        print(f"Batch of images1 shape: {images1.shape}")  # Expected: [batch_size, channels, height, width]
        print(f"Batch of images2 shape: {images2.shape}")  # Expected: [batch_size, channels, height, width]
        print(f"Batch of labels shape: {labels.shape}")    # Expected: [batch_size]
    except StopIteration:
        print("The DataLoader is empty. Please check the dataset and transformations.")
    except ValueError as ve:
        print(f"ValueError encountered: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Optionally, iterate through the entire dataset
    for batch_idx, (images1, images2, labels) in enumerate(dataloader):
        # Insert training or evaluation code here
        print(f"Batch {batch_idx + 1}:")
        print(f"  images1 shape: {images1.shape}")
        print(f"  images2 shape: {images2.shape}")
        print(f"  labels: {labels}")
        # For demonstration, we'll break after the first batch
        break


if __name__ == "__main__":
    main()
