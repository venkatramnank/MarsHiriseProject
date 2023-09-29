from torch.utils.data import Dataset, random_split, DataLoader
from utils.data.hiriseDataset import hiriseImageDataset

class hiriseDataLoader(DataLoader):
    """Custom data loader for hirise dataset

    """
    def __init__(self,  annotations_file, img_dir, classmap_file,  transform=None, target_transform=None, batch_size=64, shuffle=True):
        dataset = hiriseImageDataset(annotations_file, img_dir, classmap_file,  transform=transform, target_transform=target_transform)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

