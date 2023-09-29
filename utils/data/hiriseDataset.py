from torch.utils.data import Dataset, random_split, DataLoader
import cv2
import os 
import pandas as pd

class hiriseImageDataset(Dataset):
  """Custom Pytorch Image Dataset class for Hirise Data

    Args:
        Dataset: Pytorch Dataset
  """
  def __init__(self, annotations_file, img_dir, classmap_file,  transform=None, target_transform=None):
      self.img_labels = pd.read_csv(annotations_file, delimiter=' ',header=None)
      self.class_map = pd.read_csv(classmap_file, header=None, names=['class', 'name'])
      self.img_dir = img_dir
      self.transform = transform
      self.target_transform = target_transform
      
  
  def __len__(self):
      return len(self.img_labels)

  def __getitem__(self, idx):
      img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
      image = cv2.imread(img_path)
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      label = self.img_labels.iloc[idx, 1]
      class_label_map = self.class_map.loc[self.class_map['class'] == label].name.item()
      if self.transform:
          image = self.transform(image=gray_image)["image"]
      if self.target_transform:
          label = self.target_transform(label)
      return image, label, class_label_map