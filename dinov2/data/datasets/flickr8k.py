# datasets.py
import os
import random
from PIL import Image

from torch.utils.data import Dataset

class Flickr8k(Dataset):
    """
    A minimal Flickr8k dataset that returns (image, caption) pairs.

    Expecting a folder structure like:
      root/
        Images/
          1000268201_693b08cb0e.jpg
          ...
        captions.txt
    Where 'captions.txt' has lines like:
      image_name.jpg,Some caption text here
      ...
    """
    def __init__(self,
                 root: str,
                 split: str = "TRAIN",
                 transform=None,
                 target_transform=None):
        """
        root: path to the flickr8k dataset folder
        transform: image transform
        target_transform: transform for the text caption if needed
        split: if you want to split into train/val, you can parse it here, or ignore.
        """
        super().__init__()
        self.root = root
        self.split = split.upper()
        self.transform = transform
        self.target_transform = target_transform

        # We assume there's a "captions.txt" in root
        caption_file = os.path.join(self.root, "captions.txt")
        self.samples = []  # will store tuples (img_name, caption)

        with open(caption_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # split on the first comma
                img_name, caption = line.split(",", 1)
                img_name = img_name.strip()
                caption = caption.strip()
                self.samples.append((img_name, caption))

        # If you have a train/val split logic, apply it here (not mandatory for small sets)
        # e.g. if self.split == "TRAIN": self.samples = self.samples[:6000] 
        # else: self.samples = self.samples[6000:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption_str = self.samples[idx]
        img_path = os.path.join(self.root, "Images", img_name)
        # load the image
        with Image.open(img_path).convert("RGB") as img:
            if self.transform:
                img = self.transform(img)

        # optionally transform the text (rarely done, but you can)
        if self.target_transform:
            caption_str = self.target_transform(caption_str)

        return img, caption_str
