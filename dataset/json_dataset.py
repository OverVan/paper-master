import os
import json
from PIL import Image
from torch.utils.data import Dataset

from utils import log
from .dataset import register
from .transform import TransformLoader


@register("json_dataset")
class JsonDataset(Dataset):
    def __init__(self, data_file, transform_list, image_size=80, phase="train"):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = TransformLoader(image_size).get_composed_transform(transform_list)
        self.labels = self.meta["image_labels"]
        log("dataset loaded, {} classes {} samples".format(len(self.meta["label_names"]), len(self.meta['image_names'])), phase)

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        label = self.meta['image_labels'][i]
        return img, label

    def __len__(self):
        return len(self.meta['image_names'])