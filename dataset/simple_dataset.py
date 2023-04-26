import os
from PIL import Image
from torch.utils.data import Dataset

from utils import log
from .transform import TransformLoader
from .dataset import register


@register("simple_dataset")
class SimpleDataset(Dataset):
    def __init__(self, tail_path, transform_list, image_size=80, phase="train"):
        super(SimpleDataset, self).__init__()
        root_dir = "../datasets"
        self.dataset_dir = os.path.join(root_dir, tail_path)
        self.dataset, self.labels2inds, self.labelIds, self.labels = self._process_dir()
        self.transform = TransformLoader(image_size).get_composed_transform(transform_list)
        log("dataset {} loaded, {} classes {} samples".format(tail_path, len(self.labelIds), len(self.dataset)), phase)
        
    def _process_dir(self):
        # 固定顺序好复现正确率
        cat_container = sorted(os.listdir(self.dataset_dir))
        # 根据列表生成字典
        cats2label = {cat: label for label, cat in enumerate(cat_container)}
        # 各图片路径及其标签
        dataset = []
        # 全体图片的标签（有重复）
        labels = []
        for cat in cat_container:
            # 固定顺序好复现正确率
            for img_path in sorted(os.listdir(os.path.join(self.dataset_dir, cat))):
                if '.jpg' not in img_path:
                    continue
                label = cats2label[cat]
                dataset.append((os.path.join(self.dataset_dir, cat, img_path), label))
                labels.append(label) 
        # 各标签及其下标列表
        labels2inds = {}
        for idx, label in enumerate(labels):
            if label not in labels2inds:
                labels2inds[label] = []
            labels2inds[label].append(idx)
        # 全体标签（不重复）
        labelIds = sorted(labels2inds.keys())
        return dataset, labels2inds, labelIds, labels

    def __getitem__(self, index):
        image_path = self.dataset[index][0]
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        label = self.dataset[index][1]
        return img, label

    def __len__(self):
        return len(self.dataset)