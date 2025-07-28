from PIL import Image
from torch.utils.data import Dataset
import os

class myDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.input_dir = os.path.join(root_dir, 'input')
        self.gt_dir = os.path.join(root_dir, 'gt')
        self.image_names = os.listdir(self.input_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.image_names[idx])
        gt_path = os.path.join(self.gt_dir, self.image_names[idx])

        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.transform:
            input_img = self.transform(input_img)
            gt_img = self.transform(gt_img)

        return input_img, gt_img
