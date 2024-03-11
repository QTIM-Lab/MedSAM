import os, pdb
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, Grayscale, Lambda

import torch
from torch.utils.data import Dataset

def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


class SAMDataset(Dataset):
    def __init__(self, dataset, processor, is_grayscale=False):
        """
        dataset - Path to csv containing paths to dataset.
                  From the python -m monai.bundle run command in README.md
        processor - Designed for SAMProcessor.
        """
        if isinstance(dataset, str):
            self.dataset_dir = os.path.dirname(dataset)
            self.dataset = pd.read_csv(os.path.join(dataset))
        else:
            self.dataset = dataset
        self.processor = processor
        self.is_grayscale = is_grayscale

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, image_path, is_grayscale=True, resize=(256, 256)):
        image = Image.open(os.path.join(self.dataset_dir, image_path))
        image = image.convert('RGB')
        # image = image.convert('RGB')
        img_transform = Compose([
            Resize(resize),
            Grayscale() if is_grayscale else Lambda(lambda x: x),
            # ToTensor() # it will convert to tensor and scale between 0-1
            Lambda(lambda x: torch.from_numpy(np.array(x))),
            
        ])
        image = img_transform(image)
        return image

    def __getitem__(self, idx):
        if isinstance(self.dataset, pd.DataFrame):
            item = self.dataset.iloc[idx]
            image = item['image_path_orig']
            label = item['labels']
            image = self._load_image(image, is_grayscale=False, resize=(256, 256))
            label = self._load_image(label, is_grayscale=True, resize=(256, 256))
            label = (label > 0).float()
        else:            
            item = self.dataset[idx]
            image = item["image"]
            label = np.array(item["label"])

        # get bounding box prompt
        # print(image)
        # print(type(image))
        # print(label.shape)
        prompt = get_bounding_box(label.squeeze())

        # prepare image and prompt for the model
        print(image.min(), image.max())
        # pdb.set_trace()
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        # print(image.min(), image.max())

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        # inputs = {'image': image, 'label': label}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = label

        return inputs