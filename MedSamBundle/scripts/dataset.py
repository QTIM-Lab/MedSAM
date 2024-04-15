import os, pdb
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, Grayscale, Lambda
from transformers import SamProcessor

import torch
from torch.utils.data import Dataset

# from MedSamBundle.scripts.transforms import MedSamTransform
from scripts.transforms import MedSamTransform

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
    def __init__(self, dataset, processor=SamProcessor.from_pretrained("facebook/sam-vit-base"), is_grayscale=False, mode="train"):
        """
        dataset - Path to csv containing paths to dataset.
                  From the python -m monai.bundle run command in README.md
        processor - Designed for SAMProcessor.
        """
        # pdb.set_trace()
        self.mode = mode
        if isinstance(dataset, list):
            self.dataset = dataset
            self.input = "datastore"            
        elif isinstance(dataset, str):
            self.dataset_dir = os.path.dirname(dataset)
            if dataset.find(".csv") != -1:
                self.dataset = pd.read_csv(os.path.join(dataset))
                self.input = "csv"
            else:
                self.dataset = dataset
                self.input = "image"
        else:
            self.dataset = dataset
            self.input = "dataset"
        self.processor = processor
        self.is_grayscale = is_grayscale

    def __len__(self):
        if self.input == "csv" or self.input == "dataset" or self.input == "datastore":
            return len(self.dataset)
        elif self.input == "file":
            return 1
        elif self.input == "image":
            return 1
        else:
            return "incorrect data likely sent to this class - BB"


    def _load_image(self, image_path, is_grayscale=False):
        if self.input == "file":
            path_to_image = os.path.join(image_path)
        if self.input == "datastore":
            path_to_image = image_path
        else:
            path_to_image = os.path.join(self.dataset_dir, image_path)
        image = Image.open(path_to_image)
        shape = np.array(image.convert('RGB')).shape
        h = shape[0]
        w = shape[1]
        image_shape = (h,w)
        # Initialize the transform with the desired configuration
        # transform = MedSamTransform(is_grayscale=is_grayscale, resize=image_shape)
        # pdb.set_trace()
        transform = MedSamTransform(is_grayscale=is_grayscale, resize=image_shape)
        image = transform(image)
        return image


    def __getitem__(self, idx):
        # pdb.set_trace()
        if isinstance(self.dataset, str):
            image = self.dataset
            label = image
            image = self._load_image(image)
            label = self._load_image(label, is_grayscale=True)
            label = (label > 0).float()
        elif isinstance(self.dataset, pd.DataFrame):
            item = self.dataset.iloc[idx]
            image = item['image_path_orig']
            label = item['labels']
            image = self._load_image(image)
            label = self._load_image(label, is_grayscale=True)
            label = (label > 0).float()
        else:         
            item = self.dataset[idx]
            image = item["image"]
            label = item["label"]
            image = self._load_image(image)
            label = self._load_image(label, is_grayscale=True)
            label = (label > 0).float()

        prompt = get_bounding_box(label.squeeze())
        # prepare image and prompt for the model
        print(image.min(), image.max())
        # pdb.set_trace()
        # prompt = [600, 600, 1400, 1400]
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        # inputs.keys()        
        if self.mode == 'train':
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        print(f"""
        Expect:
        'train'
        inputs['pixel_values'].shape -> [channels, h, w]
        inputs['original_sizes'].shape -> [shape] -> shape: [h, w]
        inputs['reshaped_input_sizes'].shape -> [shape] -> shape: [h, w]
        inputs['input_boxes'].shape -> [num_boxes, bounding_box] -> bounding_box: [x1, y1, x2, y2]
        'infer'
        inputs['pixel_values'].shape -> [batch, channels, h, w]
        inputs['original_sizes'].shape -> [batch, shape] -> shape: [h, w]
        inputs['reshaped_input_sizes'].shape -> [batch, shape] -> shape: [h, w]
        inputs['input_boxes'].shape -> [batch, num_boxes, bounding_box] -> bounding_box: [x1, y1, x2, y2]
        mode: '{self.mode}'
        inputs['pixel_values'].shape: {inputs['pixel_values'].shape}
        inputs['original_sizes'].shape: {inputs['original_sizes'].shape}
        inputs['reshaped_input_sizes'].shape: {inputs['reshaped_input_sizes'].shape}
        inputs['input_boxes'].shape: {inputs['input_boxes'].shape}
        """)

        # pdb.set_trace()
        # add ground truth segmentation
        torch.unique(label)
        inputs["ground_truth_mask"] = label
        # pdb.set_trace()

        return inputs