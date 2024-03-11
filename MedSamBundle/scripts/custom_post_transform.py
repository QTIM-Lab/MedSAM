import pdb
import importlib
import monai.transforms as transforms
import torch
from transformers import SamProcessor
from PIL import Image

class CustomPostTransform(transforms.Compose):
    def __init__(self, processor_name, from_pretrained="flaviagiammarino/medsam-vit-base"):
        super().__init__([])
        self.processor_name = processor_name
        self.from_pretrained = from_pretrained
        self.processor = self.load_processor()

    def load_processor(self):
        try:
            processor_module = importlib.import_module('transformers')
            processor_class = getattr(processor_module, self.processor_name)
            return processor_class.from_pretrained(self.from_pretrained)
        except (ImportError, AttributeError):
            raise ValueError(f"Failed to load processor class: {self.processor_name}")

    def __call__(self, data):
        # Implement your custom post-processing logic here
        raw_image = Image.fromarray(data['image'])
        input_boxes = [600., 600., 1200., 1400.]
        inputs = self.processor(raw_image, input_boxes=[[input_boxes]], return_tensors="pt")
        # pdb.set_trace()
        data['pred']['pred_masks'] = data['pred']['pred_masks'].unsqueeze(2)
        # convenience for pdb - delete later
        # probs = self.processor.image_processor.post_process_masks(data['pred']['pred_masks'].sigmoid().cpu(),inputs["original_sizes"].cpu(),inputs["reshaped_input_sizes"].cpu(),binarize=False)
        probs = self.processor.image_processor.post_process_masks(
            data['pred']['pred_masks'].sigmoid().cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            binarize=False
        )
        # pdb.set_trace()
        data["probs"] = probs[0]  # Assuming you want to replace the original "pred_masks"
        return data