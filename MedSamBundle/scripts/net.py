import pdb
import torch
import torch.nn as nn

# BB
import numpy as np
from transformers import SamModel, SamProcessor
from PIL import Image

class MedSamNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Load the MedSAM model
        self._model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base", local_files_only=False)
        self._processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

    def forward(model, device, dataset_instance):
        model.to(device)
        type(dataset_instance)
        pdb.set_trace()
        dataset_instance['pixel_values'].shape
        dataset_instance['original_sizes'].shape
        dataset_instance['reshaped_input_sizes'].shape
        dataset_instance['input_boxes'].shape
        # batch = next(iter(dataset_instance))
        pv = dataset_instance["pixel_values"]
        ib = dataset_instance["input_boxes"]
        pv.shape
        ib.shape
        type(pv)
        type(ib)
        # forward pass
        outputs = model(pixel_values=pv.to(device), input_boxes=ib.to(device), multimask_output=False)
        logits = outputs.pred_masks
        l_sig = logits.sigmoid().cpu()
        os = dataset_instance['original_sizes']
        ris = dataset_instance['reshaped_input_sizes']
        processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
        probs = processor.image_processor.post_process_masks(l_sig, os, ris, binarize=False)
        binary_mask = (probs[0] > 0.85).int() * 255

        return binary_mask