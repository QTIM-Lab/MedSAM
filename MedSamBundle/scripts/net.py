import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import SegformerForSemanticSegmentation
import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F


# BB
import numpy as np
from transformers import SamModel, SamProcessor
from PIL import Image

class MyCustomNet(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()

        # Load the MedSAM model
        pretrained_model = SamModel.from_pretrained(
            "flaviagiammarino/medsam-vit-base", local_files_only=False
        )# .to(device)
        config = pretrained_model.config
        self._model = pretrained_model
        self._processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

        # Load the pretrained weights on the class itself
        self._model.load_state_dict(pretrained_model.state_dict())

    def forward(self, raw_image, input_boxes = [600., 600., 1200., 1400.]):
        # Preprocess the input using the MedSAM processor
        raw_image_PIL = Image.fromarray(np.array(raw_image[0]))
        inputs = self._processor(raw_image_PIL, input_boxes=[[input_boxes]], return_tensors="pt")

        # Perform inference with the MedSAM model
        outputs = self._model(**inputs, multimask_output=False)

        # Interpolate the output if necessary
        # pdb.set_trace()
        return outputs

    def state_dict(self, *args, **kwargs):
        """Removes the _model. prefix from the keys of the state dict."""
        state_dict = super().state_dict(*args, **kwargs)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[len("_model."):]
            new_state_dict[new_key] = value
        return new_state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Adds the _model. prefix to the keys of the state dict."""
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = "_model." + key
            new_state_dict[new_key] = value
        return super().load_state_dict(new_state_dict, *args, **kwargs)
