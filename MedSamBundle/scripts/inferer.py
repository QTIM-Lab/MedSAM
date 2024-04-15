import pdb
import torch
from monai.inferers import SimpleInferer
from transformers import SamModel, SamProcessor

class MedSamInferer(SimpleInferer):
    """
    MedSamInferer is a subclass of SimpleInferer.
    """

    def __init__(self) -> None:
        super().__init__()
        # Load the MedSAM model
        self._model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base", local_files_only=False)
        self._processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

    def forward(self, inputs, device):
        self._model.to(device)
        input_instance = inputs[0]
        pv = input_instance["pixel_values"]
        ib = input_instance["input_boxes"]
        # pv = input_instance["pixel_values"].reshape((1, 3, 1024, 1024))
        # ib = input_instance["input_boxes"].reshape(1, 1, 4)
        outputs = self._model(pixel_values=pv.to(device), input_boxes=ib.to(device), multimask_output=False)
        logits = outputs.pred_masks
        l_sig = logits.sigmoid().cpu()
        o_s = input_instance["original_sizes"].cpu()
        r_i_s = input_instance["reshaped_input_sizes"].cpu()
        # pdb.set_trace()
        probs = self._processor.image_processor.post_process_masks(l_sig, o_s, r_i_s, binarize=False)
        binary_mask = (probs[0] > 0.85).int() * 255

        return binary_mask

    def __call__(self, inputs, network=None, *args, **kwargs):
        """Unified callable function API of Inferers.

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        if network is not None:
            print("LOADING model.pt")
            self._model.load_state_dict(torch.load(network))
        device = kwargs.get('device', None) if kwargs else 'cpu'
        print("Device:", device)
        # pdb.set_trace()
        return self.forward(inputs, device)
        # Below will be run from SimpleInferer and we are overriding
        # return super().__call__(inputs, network, *args, **kwargs)