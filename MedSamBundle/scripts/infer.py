import pdb

def infer(model, device, dataset_instance):
    model.to(device)
    # forward pass
    outputs = model(pixel_values=dataset_instance["pixel_values"].to(device),
                    input_boxes=dataset_instance["input_boxes"].to(device),
                    multimask_output=False)

    predicted_masks = outputs.pred_masks.squeeze(1)
    return predicted_masks