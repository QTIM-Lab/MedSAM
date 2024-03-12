import pdb

def infer(model, device, dataset_instance):
    model.to(device)
    batch = next(iter(dataset_instance))
    # forward pass
    # pdb.set_trace()
    outputs = model(pixel_values=batch["pixel_values"].to(device),
                    input_boxes=batch["input_boxes"].to(device),
                    multimask_output=False)

    predicted_masks = outputs.pred_masks.squeeze(1)
    return predicted_masks