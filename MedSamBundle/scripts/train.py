import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from transformers import SamModel
import monai
from tqdm import tqdm
from statistics import mean


def train(model, device, train_dataloader, num_epochs=5, learning_rate=1e-5):
    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=learning_rate, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        batch_epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()
            
            # optimize
            optimizer.step()
            batch_epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(batch_epoch_losses)}')
        epoch_losses.append(mean(batch_epoch_losses))
        plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(epoch_losses)
        plt.show()
        plt.close()