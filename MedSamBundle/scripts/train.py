import os, pdb, shutil
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from transformers import SamModel
import monai
from tqdm import tqdm
from statistics import mean

from transformers import SamProcessor

def train(model, device, weights_path, train_dataloader, num_epochs=10, learning_rate=1e-5):
    # make sure we only compute gradients for mask decoder
    # pdb.set_trace()
    type(model)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=learning_rate, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        batch_epoch_losses = []
        # pdb.set_trace()
        for batch_num, batch in enumerate(tqdm(train_dataloader)):
            # forward pass
            batch["pixel_values"].shape
            batch["input_boxes"].shape
            outputs = model(pixel_values=batch["pixel_values"].to(device), input_boxes=batch["input_boxes"].to(device), multimask_output=False)
            # outputs = model(pixel_values=batch["pixel_values"].to(device),
            #                 input_boxes=batch["input_boxes"].to(device),
            #                 multimask_output=False)
            
            # compute loss
            # pdb.set_trace()
            logits = outputs.pred_masks
            o_s = batch["original_sizes"].cpu()
            r_i_s = batch["reshaped_input_sizes"].cpu()
            logits_resized = processor.image_processor.post_process_masks(logits, o_s, r_i_s, binarize=False)
            ground_truth_masks = batch["ground_truth_mask"].to(device)
            loss = seg_loss(logits_resized[0], ground_truth_masks.unsqueeze(1))
            
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()
            
            # optimize
            optimizer.step()
            batch_epoch_losses.append(loss.item())

        num_batches = batch_num+1
        total_loss = sum(batch_epoch_losses)
        avg_loss = total_loss/num_batches
        epoch_losses.append(avg_loss)
        if avg_loss <= min(epoch_losses):            
            best_model_state = model.state_dict()
            # pdb.set_trace()
            # checkpoint_path = weights_path.split('.pt')[0] + '_best' + '.pt'
            if os.path.exists(weights_path):            
                os.remove(weights_path)
            checkpoint_path = os.path.join(os.path.dirname(weights_path), 'model_best.pt')
            with open(os.path.join(os.path.dirname(weights_path), 'ckpt_best_model.txt'), 'w') as file:
                file.write(f"Best model is from epoch: {epoch+1}\n")
            torch.save(best_model_state, checkpoint_path)
            print("Best model saved!")
        # print(f'EPOCH: {epoch}')
        # print(f'Mean loss: {mean(batch_epoch_losses)}')
        # epoch_losses.append(mean(batch_epoch_losses))
        # plt.figure()
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.plot(epoch_losses)
        # plt.show()
        # plt.close()