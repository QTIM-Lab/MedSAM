import pdb
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import cv2
import numpy as np

# from losses import DiceLoss
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Assuming logits are the raw model outputs and targets are the ground truth masks
        probs = torch.sigmoid(logits)
        num = targets.size(0)
        
        # p = probs.view(num, -1)
        p = probs.reshape(num, -1)
        t = targets.view(num, -1)
        
        intersection = torch.einsum('bi,bi->b', [p, t])
        p_sum = torch.einsum('bi->b', [p])
        t_sum = torch.einsum('bi->b', [t])
        
        dice = (2.0 * intersection + self.smooth) / (p_sum + t_sum + self.smooth)
        loss = 1 - dice.mean()
        
        return loss

# BB Not used that I can tell. Seems DiceLoss above replaced these.
# def dice_coefficient_torch(y_true, y_pred):
#     intersection = torch.sum(y_true * y_pred)
#     dice = (2.0 * intersection) / (torch.sum(y_true) + torch.sum(y_pred))
#     return dice

# # Calculate pairwise Dice coefficient (Dpw)
# def dice_coefficient(y_true, y_pred):
#     intersection = np.sum(y_true * y_pred)
#     dice = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))
#     return dice

# Function to find maximum diameter and draw the line
# def find_and_draw_max_diameter(contour, image, color_tuple):
#     max_diameter = 0
#     max_diameter_points = None
#     # print("CONTOUR: ", contour)
#     for i in range(len(contour)):
#         for j in range(i + 1, len(contour)):
#             dist = np.linalg.norm(contour[i][0] - contour[j][0])  # Euclidean distance
#             # print("DIST: ", dist)
#             if dist > max_diameter:
#                 max_diameter = dist
#                 max_diameter_points = (tuple(contour[i][0]), tuple(contour[j][0]))

#     if max_diameter_points is not None:
#         cv2.line(image, max_diameter_points[0], max_diameter_points[1], color_tuple, 2)

#     return max_diameter


def train(model, weights_path, train_dataloader, num_epochs=1, learning_rate=5e-6):
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
        print("weights loaded")
    train_losses = []
    best_model_state = None

    # Define loss function and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    num_batches = len(train_dataloader)

    print("num batches: ", num_batches)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0  # Track total loss for the epoch
        pdb.set_trace()
        print('test2')
        train_dataloader
        train_dataloader.__len__()
        listed = [i for i in train_dataloader.dataset]
        dir(train_dataloader.dataset)
        train_dataloader.dataset.data_list
        for batch_idx, (images, masks) in enumerate(train_dataloader):
            # images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            print("outputs: ", outputs)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{num_batches}] - Loss: {loss.item():.4f}")
        
        average_loss = total_loss / num_batches
        train_losses.append(average_loss)

        # Step the scheduler to update the learning rate
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {average_loss:.4f}")

        # only save on last iter
        if epoch == num_epochs - 1:
            best_model_state = model.state_dict()
            # checkpoint_path = os.path.join(savedir, 'model_ckpt_' + str(epoch) + '.pt')
            checkpoint_path = weights_path.split('.pt')[0] + '_ckpt_' + str(epoch) + '.pt'
            torch.save(best_model_state, checkpoint_path)
            print("Best model saved!")

    # print("Training ended")

    # plt.figure()
    # plt.plot(range(1, epoch+2), train_losses, label='Train Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Curves')
    # plt.legend()
    # plt.savefig(os.path.join(savedir, 'loss_curves.png'))
    # plt.close()
    # print("Training over, fig saved, evaling")
    # print("All done!")


