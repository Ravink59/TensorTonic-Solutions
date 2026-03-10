import numpy as np

def dice_loss(p, y, eps=1e-8):
    
    # Convert inputs to float arrays
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute intersection
    intersection = np.sum(p * y)

    # Compute sums
    sum_p = np.sum(p)
    sum_y = np.sum(y)

    # Dice coefficient
    dice = (2 * intersection + eps) / (sum_p + sum_y + eps)

    # Dice loss
    loss = 1 - dice

    return float(loss)