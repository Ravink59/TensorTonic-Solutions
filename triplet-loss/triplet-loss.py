import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):

    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)

    # Convert single vectors to batch form
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
        positive = positive.reshape(1, -1)
        negative = negative.reshape(1, -1)

    # Squared Euclidean distances
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)

    # Triplet loss
    loss = np.maximum(0, d_ap - d_an + margin)

    # Mean loss
    return np.mean(loss)