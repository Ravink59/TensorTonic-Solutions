import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y)

    # Handle single vector inputs
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    y = y.reshape(-1)

    # Validate labels
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")

    # Euclidean distance
    diff = a - b
    d = np.sqrt(np.sum(diff**2, axis=1))

    # Contrastive loss components
    positive_loss = y * (d ** 2)
    negative_loss = (1 - y) * (np.maximum(0, margin - d) ** 2)

    losses = positive_loss + negative_loss

    if reduction == "mean":
        return float(np.mean(losses))
    elif reduction == "sum":
        return float(np.sum(losses))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")