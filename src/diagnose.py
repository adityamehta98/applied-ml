"""Turn two loss curves into a diagnosis and the first thing to try."""
import math

OVERFIT_RISE = 1.10      # validation ended 10% above its best while training kept improving
NOT_LEARNING = 0.95      # training loss improved by less than 5% overall
SPIKE = 1.5              # a single epoch where training loss jumped by half or more
BIG_GAP = 1.3            # validation loss 30% above training loss at the end

ACTIONS = {
    "diverged": "Lower the learning rate (try 10x smaller) and check for bad inputs.",
    "too_short": "Train for more epochs before judging.",
    "overfitting": "Stop at the best epoch, add data or weight decay, or shrink the model.",
    "underfitting": "Raise the learning rate, train longer, or use a bigger model.",
    "unstable": "Lower the learning rate or raise the batch size.",
    "healthy": "Keep this run. Try a small change to the learning rate to see if it improves.",
}


def diagnose(train_losses, val_losses):
    """Return {"label", "best_epoch", "action"} for a pair of per-epoch loss lists."""
    if any(math.isnan(x) or math.isinf(x) for x in train_losses + val_losses):
        label = "diverged"
    elif len(train_losses) < 3:
        label = "too_short"
    else:
        best = min(val_losses)
        train_improved = train_losses[-1] < train_losses[0]
        if val_losses[-1] > best * OVERFIT_RISE and train_improved:
            label = "overfitting"
        elif train_losses[-1] > train_losses[0] * NOT_LEARNING:
            label = "underfitting"
        elif any(b > a * SPIKE for a, b in zip(train_losses, train_losses[1:])):
            label = "unstable"
        elif val_losses[-1] > train_losses[-1] * BIG_GAP:
            label = "overfitting"
        else:
            label = "healthy"
    best_epoch = val_losses.index(min(val_losses)) if val_losses and label != "diverged" else None
    return {"label": label, "best_epoch": best_epoch, "action": ACTIONS[label]}
