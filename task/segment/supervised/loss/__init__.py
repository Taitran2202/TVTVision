from .dice_bce_loss import DiceBCELoss
from .dice_loss import DiceLoss


def build_loss(loss_name):
    return (DiceLoss(), "Dice Loss") if loss_name == 'dice_loss' else (DiceBCELoss(), "Dice BCE Loss")
