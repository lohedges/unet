# A set of custom Keras loss functions.
# Adapted from: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch/execution

import tensorflow.keras.backend as K

def DiceLoss(targets, inputs, smooth=1e-6):

    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice

def DiceBCELoss(targets, inputs, smooth=1e-6):

    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE =  binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

def IoULoss(targets, inputs, smooth=1e-6):

    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

def FocalLoss(targets, inputs, alpha=0.8, gamma=2):

    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)

    return focal_loss

def TverskyLoss(targets, inputs, alpha=0.5, beta=0.5, smooth=1e-6):

    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    return 1 - Tversky

def FocalTverskyLoss(targets, inputs, alpha=0.5, beta=0.5, gamma=1, smooth=1e-6):

    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    FocalTversky = K.pow((1 - Tversky), gamma)

    return FocalTversky

def Combo_loss(targets, inputs, alpha=0.5, ce_ratio=0.5):

    targets = K.flatten(targets)
    inputs = K.flatten(inputs)

    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, e, 1.0 - e)
    out = - (alpha * ((targets * K.log(inputs)) + ((1 - alpha) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)

    return combo
