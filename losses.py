import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()

        inter = (outputs * targets_one_hot).sum(dim=(2, 3))
        union = outputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        ce = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()

        tp = (outputs * targets_one_hot).sum(dim=(2, 3))
        fp = (outputs * (1 - targets_one_hot)).sum(dim=(2, 3))
        fn = ((1 - outputs) * targets_one_hot).sum(dim=(2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class LovaszSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()

        losses = []
        for i in range(outputs.shape[1]):
            pred = outputs[:, i]
            target = targets_one_hot[:, i]
            inter = (pred * target).sum()
            union = pred.sum() + target.sum() - inter
            iou = inter / (union + 1e-5)
            losses.append(1.0 - iou)
        return torch.stack(losses).mean()


class HybridLoss(nn.Module):
    def __init__(self, weights=[0.3, 0.4, 0.2, 0.1], label_smoothing=0.1, class_weights=None, use_lovasz=True):
        super().__init__()
        self.weights = weights
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
        self.lovasz = LovaszSoftmax()
        self.use_lovasz = use_lovasz

    def forward(self, outputs, targets, aux_outputs=None):
        if aux_outputs is None:
            return self._single_loss(outputs, targets)
        else:
            total_loss, log = self._single_loss(outputs, targets)
            aux_w = [0.4, 0.3, 0.3]
            for i, aux_out in enumerate(aux_outputs):
                aux_loss, aux_log = self._single_loss(aux_out, targets)
                total_loss += aux_w[i] * aux_loss
                for k, v in aux_log.items():
                    log[f'aux{i+1}_{k}'] = v
            return total_loss, log

    def _single_loss(self, outputs, targets):
        ce = self.ce(outputs, targets)
        dice = self.dice(outputs, targets)
        focal = self.focal(outputs, targets)
        tversky = self.tversky(outputs, targets)
        lovasz = self.lovasz(outputs, targets)

        extra = lovasz if self.use_lovasz else tversky
        loss = (self.weights[0] * ce +
                self.weights[1] * dice +
                self.weights[2] * focal +
                self.weights[3] * extra)

        return loss, {
            'ce_loss': ce.item(),
            'dice_loss': dice.item(),
            'focal_loss': focal.item(),
            'tversky_loss': tversky.item(),
            'lovasz_loss': lovasz.item()
        }


def get_loss_function(num_classes=16, loss_weights=[0.3, 0.4, 0.2, 0.1],
                      label_smoothing=0.1, class_weights=None, use_lovasz=True):
    return HybridLoss(
        weights=loss_weights,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        use_lovasz=use_lovasz
    )