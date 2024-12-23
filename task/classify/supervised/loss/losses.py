import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class BalancedSoftmax(_Loss):
    def __init__(self, num_per_cls: list):
        super(BalancedSoftmax, self).__init__()
        self.num_per_cls = torch.tensor(num_per_cls)

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(labels=label, logits=input, num_per_cls=self.num_per_cls, reduction=reduction)


def balanced_softmax_loss(labels, logits, num_per_cls, reduction):
    npc = num_per_cls.type_as(logits)
    npc = npc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + npc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
