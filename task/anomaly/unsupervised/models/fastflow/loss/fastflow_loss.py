import torch
import torch.nn as nn
from torch import Tensor


class FastFlowLoss(nn.Module):
    """FastFlow Loss."""

    def __init__(self) -> None:
        super(FastFlowLoss, self).__init__()

    def forward(self, hidden_variables, jacobians) -> Tensor:
        loss = torch.tensor(0.0, device=hidden_variables[0].device)
        for (hidden_variable, jacobian) in zip(hidden_variables, jacobians):
            loss += torch.mean(0.5 * torch.sum(hidden_variable **
                               2, dim=(1, 2, 3)) - jacobian)
        return loss
