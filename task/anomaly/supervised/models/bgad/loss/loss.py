import torch
import torch.nn as nn


def normal_fl_weighting(logps, gamma=0.5, alpha=11.7, normalizer=10):
    """
    Normal focal weighting.
    Args:
        logps: og-likelihoods, shape (N, ).
        gamma: gamma hyperparameter for normal focal weighting.
        alpha: alpha hyperparameter for abnormal focal weighting.
    """
    logps = logps / normalizer
    mask_larger = logps > -0.2
    mask_lower = logps <= -0.2
    probs = torch.exp(logps)
    fl_weights = alpha * (1 - probs).pow(gamma) * torch.abs(logps)
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_larger] = 1.0
    weights[mask_lower] = fl_weights[mask_lower]

    return weights


def get_logp_boundary(logps, mask, pos_beta=0.05, margin_tau=0.1, normalizer=10):
    """
    Find the equivalent log-likelihood decision boundaries from normal log-likelihood distribution.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, )
        pos_beta: position hyperparameter: beta
        margin_tau: margin hyperparameter: tau
    """
    normal_logps = logps[mask == 0].detach()
    n_idx = int(((mask == 0).sum() * pos_beta).item())
    sorted_indices = torch.sort(normal_logps)[1]

    n_idx = sorted_indices[n_idx]
    b_n = normal_logps[n_idx]  # normal boundary
    b_n = b_n / normalizer

    b_a = b_n - margin_tau  # abnormal boundary

    return b_n, b_a


def calculate_bg_spp_loss_normal(logps, mask, boundaries, normalizer=10, weights=None):
    """
    Calculate boudary guided semi-push-pull contrastive loss.
    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: normal boundary and abnormal boundary
    """
    logps = logps / normalizer
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[mask == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    loss_n = b_n - normal_logps_inter

    if weights is not None:
        nor_weights = weights[mask == 0][normal_logps <= b_n]
        loss_n = loss_n * nor_weights

    loss_n = torch.mean(loss_n)

    return loss_n


class Boundary(nn.Module):
    def __init__(self, pos_beta=0.05, margin_tau=0.1, normalizer=10):
        super(Boundary, self).__init__()
        self.pos_beta = pos_beta
        self.margin_tau = margin_tau
        self.normalizer = normalizer

    def forward(self, logps, mask):
        normal_logps = logps[mask == 0].detach()
        n_idx = int(((mask == 0).sum() * self.pos_beta).item())
        sorted_indices = torch.sort(normal_logps)[1]

        n_idx = sorted_indices[n_idx]
        b_n = normal_logps[n_idx]  # normal boundary
        b_n = b_n / self.normalizer

        b_a = b_n - self.margin_tau  # abnormal boundary

        return b_n, b_a
