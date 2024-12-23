import torch.nn as nn
from .yolo_free_v2_basic import Conv


class DecoupledHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, num_classes=80):
        super().__init__()
        # --------- Basic Parameters ----------
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_cls_head = cfg.num_cls_head
        self.num_reg_head = cfg.num_reg_head

        # --------- Network Parameters ----------
        # cls head
        cls_feats = []
        self.cls_out_dim = max(out_dim, num_classes)
        for i in range(cfg.num_cls_head):
            if i == 0:
                cls_feats.append(
                    Conv(in_dim, self.cls_out_dim, k=3, p=1, s=1,
                         act_type=cfg.head_act,
                         norm_type=cfg.head_norm,
                         depthwise=cfg.head_depthwise)
                )
            else:
                cls_feats.append(
                    Conv(self.cls_out_dim, self.cls_out_dim, k=3, p=1, s=1,
                         act_type=cfg.head_act,
                         norm_type=cfg.head_norm,
                         depthwise=cfg.head_depthwise)
                )
        # reg head
        reg_feats = []
        self.reg_out_dim = max(out_dim, 4*cfg.reg_max)
        for i in range(cfg.num_reg_head):
            if i == 0:
                reg_feats.append(
                    Conv(in_dim, self.reg_out_dim, k=3, p=1, s=1,
                         act_type=cfg.head_act,
                         norm_type=cfg.head_norm,
                         depthwise=cfg.head_depthwise)
                )
            else:
                reg_feats.append(
                    Conv(self.reg_out_dim, self.reg_out_dim, k=3, p=1, s=1,
                         act_type=cfg.head_act,
                         norm_type=cfg.head_norm,
                         depthwise=cfg.head_depthwise)
                )
        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)

        # Pred
        self.cls_pred = nn.Conv2d(self.cls_out_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(
            self.reg_out_dim, 4*cfg.reg_max, kernel_size=1)

    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        cls_pred = self.cls_pred(cls_feats)
        reg_pred = self.reg_pred(reg_feats)

        return cls_pred, reg_pred


# build detection head
def build_head(cfg, in_dim, out_dim, num_classes=80):
    if cfg['head'] == 'decoupled_head':
        head = DecoupledHead(cfg, in_dim, out_dim, num_classes)

    return head
