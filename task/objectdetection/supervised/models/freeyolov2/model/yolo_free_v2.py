import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .module import build_backbone, build_neck, build_fpn, build_head
from ....utils.nms import multiclass_nms


class FreeYOLOv2(nn.Module):
    def __init__(self, cfg, device, num_classes=20, conf_thresh=0.01, topk=100, nms_thresh=0.5, no_decode=False):
        super(FreeYOLOv2, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.cfg = cfg
        self.device = device
        self.stride = list(cfg.stride)
        self.reg_max = cfg.reg_max
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.no_decode = no_decode

        # ---------------------- Network Parameters ----------------------
        # ----------- proj_conv ------------
        self.proj = nn.Parameter(torch.linspace(
            0, cfg.reg_max, cfg.reg_max), requires_grad=False)
        self.proj_conv = nn.Conv2d(self.reg_max, 1, kernel_size=1, bias=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view(
            [1, cfg.reg_max, 1, 1]).clone().detach(), requires_grad=False)

        # ----------- Backbone -----------
        self.backbone, feats_dim = build_backbone(cfg, cfg.pretrained)

        # ----------- Neck: SPP -----------
        self.neck = build_neck(
            cfg=cfg, in_dim=feats_dim[-1], out_dim=feats_dim[-1])
        feats_dim[-1] = self.neck.out_dim

        # ----------- Neck: FPN -----------
        self.fpn = build_fpn(cfg=cfg, in_dims=feats_dim,
                             out_dim=round(256*cfg.width))
        self.head_dim = self.fpn.out_dim

        # ----------- Heads -----------
        self.det_heads = nn.ModuleList(
            [build_head(cfg, head_dim, head_dim, num_classes)
             for head_dim in self.head_dim
             ])

    # ---------------------- Basic Functions ----------------------
    # generate anchor points
    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid(
            [torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y],
                                dim=-1).float().view(-1, 2)
        anchor_xy += 0.5  # add center offset
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors

    # post-process
    def post_process(self, cls_preds, box_preds):
        """
        Input:
            cls_preds: List(Tensor) [[H x W, C], ...]
            box_preds: List(Tensor) [[H x W, 4], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []

        for cls_pred_i, box_pred_i in zip(cls_preds, box_preds):
            # (H x W x C,)
            scores_i = cls_pred_i.sigmoid().flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, box_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            topk_scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor')
            topk_labels = topk_idxs % self.num_classes
            topk_bboxes = box_pred_i[anchor_idxs]

            all_scores.append(topk_scores)
            all_labels.append(topk_labels)
            all_bboxes.append(topk_bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels

    @torch.no_grad()
    def inference_single_image(self, x):
        # ---------------- Backbone ----------------
        pyramid_feats = self.backbone(x)

        # ---------------- Neck: SPP ----------------
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # ---------------- Neck: PaFPN ----------------
        pyramid_feats = self.fpn(pyramid_feats)

        # ---------------- Heads ----------------
        all_cls_preds = []
        all_box_preds = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.det_heads)):
            # ---------------- Pred ----------------
            cls_pred, reg_pred = head(feat)

            # anchors: [M, 2]
            B, _, H, W = reg_pred.size()
            fmp_size = [H, W]
            anchors = self.generate_anchors(level, fmp_size)

            # process preds
            cls_pred = cls_pred.permute(
                0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(
                0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

            # ----------------------- Decode bbox -----------------------
            B, M = reg_pred.shape[:2]
            # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
            reg_pred = reg_pred.reshape([B, M, 4, self.reg_max])
            # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
            reg_pred = reg_pred.permute(0, 3, 2, 1).contiguous()
            # [B, reg_max, 4, M] -> [B, 1, 4, M]
            reg_pred = self.proj_conv(F.softmax(reg_pred, dim=1))
            # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
            reg_pred = reg_pred.view(B, 4, M).permute(0, 2, 1).contiguous()
            # tlbr -> xyxy
            x1y1_pred = anchors[None] - reg_pred[..., :2] * self.stride[level]
            x2y2_pred = anchors[None] + reg_pred[..., 2:] * self.stride[level]
            box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

            # collect preds
            all_cls_preds.append(cls_pred[0])
            all_box_preds.append(box_pred[0])

        if self.no_decode:
            # no post process
            cls_preds = torch.cat(all_cls_preds, dim=0)
            box_pred = torch.cat(all_box_preds, dim=0)
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([box_pred, cls_preds.sigmoid()], dim=-1)

            return outputs
        else:
            # post process
            bboxes, scores, labels = self.post_process(
                all_cls_preds, all_box_preds)

            return bboxes, scores, labels

    def forward(self, x):
        x = self.normalize(x)

        if self.training:
            # ---------------- Backbone ----------------
            pyramid_feats = self.backbone(x)

            # ---------------- Neck: SPP ----------------
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # ---------------- Neck: PaFPN ----------------
            pyramid_feats = self.fpn(pyramid_feats)

            # ---------------- Heads ----------------
            all_anchors = []
            all_cls_preds = []
            all_reg_preds = []
            all_box_preds = []
            all_strides = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.det_heads)):
                # ---------------- Pred ----------------
                cls_pred, reg_pred = head(feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)
                # stride tensor: [M, 1]
                stride_tensor = torch.ones_like(
                    anchors[..., :1]) * self.stride[level]

                # process preds
                cls_pred = cls_pred.permute(
                    0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(
                    0, 2, 3, 1).contiguous().view(B, -1, 4*self.reg_max)

                # ----------------------- Decode bbox -----------------------
                B, M = reg_pred.shape[:2]
                # [B, M, 4*(reg_max)] -> [B, M, 4, reg_max] -> [B, 4, M, reg_max]
                reg_pred_ = reg_pred.reshape([B, M, 4, self.reg_max])
                # [B, M, 4, reg_max] -> [B, reg_max, 4, M]
                reg_pred_ = reg_pred_.permute(0, 3, 2, 1).contiguous()
                # [B, reg_max, 4, M] -> [B, 1, 4, M]
                reg_pred_ = self.proj_conv(F.softmax(reg_pred_, dim=1))
                # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
                reg_pred_ = reg_pred_.view(
                    B, 4, M).permute(0, 2, 1).contiguous()
                # tlbr -> xyxy
                x1y1_pred = anchors[None] - \
                    reg_pred_[..., :2] * self.stride[level]
                x2y2_pred = anchors[None] + \
                    reg_pred_[..., 2:] * self.stride[level]
                box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

                # collect preds
                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)
                all_box_preds.append(box_pred)
                all_anchors.append(anchors)
                all_strides.append(stride_tensor)

            # output dict
            outputs = {"pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                       # List(Tensor) [B, M, 4*(reg_max)]
                       "pred_reg": all_reg_preds,
                       # List(Tensor) [B, M, 4]
                       "pred_box": all_box_preds,
                       "anchors": all_anchors,           # List(Tensor) [M, 2]
                       # List(Int) = [8, 16, 32]
                       "strides": self.stride,
                       "stride_tensor": all_strides      # List(Tensor) [M, 1]
                       }
            return outputs
        else:
            return self.inference_single_image(x)
