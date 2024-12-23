import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .module import *
from ....utils.nms import multiclass_nms


class YOLOXv2(nn.Module):
    def __init__(self, cfg, device, num_classes=20, conf_thresh=0.05, nms_thresh=0.6, topk=1000, no_decode=False):
        super(YOLOXv2, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.cfg = cfg
        self.device = device
        self.strides = cfg.stride
        self.num_levels = len(self.strides)
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.no_decode = no_decode
        self.head_dim = round(256 * cfg.width)

        # ------------------- Network Structure -------------------
        # backbone network
        self.backbone, feat_dims = build_backbone(cfg)

        # Neck: SPP
        self.neck = build_neck(cfg, feat_dims[-1], feat_dims[-1])
        feat_dims[-1] = self.neck.out_dim

        # Neck: FPN
        self.fpn = build_fpn(cfg, feat_dims, out_dim=self.head_dim)
        self.fpn_dims = self.fpn.out_dim

        # Heads
        self.det_heads = build_det_head(
            cfg, self.fpn_dims, self.head_dim, self.num_levels)

        # ----------- Preds -----------
        self.pred_layers = build_pred_layer(
            cls_dim=self.det_heads.cls_head_dim,
            reg_dim=self.det_heads.reg_head_dim,
            strides=self.strides,
            num_classes=num_classes,
            num_coords=4,
            num_levels=self.num_levels
        )

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
                                dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors

    def decode_boxes(self, anchors, pred_regs, stride):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors + pred_regs[..., :2] * stride
        # size of bbox
        pred_box_wh = pred_regs[..., 2:].exp() * stride

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box

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
            cls_pred_i = cls_pred_i[0]
            box_pred_i = box_pred_i[0]

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
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

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
        # backbone
        pyramid_feats = self.backbone(x)

        # Neck: SPP
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # Neck: PaFPN
        pyramid_feats = self.fpn(pyramid_feats)

        # Heads
        cls_feats, reg_feats = self.det_heads(pyramid_feats)

        # Preds
        outputs = self.pred_layers(cls_feats, reg_feats)

        all_cls_preds = outputs['pred_cls']
        all_box_preds = outputs['pred_box']

        if self.no_decode:
            cls_preds = torch.cat(all_cls_preds, dim=1)[0]
            box_preds = torch.cat(all_box_preds, dim=1)[0]
            scores = cls_preds.sigmoid()
            bboxes = box_preds
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

            return outputs

        else:
            # post process
            bboxes, scores, labels = self.post_process(
                all_cls_preds, all_box_preds)

            return bboxes, scores, labels

    def forward(self, x):
        x = self.normalize(x)

        if self.training:
            # Backbone
            pyramid_feats = self.backbone(x)

            # Neck: SPP
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # Neck: PaFPN
            pyramid_feats = self.fpn(pyramid_feats)

            # Heads
            cls_feats, reg_feats = self.det_heads(pyramid_feats)

            # Preds
            outputs = self.pred_layers(cls_feats, reg_feats)

            return outputs
        else:
            return self.inference_single_image(x)
