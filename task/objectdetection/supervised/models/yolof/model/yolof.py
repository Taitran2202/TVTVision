import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .module import build_backbone, build_neck, build_head
from ....utils.nms import multiclass_nms


class YOLOF(nn.Module):
    def __init__(self,
                 device,
                 cfg,
                 num_classes: int = 80,
                 conf_thresh: float = 0.05,
                 nms_thresh: float = 0.6,
                 topk: int = 1000,
                 no_decode: bool = False
                 ):
        super(YOLOF, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.cfg = cfg
        self.device = device
        self.topk = topk
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.no_decode = no_decode

        # ---------------------- Network Parameters ----------------------
        # Backbone
        self.backbone, feat_dims = build_backbone(
            cfg, cfg.pretrained)

        # Neck
        self.neck = build_neck(cfg, feat_dims[-1], cfg.head_dim)

        # Heads
        self.head = build_head(
            cfg, cfg.head_dim, cfg.head_dim, num_classes)

    def post_process(self, cls_pred, box_pred):
        """
        Input:
            cls_pred: (Tensor) [[H x W x KA, C]
            box_pred: (Tensor)  [H x W x KA, 4]
        """
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]

        # (H x W x KA x C,)
        scores_i = cls_pred.sigmoid().flatten()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk, box_pred.size(0))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = scores_i.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        # filter out the proposals with low confidence score
        keep_idxs = topk_scores > self.conf_thresh
        topk_idxs = topk_idxs[keep_idxs]

        # final scores
        scores = topk_scores[keep_idxs]
        # final labels
        labels = topk_idxs % self.num_classes
        # final bboxes
        anchor_idxs = torch.div(
            topk_idxs, self.num_classes, rounding_mode='floor')
        bboxes = box_pred[anchor_idxs]

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

        # ---------------- Neck ----------------
        feat = self.neck(pyramid_feats[-1])

        # ---------------- Heads ----------------
        outputs = self.head(feat)

        # ---------------- PostProcess ----------------
        all_cls_preds = outputs["pred_cls"]
        all_box_preds = outputs["pred_box"]

        if self.no_decode:
            cls_preds = torch.cat(all_cls_preds, dim=1)[0]
            box_preds = torch.cat(all_box_preds, dim=1)[0]
            scores = cls_preds.sigmoid()
            bboxes = box_preds
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)
        else:

            bboxes, scores, labels = self.post_process(
                all_cls_preds, all_box_preds)

            return bboxes, scores, labels

    def forward(self, x):
        x = self.normalize(x)

        if self.training:
            # ---------------- Backbone ----------------
            pyramid_feats = self.backbone(x)

            # ---------------- Neck ----------------
            feat = self.neck(pyramid_feats[-1])

            # ---------------- Heads ----------------
            outputs = self.head(feat)

            return outputs
        else:
            return self.inference_single_image(x)
