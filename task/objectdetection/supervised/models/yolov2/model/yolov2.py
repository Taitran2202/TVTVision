import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .module import *
from ....utils.nms import multiclass_nms


class YOLOv2(nn.Module):
    def __init__(self, cfg, device, num_classes=20, conf_thresh=0.01, nms_thresh=0.5, topk=100, no_decode=False):
        super(YOLOv2, self).__init__()
        # ------------------- Basic parameters -------------------
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.stride = 32
        self.no_decode = no_decode

        # ------------------- Anchor box -------------------
        self.anchor_size = torch.as_tensor(
            cfg.anchor_size).float().view(-1, 2)  # [A, 2]
        self.num_anchors = self.anchor_size.shape[0]

        # ------------------- Network Structure -------------------
        # backbone network
        self.backbone, feat_dim = build_backbone(
            cfg.backbone, cfg.pretrained)

        # neck
        self.neck = build_neck(cfg, feat_dim, out_dim=512)
        head_dim = self.neck.out_dim

        # non-shared heads
        self.head = build_head(cfg, head_dim, head_dim, num_classes)

        # pred
        self.obj_pred = nn.Conv2d(head_dim, 1*self.num_anchors, kernel_size=1)
        self.cls_pred = nn.Conv2d(
            head_dim, num_classes*self.num_anchors, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4*self.num_anchors, kernel_size=1)

        self.init_bias()

    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.obj_pred.bias, bias_value)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def generate_anchors(self, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        fmp_h, fmp_w = fmp_size

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid(
            [torch.arange(fmp_h), torch.arange(fmp_w)])
        anchor_xy = torch.stack([anchor_x, anchor_y],
                                dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, A, 2] -> [M, 2]
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1)
        anchor_xy = anchor_xy.view(-1, 2).to(self.device)

        # [A, 2] -> [1, A, 2] -> [HW, A, 2] -> [M, 2]
        anchor_wh = self.anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)

        return anchors

    def decode_boxes(self, anchors, reg_pred):
        """
            Convert predicted bounding boxes from (tx, ty, tw, th) format to the common (x1, y1, x2, y2) format.
        """

        # Calculate the center coordinates and width-height of the predicted bounding boxes
        pred_ctr = (torch.sigmoid(
            reg_pred[..., :2]) + anchors[..., :2]) * self.stride
        pred_wh = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]

        # Convert all bounding boxes from center-width-height format to (x1, y1, x2, y2) format
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box

    def postprocess(self, obj_pred, cls_pred, reg_pred, anchors):
        """
        Input:
            obj_pred: (Tensor) [H*W*A, 1]
            cls_pred: (Tensor) [H*W*A, C]
            reg_pred: (Tensor) [H*W*A, 4]
        """
        # (H x W x A x C,)
        scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid()).flatten()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk, reg_pred.size(0))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = scores.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        # filter out the proposals with low confidence score
        keep_idxs = topk_scores > self.conf_thresh
        scores = topk_scores[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]

        anchor_idxs = torch.div(
            topk_idxs, self.num_classes, rounding_mode='floor')
        labels = topk_idxs % self.num_classes

        reg_pred = reg_pred[anchor_idxs]
        anchors = anchors[anchor_idxs]

        # decode box: [M, 4]
        bboxes = self.decode_boxes(anchors, reg_pred)

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
        bs = x.shape[0]
        # backbone
        feat = self.backbone(x)

        # neck
        feat = self.neck(feat)

        # detection head
        cls_feat, reg_feat = self.head(feat)

        # prediction layer
        obj_pred = self.obj_pred(reg_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]

        # anchors: [M, 2]
        anchors = self.generate_anchors(fmp_size)

        # [B, A*C, H, W] -> [B, H, W, A*C] -> [B, H*W*A, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(
            bs, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

        obj_pred = obj_pred[0]       # [H*W*A, 1]
        cls_pred = cls_pred[0]       # [H*W*A, NC]
        reg_pred = reg_pred[0]       # [H*W*A, 4]

        if self.no_decode:
            scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())
            bboxes = self.decode_boxes(anchors, reg_pred)
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

            return outputs
        else:
            # post process
            bboxes, scores, labels = self.postprocess(
                obj_pred, cls_pred, reg_pred, anchors)

            return bboxes, scores, labels

    def forward(self, x):
        x = self.normalize(x)

        if self.training:
            bs = x.shape[0]

            # backbone
            feat = self.backbone(x)

            # neck
            feat = self.neck(feat)

            # detection head
            cls_feat, reg_feat = self.head(feat)

            # prediction layer
            obj_pred = self.obj_pred(reg_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[-2:]

            # anchors: [M, 2]
            anchors = self.generate_anchors(fmp_size)

            # [B, A*C, H, W] -> [B, H, W, A*C] -> [B, H*W*A, C]
            obj_pred = obj_pred.permute(
                0, 2, 3, 1).contiguous().view(bs, -1, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(
                bs, -1, self.num_classes)
            reg_pred = reg_pred.permute(
                0, 2, 3, 1).contiguous().view(bs, -1, 4)

            # decode bbox
            box_pred = self.decode_boxes(anchors, reg_pred)

            # output dict
            outputs = {"pred_obj": obj_pred,                   # (Tensor) [B, M, 1]
                       # (Tensor) [B, M, C]
                       "pred_cls": cls_pred,
                       # (Tensor) [B, M, 4]
                       "pred_box": box_pred,
                       "stride": self.stride,                  # (Int)
                       # (List) [fmp_h, fmp_w]
                       "fmp_size": fmp_size
                       }
            return outputs
        else:
            return self.inference_single_image(x)
