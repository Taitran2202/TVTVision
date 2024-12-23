import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .module import build_backbone, build_neck, build_head
from ....utils.nms import multiclass_nms


class YOLOv1(nn.Module):
    def __init__(self, cfg, device, num_classes=20, conf_thresh=0.01, nms_thresh=0.5, no_decode=False):
        super(YOLOv1, self).__init__()
        # --------- Basic Parameters ----------
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.device = device
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.no_decode = no_decode
        self.stride = 32

        # --------- Network Parameters ----------
        # backbone
        self.backbone, feat_dim = build_backbone(
            model_name=cfg.backbone, pretrained=cfg.pretrained)

        # neck
        self.neck = build_neck(cfg, feat_dim, out_dim=512)
        head_dim = self.neck.out_dim

        # non-shared heads
        self.head = build_head(cfg, head_dim, head_dim, num_classes)

        # pred
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)

    def create_grid(self, fmp_size):
        """
        Function to create a grid matrix G, where each element is the pixel coordinate on the feature map.
        """
        # Width and height of the feature map
        ws, hs = fmp_size

        # Generate the x and y coordinates of the grid
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

        # Concatenate the x and y coordinates: [H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        # Reshape to [HW, 2] and move to the device
        grid_xy = grid_xy.view(-1, 2).to(self.device)

        return grid_xy

    def decode_boxes(self, pred, fmp_size):
        """
        Convert predicted bounding boxes from (tx, ty, tw, th) format to the common (x1, y1, x2, y2) format.
        """
        # Generate the grid coordinate matrix
        grid_cell = self.create_grid(fmp_size)

        # Calculate the center coordinates and width-height of the predicted bounding boxes
        pred_ctr = (torch.sigmoid(pred[..., :2]) + grid_cell) * self.stride
        pred_wh = torch.exp(pred[..., 2:]) * self.stride

        # Convert all bounding boxes from center-width-height format to (x1, y1, x2, y2) format
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box

    def postprocess(self, bboxes, scores):
        """
        Input:
            bboxes: [HxW, 4]
            scores: [HxW, num_classes]
        Output:
            bboxes: [N, 4]
            score:  [N,]
            labels: [N,]
        """

        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels

    @torch.no_grad()
    def inference_single_image(self, x):
        # backbone
        feat = self.backbone(x)

        # neck
        feat = self.neck(feat)

        # detection head
        cls_feat, reg_feat = self.head(feat)

        # prediction layer
        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        fmp_size = obj_pred.shape[-2:]

        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        obj_pred = obj_pred[0]
        cls_pred = cls_pred[0]
        reg_pred = reg_pred[0]

        # score for each bounding box
        scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())

        # decode bbox
        bboxes = self.decode_boxes(reg_pred, fmp_size)

        if self.no_decode:
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

            return outputs
        else:
            scores = scores.cpu().numpy()
            bboxes = bboxes.cpu().numpy()

            # post process
            bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels

    def forward(self, x):
        x = self.normalize(x)

        if self.training:
            # backbone
            feat = self.backbone(x)

            # neck
            feat = self.neck(feat)

            # detection head
            cls_feat, reg_feat = self.head(feat)

            # prediction layer
            obj_pred = self.obj_pred(cls_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[-2:]

            # Make some view adjustments to the size of pred to facilitate subsequent processing
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # decode bbox
            box_pred = self.decode_boxes(reg_pred, fmp_size)

            # network output
            outputs = {
                "pred_obj": obj_pred,                  # (Tensor) [B, M, 1]
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
