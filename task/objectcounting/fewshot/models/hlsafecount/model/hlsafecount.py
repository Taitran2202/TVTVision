import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .module import ResNet, SAFECountMultiBlock, Regressor, DetectPeaks


def crop_roi_feat(feat, boxes, out_stride):
    """
    feat: 1 x c x h x w
    boxes: m x 4, 4: [y_tl, x_tl, y_br, x_br]
    """
    _, _, h, w = feat.shape
    boxes_scaled = boxes / out_stride
    boxes_scaled[:, :2] = torch.floor(boxes_scaled[:, :2])  # y_tl, x_tl: floor
    boxes_scaled[:, 2:] = torch.ceil(boxes_scaled[:, 2:])  # y_br, x_br: ceil
    boxes_scaled[:, :2] = torch.clamp_min(boxes_scaled[:, :2], 0)
    boxes_scaled[:, 2] = torch.clamp_max(
        boxes_scaled[:, 2], torch.tensor(h).to(boxes_scaled.device))
    boxes_scaled[:, 3] = torch.clamp_max(
        boxes_scaled[:, 3], torch.tensor(w).to(boxes_scaled.device))
    feat_boxes = []
    for idx_box in range(0, boxes.shape[0]):
        y_tl, x_tl, y_br, x_br = boxes_scaled[idx_box]
        y_tl, x_tl, y_br, x_br = int(y_tl), int(x_tl), int(y_br), int(x_br)
        feat_box = feat[:, :, y_tl: (y_br + 1), x_tl: (x_br + 1)]
        feat_boxes.append(feat_box)
    return feat_boxes


class HLSAFECount(nn.Module):
    def __init__(self, num_block: int, backbone_type: str, backbone_out_layers: list, backbone_out_stride: int,
                 pretrained: bool, embed_dim: int, mid_dim: int, head: int, dropout: float, exemplar_scales: list[float],
                 image_support=None, boxes_support=None):
        super(HLSAFECount, self).__init__()
        self.backbone_type = backbone_type
        self.out_layers = backbone_out_layers
        self.out_stride = backbone_out_stride
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.head = head
        self.dropout = dropout
        self.exemplar_scales = exemplar_scales
        self.image_support = nn.Parameter(image_support, requires_grad=False)
        self.boxes_support = nn.Parameter(boxes_support, requires_grad=False)

        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = ResNet(
            backbone_type=self.backbone_type,
            out_stride=self.out_stride,
            out_layers=self.out_layers,
            pretrained=pretrained
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.in_conv = nn.Conv2d(
            self.backbone.out_dim, embed_dim, kernel_size=1, stride=1
        )
        self.safecount = SAFECountMultiBlock(
            num_block=num_block,
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            head=head,
            dropout=dropout
        )

        self.count_regressor = Regressor(in_dim=embed_dim)
        self.detect_peaks = DetectPeaks()

        for m in [self.in_conv, self.safecount, self.count_regressor]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, image):
        # normalize
        image = self.normalize(image)

        image_support = self.image_support  # new code
        image_support = self.normalize(image_support)

        boxes = self.boxes_support
        boxes = boxes.squeeze(0)  # [1,m,4] -> [m,4]\
        fq = self.in_conv(self.backbone(image))
        _, _, h, w = image_support.shape

        feat_scale_list = []
        boxes_scale_list = []
        for scale in self.exemplar_scales:
            h_rsz = int(h * scale) // 16 * 16
            w_rsz = int(w * scale) // 16 * 16
            image_scale = F.interpolate(
                image_support, size=(w_rsz, h_rsz), mode="bilinear")
            scale_h = h_rsz / h
            scale_w = w_rsz / w
            boxes_scale = boxes.clone()
            boxes_scale[:, 0] *= scale_h
            boxes_scale[:, 1] *= scale_w
            boxes_scale[:, 2] *= scale_h
            boxes_scale[:, 3] *= scale_w
            feat_scale = self.in_conv(self.backbone(image_scale))
            feat_scale_list.append(feat_scale)
            boxes_scale_list.append(boxes_scale)

        feat_boxes_list = []
        for feat, boxes in zip(feat_scale_list, boxes_scale_list):
            feat_boxes = crop_roi_feat(feat, boxes, self.out_stride)
            for feat_box in feat_boxes:
                feat_box = F.interpolate(feat_box, (3, 3), mode='bilinear')
                feat_boxes_list.append(feat_box)

        fs = torch.cat(feat_boxes_list, dim=0)
        output = self.safecount(fq, fs)

        return self.count_regressor(output), self.detect_peaks(self.count_regressor(output)).sum()
