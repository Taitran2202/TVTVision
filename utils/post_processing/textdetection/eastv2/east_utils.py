import cv2
import numpy as np
from task.textdetection.supervised.models.east.data.module import get_rotate_mat
from .locality_aware_nms import nms_locality


def adjustResultCoordinates(boxes, ratio_w, ratio_h):
    '''refine boxes
    Input:
            boxes  : detected polys <numpy.ndarray, (n,9)>
            ratio_w: ratio of width
            ratio_h: ratio of height
    Output:
            refined boxes
    '''
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] *= ratio_w
    boxes[:, [1, 3, 5, 7]] *= ratio_h
    return np.around(boxes)


def is_valid_poly(res, score_shape, scale):
    '''check if the poly in image scope
    Input:
            res        : restored poly in original image
            score_shape: score map shape
            scale      : feature map -> image
    Output:
            True if valid
    '''
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_rectangle_quad(origin, geometry):
    """
    Restore rectangle from quadrangle.
    """
    # quad
    origin_concat = np.concatenate(
        (origin, origin, origin, origin), axis=1)  # (n, 8)
    pred_quads = origin_concat - geometry
    pred_quads = pred_quads.reshape((-1, 4, 2))  # (n, 4, 2)
    return pred_quads


def get_boxes(score, geo, score_thresh=0.8, nms_thresh=0.2, cover_thresh=0.1):
    """
    restore text boxes from score map and geo map
    """
    score_map = score[0]
    geo_map = np.swapaxes(geo, 1, 0)
    geo_map = np.swapaxes(geo_map, 1, 2)
    # filter the score map
    xy_text = np.argwhere(score_map > score_thresh)
    if len(xy_text) == 0:
        return []

    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore quad proposals
    text_box_restored = restore_rectangle_quad(
        xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    
    import lanms
    boxes = lanms.merge_quadrangle_n9(boxes, nms_thresh)

    # boxes = nms_locality(boxes.astype(np.float64), nms_thresh)
    if boxes.shape[0] == 0:
        return []
    
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape(
            (-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > cover_thresh]
    
    return boxes
