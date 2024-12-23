import cv2
import numpy as np
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
    boxes[:, [0, 2, 4, 6]] *= ratio_w * 4
    boxes[:, [1, 3, 5, 7]] *= ratio_h * 4
    return np.around(boxes)


def line_cross_point(A1, B1, C1, A2, B2, C2):
    """
    F0(x) = a0*x + b0*y + c0 = 0, F1(x) = a1*x + b1*y + c1 = 0
    x = (b0*c1 – b1*c0)/D
    y = (a1*c0 – a0*c1)/D
    D = a0*b1 – a1*b0，
    """
    D = A1*B2 - A2*B1
    x = (B1*C2 - B2*C1) / (D + 1e-8)
    y = (A2*C1 - A1*C2) / (D + 1e-8)
    return x, y


def restore_bboxes(cls, rho, theta, cls_thresh=None, nms_thresh=None, nms=True):
    # AX+BY+C=0  <---> ρ=x*cosθ+y*sinθ
    height = cls.shape[1]

    region = (cls[0, :, :] >= cls_thresh)
    ret, markers = cv2.connectedComponents(np.uint8(region))
    for i in range(1, ret):
        y, x = np.where(markers == i)
        if len(x) > 0:
            continue
        for j in range(len(x)):
            region[y[j], x[j]] = False

    r, c = np.where(region)
    # the 'C' is actually "C + A*dx + B*dy"
    A, B, C = np.cos(theta), np.sin(theta), -rho
    A0, B0, C0 = A[0, r, c], B[0, r, c], C[0, r, c]
    C0 -= A0*c + B0*(height-r)  # recover C in global coordinate
    A1, B1, C1 = A[1, r, c], B[1, r, c], C[1, r, c]
    C1 -= A1*c + B1*(height-r)  # recover C in global coordinate
    A2, B2, C2 = A[2, r, c], B[2, r, c], C[2, r, c]
    C2 -= A2*c + B2*(height-r)  # recover C in global coordinate
    A3, B3, C3 = A[3, r, c], B[3, r, c], C[3, r, c]
    C3 -= A3*c + B3*(height-r)  # recover C in global coordinate

    # lines -> bboxes
    x1, y1 = line_cross_point(A0, B0, C0, A1, B1, C1)
    x2, y2 = line_cross_point(A1, B1, C1, A2, B2, C2)
    x3, y3 = line_cross_point(A2, B2, C2, A3, B3, C3)
    x0, y0 = line_cross_point(A3, B3, C3, A0, B0, C0)
    bboxes = np.vstack((x0, height-y0, x1, height-y1, x2,
                       height-y2, x3, height-y3, cls[0, r, c])).T
    if nms:
        bboxes = nms_locality(
            bboxes.astype('float32'), nms_thresh)
    return bboxes
