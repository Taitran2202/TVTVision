import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.post_processing.objectdetection.yolo import test_net
from task.objectdetection.supervised.utils.save_result import save_result


class VOCAPIEvaluator():
    """ VOC AP Evaluation class """

    def __init__(self) -> None:
        pass

    def calculate_iou(self, box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1_gt, y1_gt, x2_gt, y2_gt = box2

        xi1 = max(x1, x1_gt)
        yi1 = max(y1, y1_gt)
        xi2 = min(x2, x2_gt)
        yi2 = min(y2, y2_gt)

        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def compute_ap(self, rec, prec):
        """Compute the Average Precision given recall and precision arrays."""
        # Append sentinel values to the arrays
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Find the points where the recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Compute the interpolated precision at the change points
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def evaluate(self, net, validset, num_classes, result_dir='', device='cpu'):
        net.eval()

        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(len(validset))]
                     for _ in range(num_classes)]

        for i, (img_path, target) in enumerate(tqdm(validset)):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes, scores, labels = test_net(
                net, image, device, (validset.resize, validset.resize))

            save_result(
                img_path=img_path,
                img=image[:, :, ::-1].copy(),
                target=target,
                pre_bboxes=bboxes,
                pre_scores=scores,
                pre_labels=labels,
                class_names=validset.class_names,
                result_dir=result_dir
            )

            for j in range(num_classes):
                inds = np.where(labels == j)[0]
                if len(inds) == 0:
                    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue

                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)
                all_boxes[j][i] = c_dets

        # Calculate mAP for each class
        average_precisions = defaultdict(list)
        for cls in range(num_classes):
            true_positives = []
            num_gt_boxes = 0

            # for image_idx in range(len(validset)):
            for image_idx, (_, target) in enumerate(validset):
                # Get predictions and ground truth for the current class
                predictions = all_boxes[cls][image_idx]
                gt_boxes = target["boxes"][target["labels"] == cls]
                num_gt_boxes += len(gt_boxes)

                # Check if there are any true objects for this class in the image
                if len(gt_boxes) == 0:
                    continue

                # Assign detections to ground truth objects
                for box in gt_boxes:
                    if len(predictions) == 0:
                        true_positives.append(0)
                        continue

                    ious = [self.calculate_iou(detection[:4], box)
                            for detection in predictions]
                    max_iou = max(ious)
                    if max_iou >= 0.5:
                        true_positives.append(1)
                        index = ious.index(max_iou)
                        predictions = np.delete(predictions, index, axis=0)
                    else:
                        true_positives.append(0)

            true_positives = np.array(true_positives)

            # Compute precision and recall
            false_positives = 1 - true_positives
            true_positives = np.cumsum(true_positives)
            false_positives = np.cumsum(false_positives)
            recall = true_positives / float(num_gt_boxes)
            precision = true_positives / \
                np.maximum(true_positives + false_positives,
                           np.finfo(np.float64).eps)

            precision = np.concatenate(([1], precision))
            recall = np.concatenate(([0], recall))

            # Compute AP and store it for this class
            ap = self.compute_ap(recall, precision)
            average_precisions[cls].append(ap)

        # Compute mean AP across all classes
        mean_ap = np.mean(list(average_precisions.values()))
        return mean_ap
