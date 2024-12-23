import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_result(img_path, img, target, pre_bboxes, pre_scores, pre_labels, class_names, result_dir=""):
    img = np.array(img)
    im1 = img.copy()
    im2 = img.copy()

    # make result file list
    filename, _ = os.path.splitext(os.path.basename(img_path))

    # draw bounding boxes
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]

    boxes = target["boxes"]
    labels = target["labels"]

    # img, gt truth
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = map(int, box)
        color = [int(c * 255) for c in colors[int(label)]]

        cv2.rectangle(im1, (xmin, ymin), (xmax, ymax), color, 2)
        label_text = class_names[int(label)]
        cv2.putText(im1, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # img, predicted
    for box, label, score in zip(pre_bboxes, pre_labels, pre_scores):
        xmin, ymin, xmax, ymax = map(int, box)
        color = [int(c * 255) for c in colors[int(label)]]

        cv2.rectangle(im2, (xmin, ymin), (xmax, ymax), color, 2)
        label_text = f"{class_names[int(label)]}: {score:.2f}"
        cv2.putText(im2, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    temp = np.hstack([im1, im2])

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, temp)
