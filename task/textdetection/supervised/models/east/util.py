import os
import cv2
import numpy as np


def overlay(image, single_img_bbox):
    boxed_img = image.copy()
    for word_box in single_img_bbox:
        cv2.polylines(
            boxed_img,
            [np.array(word_box).astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=(0, 255, 0),
            thickness=3,
        )

    temp1 = np.hstack([image, boxed_img])

    return temp1


def save_result_synth(img_path, img, pre_box, gt_box=None, result_dir=""):
    img = np.array(img)
    img_copy = img.copy()

    # make result file list
    filename, _ = os.path.splitext(os.path.basename(img_path))

    # draw bounding boxes for prediction, color green
    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(
                img,
                [np.array(gt_box[j]["points"]).astype(
                    np.int32).reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

    # draw overlay image
    overlay_img = overlay(img_copy, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def save_result_icdar13(img_path, img, pre_box, gt_box=None, result_dir=""):
    img = np.array(img)
    img_copy = img.copy()

    # make result file list
    filename, _ = os.path.splitext(os.path.basename(img_path))

    for _, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(
                img,
                [np.array(gt_box[j]["points"]).reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

    # draw overlay image
    overlay_img = overlay(img_copy, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def save_result_icdar1517(img_path, img, pre_box, gt_box=None, result_dir=""):
    img = np.array(img)
    img_copy = img.copy()

    # make result file list
    filename, _ = os.path.splitext(os.path.basename(img_path))

    for _, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            _gt_box = np.array(
                gt_box[j]["points"]).reshape(-1, 2).astype(np.int32)
            if gt_box[j]["text"] == "###":
                cv2.polylines(img, [_gt_box], True, color=(
                    128, 128, 128), thickness=2)
            else:
                cv2.polylines(img, [_gt_box], True,
                              color=(0, 0, 255), thickness=2)

    # draw overlay image
    overlay_img = overlay(img_copy, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)
