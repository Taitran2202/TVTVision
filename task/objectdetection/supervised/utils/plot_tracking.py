import cv2
import numpy as np


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, boxes, scores, labels, obj_ids, frame_id=0, fps=0., class_names=None, is_visual_class=False):
    im = np.ascontiguousarray(np.copy(image))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    cv2.putText(im, 'frame: %d fps: %.2f num_obj: %d' % (frame_id, fps, len(boxes)), (0, int(
        15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, w, h = box
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        if is_visual_class:
            id_text = 'id: %s, %s: %.2f' % (
                int(obj_id), class_names[int(label)], score)
        else:
            id_text = 'id: %s' % (int(obj_id))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4],
                      color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (0, 0, 255), thickness=text_thickness)

    return im
