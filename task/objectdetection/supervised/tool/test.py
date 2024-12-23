import os
import cv2
import time
import numpy as np
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from ..data import create_dataset
from ..trackers import ByteTracker
from ..utils.plot_tracking import plot_tracking
from utils.post_processing.objectdetection.yolo import test_net


def load_model(cfg, model_name, target):
    testset = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=target,
        is_train=False,
        class_names=cfg.DATASET.class_names,
        resize=cfg.DATASET.resize[0],
        keep_difficult=cfg.DATASET.keep_difficult,
        use_mosaic=cfg.DATASET.use_mosaic,
        use_mixup=cfg.DATASET.use_mixup
    )
    global device
    use_gpu = cfg.TRAIN.use_gpu
    device = torch.device(
        "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    # build Model
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{model_name.split('-')[1]}"
    model_path = f"task.objectdetection.supervised.models.{cfg.EXP_NAME.split('-')[0].lower()}"
    model = importlib.import_module(
        model_path).build_model(cfg=cfg, device=device, size=cfg.EXP_NAME.split('-')[1].lower())
    model.to(device)
    savedir = os.path.join(cfg.RESULT.savedir, cfg.TASK,
                           cfg.METHOD, target, cfg.EXP_NAME)
    resume = cfg.RESUME.bestmodel
    checkpoint = torch.load(f'{savedir}/{resume}', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    params = {
        'resize': cfg.DATASET.resize,
        'class_names': cfg.DATASET.class_names,
        'vis_thresh': cfg.TEST.vis_thresh,
        'savedir': savedir
    }
    return testset, model, params


def result_plot(idx, model, testset, params, test_img):
    resize = params['resize']
    class_names = params['class_names']
    vis_thresh = params['vis_thresh']

    if test_img:
        img_path = testset.file_list[idx]
    else:
        img_path, target = testset[idx]

    input_i = cv2.imread(img_path)
    input_i = cv2.cvtColor(input_i, cv2.COLOR_BGR2RGB)
    img1 = input_i.copy()
    img2 = input_i.copy()

    bboxes, scores, labels = test_net(
        model,
        input_i,
        device,
        resize
    )

    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]

    # Create figure and axes
    fig, ax = plt.subplots(1, 2 if test_img else 3, figsize=(15, 10))

    ax[0].imshow(input_i)
    ax[0].set_title('Input')

    ax[1].imshow(img1)
    # Create a Rectangle patch
    for box, label, score in zip(bboxes, labels, scores):
        if score > vis_thresh:
            xmin, ymin, xmax, ymax = map(int, box)
            w, h = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin),
                w,
                h,
                linewidth=2,
                edgecolor=colors[int(label)],
                facecolor="none",
            )
            # Add the patch to the Axes
            ax[1].add_patch(rect)
            text = '%s: %.2f' % (class_names[int(label)], score)
            ax[1].text(
                xmin,
                ymin,
                s=text,
                color="white",
                verticalalignment="top",
                bbox={"color": colors[int(label)], "pad": 0},
            )
    ax[1].set_title('Model Predicted')

    if not test_img:
        ax[2].imshow(img2)
        # Create a Rectangle patch
        for box, label in zip(target["boxes"], target["labels"]):
            xmin, ymin, xmax, ymax = map(int, box)
            w, h = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin),
                w,
                h,
                linewidth=2,
                edgecolor=colors[int(label)],
                facecolor="none",
            )
            # Add the patch to the Axes
            ax[2].add_patch(rect)
            ax[2].text(
                xmin,
                ymin,
                s=class_names[int(label)],
                color="white",
                verticalalignment="top",
                bbox={"color": colors[int(label)], "pad": 0},
            )
        ax[2].set_title('Target Ground Truth')

    plt.show()


def result_save_plot(cfg, model, testset, params, test_img):
    resize = params['resize']
    class_names = params['class_names']
    vis_thresh = params['vis_thresh']

    if not test_img:
        savedir = os.path.join(params['savedir'], cfg.RESULT.saveall)
    else:
        savedir = os.path.join(params['savedir'], cfg.RESULT.save_test_img)
    os.makedirs(savedir, exist_ok=True)

    for idx, file_path in enumerate(tqdm(testset.file_list)):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_prefix = os.path.basename(os.path.dirname(file_path))
        save_path = f"{savedir}/result_{file_prefix}_{file_name}.png"

        if test_img:
            img_path = testset.file_list[idx]
        else:
            img_path, target = testset[idx]

        input_i = cv2.imread(img_path)
        input_i = cv2.cvtColor(input_i, cv2.COLOR_BGR2RGB)
        img1 = input_i.copy()
        img2 = input_i.copy()

        bboxes, scores, labels = test_net(
            model,
            input_i,
            device,
            resize
        )

        """Plots predicted bounding boxes on the image"""
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]

        # Create figure and axes
        fig, ax = plt.subplots(1, 2 if test_img else 3, figsize=(15, 10))

        ax[0].imshow(input_i)
        ax[0].set_title('Input')

        ax[1].imshow(img1)
        # Create a Rectangle patch
        for box, label, score in zip(bboxes, labels, scores):
            if score > vis_thresh:
                xmin, ymin, xmax, ymax = map(int, box)
                w, h = xmax - xmin, ymax - ymin
                rect = patches.Rectangle(
                    (xmin, ymin),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=colors[int(label)],
                    facecolor="none",
                )
                # Add the patch to the Axes
                ax[1].add_patch(rect)
                text = '%s: %.2f' % (class_names[int(label)], score)
                ax[1].text(
                    xmin,
                    ymin,
                    s=text,
                    color="white",
                    verticalalignment="top",
                    bbox={"color": colors[int(label)], "pad": 0},
                )
        ax[1].set_title('Model Predicted')

        if not test_img:
            ax[2].imshow(img2)
            # Create a Rectangle patch
            for box, label in zip(target["boxes"], target["labels"]):
                xmin, ymin, xmax, ymax = map(int, box)
                w, h = xmax - xmin, ymax - ymin
                rect = patches.Rectangle(
                    (xmin, ymin),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=colors[int(label)],
                    facecolor="none",
                )
                # Add the patch to the Axes
                ax[2].add_patch(rect)
                ax[2].text(
                    xmin,
                    ymin,
                    s=class_names[int(label)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": colors[int(label)], "pad": 0},
                )
            ax[2].set_title('Target Ground Truth')

        plt.savefig(save_path)
        plt.close(fig)


def track_camera_run(cfg, model, params):
    savedir = os.path.join(params['savedir'], cfg.RESULT.save_track_camera)
    os.makedirs(savedir, exist_ok=True)

    tracker = ByteTracker(
        track_thresh=cfg.VIDEO.track_thresh,
        track_buffer=cfg.VIDEO.track_buffer,
        frame_rate=cfg.VIDEO.frame_rate,
        match_thresh=cfg.VIDEO.match_thresh,
        mot20=cfg.VIDEO.mot20
    )

    resize = params['resize']
    class_names = params['class_names']
    aspect_ratio_thresh = cfg.VIDEO.aspect_ratio_thresh
    min_box_area = cfg.VIDEO.min_box_area

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_id = 0

    # For saving
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_size = (640, 480)
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    save_video_name = os.path.join(savedir, cur_time + '.avi')
    fps = 15.0
    out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)

    while True:
        ret, frame = cap.read()
        if ret:
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

            # ------------------------- Detection ---------------------------
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            bboxes, scores, labels = test_net(
                model,
                frame,
                device,
                resize
            )
            # track
            if len(bboxes) > 0:
                online_targets = tracker.update(scores, bboxes, labels)
                online_xywhs = []
                online_ids = []
                online_scores = []
                online_labels = []
                for t in online_targets:
                    xywh = t.xywh
                    tid = t.track_id
                    vertical = xywh[2] / xywh[3] > aspect_ratio_thresh
                    if xywh[2] * xywh[3] > min_box_area and not vertical:
                        online_xywhs.append(xywh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_labels.append(t.label)

                # plot tracking results
                online_im = plot_tracking(frame, online_xywhs, online_scores, online_labels, online_ids,
                                          frame_id + 1, 1. / (time.time() - t0), class_names, cfg['VIDEO']['is_visual_class'])
            else:
                online_im = frame

            frame_resized = cv2.resize(online_im, save_size)
            out.write(frame_resized[:, :, ::-1].copy())
            cv2.imshow('Tracking Camera', online_im[:, :, ::-1].copy())

        else:
            break
        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def track_video_run(cfg, model, params):
    savedir = os.path.join(params['savedir'], cfg.RESULT.save_track_video)
    os.makedirs(savedir, exist_ok=True)

    tracker = ByteTracker(
        track_thresh=cfg.VIDEO.track_thresh,
        track_buffer=cfg.VIDEO.track_buffer,
        frame_rate=cfg.VIDEO.frame_rate,
        match_thresh=cfg.VIDEO.match_thresh,
        mot20=cfg.VIDEO.mot20
    )

    resize = params['resize']
    class_names = params['class_names']
    aspect_ratio_thresh = cfg.VIDEO.aspect_ratio_thresh
    min_box_area = cfg.VIDEO.min_box_area

    # read a video
    video = cv2.VideoCapture(params['path_to_vid'])
    fps = video.get(cv2.CAP_PROP_FPS)

    # For saving
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_size = (640, 480)
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    save_video_name = os.path.join(savedir, cur_time + '.avi')
    out = cv2.VideoWriter(save_video_name, fourcc, fps, save_size)

    # start tracking
    frame_id = 0
    while True:
        ret, frame = video.read()
        if ret:
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

            # ------------------------- Detection ---------------------------
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            bboxes, scores, labels = test_net(
                model,
                frame,
                device,
                resize
            )
            # track
            if len(bboxes) > 0:
                online_targets = tracker.update(scores, bboxes, labels)
                online_xywhs = []
                online_ids = []
                online_scores = []
                online_labels = []
                for t in online_targets:
                    xywh = t.xywh
                    tid = t.track_id
                    vertical = xywh[2] / xywh[3] > aspect_ratio_thresh
                    if xywh[2] * xywh[3] > min_box_area and not vertical:
                        online_xywhs.append(xywh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        online_labels.append(t.label)

                # plot tracking results
                online_im = plot_tracking(frame, online_xywhs, online_scores, online_labels, online_ids,
                                          frame_id + 1, 1. / (time.time() - t0), class_names, cfg['VIDEO']['is_visual_class'])
            else:
                online_im = frame

            frame_resized = cv2.resize(online_im, save_size)
            out.write(frame_resized[:, :, ::-1].copy())
            cv2.imshow('Tracking Camera', online_im[:, :, ::-1].copy())

        else:
            break
        frame_id += 1

    video.release()
    out.release()
    cv2.destroyAllWindows()
