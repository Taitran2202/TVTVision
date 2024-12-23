import cv2
import numpy as np
from queue import Queue


# def process_point(x, y, l, kernal, pred, queue, next_queue, dx, dy):
#     is_edge = True
#     for j in range(4):
#         tmpx = x + dx[j]
#         tmpy = y + dy[j]
#         if 0 <= tmpx < kernal.shape[0] and 0 <= tmpy < kernal.shape[1]:
#             if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
#                 continue

#             queue.put((tmpx, tmpy, l))
#             pred[tmpx, tmpy] = l
#             is_edge = False
#     if is_edge:
#         next_queue.put((x, y, l))


# def pse(kernals, min_area):
#     kernal_num = len(kernals)
#     pred = np.zeros(kernals[0].shape, dtype='int32')

#     label_num, label = cv2.connectedComponents(
#         kernals[kernal_num - 1], connectivity=4)

#     for label_idx in range(1, label_num):
#         if np.sum(label == label_idx) < min_area:
#             label[label == label_idx] = 0

#     queue = Queue(maxsize=0)
#     next_queue = Queue(maxsize=0)
#     points = np.argwhere(label > 0)

#     for point in points:
#         x, y = point
#         l = label[x, y]
#         queue.put((x, y, l))
#         pred[x, y] = l

#     dx = [-1, 1, 0, 0]
#     dy = [0, 0, -1, 1]

#     for kernal_idx in range(kernal_num - 2, -1, -1):
#         kernal = kernals[kernal_idx].copy()

#         num_threads = 4  # Adjust the number of threads as needed
#         threads = []

#         while not queue.empty():
#             thread_chunk = []
#             for _ in range(num_threads):
#                 if not queue.empty():
#                     thread_chunk.append(queue.get())
#             for chunk in thread_chunk:
#                 thread = threading.Thread(target=process_point, args=(
#                     chunk[0], chunk[1], chunk[2], kernal, pred, queue, next_queue, dx, dy))
#                 thread.start()
#                 threads.append(thread)

#             for thread in threads:
#                 thread.join()

#             queue, next_queue = next_queue, queue

#     return pred

def pse(kernels, min_area):
    kernel_num = len(kernels)
    pred = np.zeros(kernels[0].shape, dtype='int32')

    _, label = cv2.connectedComponents(kernels[kernel_num - 1], connectivity=4)

    valid_labels = np.unique(label)[1:]
    valid_sizes = np.bincount(label.flat)[1:]
    small_labels = valid_labels[valid_sizes < min_area]
    for small_label in small_labels:
        label[label == small_label] = 0

    points = np.argwhere(label > 0)
    queue = Queue(maxsize=0)
    next_queue = Queue(maxsize=0)

    for point in points:
        x, y = point
        l = label[x, y]
        queue.put((x, y, l))
        pred[x, y] = l

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for kernel_idx in range(kernel_num - 2, -1, -1):
        kernel = kernels[kernel_idx].copy()
        while not queue.empty():
            x, y, l = queue.get()
            is_edge = True
            for j in range(4):
                tmpx = x + dx[j]
                tmpy = y + dy[j]
                if 0 <= tmpx < kernel.shape[0] and 0 <= tmpy < kernel.shape[1] and kernel[tmpx, tmpy] and not pred[tmpx, tmpy]:
                    queue.put((tmpx, tmpy, l))
                    pred[tmpx, tmpy] = l
                    is_edge = False
            if is_edge:
                next_queue.put((x, y, l))
        queue, next_queue = next_queue, queue

    return pred
