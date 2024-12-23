import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image(image, target, class_names, normalize_box: bool):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]

    im = image
    h, w, _ = image.shape

    boxes = target["boxes"][0]
    if normalize_box:
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

    labels = target["labels"][0]

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    for box, label in zip(boxes, labels):
        label = label.item()
        xmin, ymin, xmax, ymax = map(int, box)
        w, h = xmax - xmin, ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin),
            w,
            h,
            linewidth=2,
            edgecolor=colors[label],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            xmin,
            ymin,
            s=class_names[label],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[label], "pad": 0},
        )

    def on_key(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_release_event', on_key)

    plt.show()
