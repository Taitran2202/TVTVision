import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image(image, density_gt, boxes, cnt_gt):
    # Create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=3)
    # Display the image
    ax[0].imshow(image)
    ax[0].set_title("Augmented image")
    ax[0].set_axis_off()

    ax[1].imshow(image)
    # draw each box on the image
    for box in boxes:
        y1, x1, y2, x2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax[1].add_patch(rect)
    ax[1].set_title("Augmented image")
    ax[1].set_axis_off()

    ax[2].imshow(density_gt)
    ax[2].set_title("Count: {}".format(cnt_gt))
    ax[2].set_axis_off()

    def on_key(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_release_event', on_key)

    plt.show()
