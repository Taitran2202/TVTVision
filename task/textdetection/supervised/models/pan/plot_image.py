import matplotlib.pyplot as plt


def plot_image(image, gt_texts, gt_kernels, training_masks, gt_instances):
    # Create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=5)

    # Display the image
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].set_axis_off()
    ax[1].imshow(gt_texts, interpolation="nearest", cmap='jet')
    ax[1].set_title("Gt Texts")
    ax[1].set_axis_off()
    ax[2].imshow(gt_kernels, interpolation="nearest", cmap='jet')
    ax[2].set_title("Gt Kernels")
    ax[2].set_axis_off()
    ax[3].imshow(training_masks, interpolation="nearest", cmap='jet')
    ax[3].set_title("Training Masks")
    ax[3].set_axis_off()
    ax[4].imshow(gt_instances, interpolation="nearest", cmap='jet')
    ax[4].set_title("GT Instances")
    ax[4].set_axis_off()

    def on_key(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_release_event', on_key)

    plt.tight_layout()
    plt.show()
