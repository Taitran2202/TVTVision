import matplotlib.pyplot as plt


def plot_image(image, region_score, affinity_score, confidence_mask):
    # Create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=4)

    # Display the image
    ax[0].imshow(image)
    ax[0].set_title("Augmented image")
    ax[0].set_axis_off()
    ax[1].imshow(region_score, interpolation="nearest", cmap='jet')
    ax[1].set_title("Augmented region score")
    ax[1].set_axis_off()
    ax[2].imshow(affinity_score, interpolation="nearest", cmap='jet')
    ax[2].set_title("Augmented affinity score")
    ax[2].set_axis_off()
    ax[3].imshow(confidence_mask, interpolation="nearest", cmap='jet')
    ax[3].set_title("Augmented confidence mask")
    ax[3].set_axis_off()

    def on_key(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_release_event', on_key)

    plt.tight_layout()
    plt.show()
