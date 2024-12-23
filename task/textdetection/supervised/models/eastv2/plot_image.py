import matplotlib.pyplot as plt


def plot_image(image, score_map, training_mask):
    # Create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=3)

    # Display the image
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].set_axis_off()
    ax[1].imshow(score_map, interpolation="nearest", cmap='jet')
    ax[1].set_title("Score map")
    ax[1].set_axis_off()
    ax[2].imshow(training_mask, interpolation="nearest", cmap='jet')
    ax[2].set_title("Training map")
    ax[2].set_axis_off()

    def on_key(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_release_event', on_key)

    plt.tight_layout()
    plt.show()
