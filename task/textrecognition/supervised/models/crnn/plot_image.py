import matplotlib.pyplot as plt


def plot_image(image, target, text):
    # Create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=2)

    # Display the image
    ax[0].imshow(image, interpolation="nearest", cmap='gray')
    ax[0].set_title("Text: {}".format(text))
    ax[0].set_axis_off()
    ax[1].imshow(image, interpolation="nearest", cmap='gray')
    ax[1].set_title("Text: {}".format(target))
    ax[1].set_axis_off()

    def on_key(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_release_event', on_key)

    plt.tight_layout()
    plt.show()
