import matplotlib.pyplot as plt


def plot_image(image, text, ignored_map):
    # Create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=3)

    # Display the image
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[1].imshow(text, interpolation="nearest", cmap='gray')
    ax[1].set_title('Text')
    ax[2].imshow(ignored_map, interpolation="nearest", cmap='gray')
    ax[2].set_title('Ignored Text')

    def on_key(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_release_event', on_key)

    plt.tight_layout()
    plt.show()
