import matplotlib.pyplot as plt


def plot_image(image, label, idx_to_class):
    # Create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=2)

    # Display the image
    ax[0].imshow(image)
    ax[0].set_title(f'Label: {idx_to_class[label]}')
    ax[0].set_axis_off()
    ax[1].imshow(image)
    ax[1].set_title(f'Class No: {str(label.item())}')
    ax[1].set_axis_off()

    def on_key(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_release_event', on_key)

    plt.tight_layout()
    plt.show()
