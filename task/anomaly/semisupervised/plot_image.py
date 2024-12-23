import matplotlib.pyplot as plt


def plot_image(image, target):
    # mode
    mode = len(target) == 2

    # Create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=2 if mode else 3)

    # Display the image
    if mode:
        ax[0].imshow(image)
        ax[0].set_title('Input: {}'.format(
            'Normal' if target['label'] == 0 else 'Abnormal'))
        ax[0].set_axis_off()
        ax[1].imshow(target['mask'][0], interpolation="nearest", cmap='gray')
        ax[1].set_title('Ground Truth')
        ax[1].set_axis_off()
    else:
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].set_axis_off()
        ax[1].imshow(target['augmented_image'][0].permute(1, 2, 0))
        ax[1].set_title('Input: {}'.format(
            'Normal' if target['label'] == 0 else 'Abnormal'))
        ax[1].set_axis_off()
        ax[2].imshow(target['mask'][0], interpolation="nearest", cmap='gray')
        ax[2].set_title('Ground Truth')
        ax[2].set_axis_off()

    def on_key(event):
        if event.key == 'escape':
            plt.close()

    fig.canvas.mpl_connect('key_release_event', on_key)

    plt.tight_layout()
    plt.show()
