import torch.nn as nn


class CNN_Backbone(nn.Module):
    def __init__(self, img_channel, img_height, img_width, leaky_relu):
        super(CNN_Backbone, self).__init__()
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.ModuleList()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.append(
                nn.Conv2d(input_channel, output_channel,
                          kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.append(nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(
                0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.append(relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        # (256, img_height // 8, img_width // 4)
        cnn.append(nn.MaxPool2d(kernel_size=(2, 1)))

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        # (512, img_height // 16, img_width // 4)
        cnn.append(nn.MaxPool2d(kernel_size=(2, 1)))

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        self.cnn = nn.Sequential(*cnn)

        self.output_channel, self.output_height, self.output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1

    def forward(self, x):
        x = self.cnn(x)
        return x
