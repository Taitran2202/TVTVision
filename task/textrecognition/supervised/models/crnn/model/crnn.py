import torch.nn as nn
import torch.nn.functional as F
from .module import CNN_Backbone


class CRNN(nn.Module):
    def __init__(self, img_channel: int, img_height: int, img_width: int, num_class: int,
                 map_to_seq_hidden: int = 64, rnn_hidden: int = 256, leaky_relu: bool = False):
        super(CRNN, self).__init__()
        self.cnn = CNN_Backbone(img_channel, img_height, img_width, leaky_relu)
        self.map_to_seq = nn.Linear(
            self.cnn.output_channel * self.cnn.output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        return F.log_softmax(self.dense(recurrent), dim=2)
