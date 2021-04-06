import torch.nn as nn


class PositionAwareLayer(nn.Module):

    def __init__(self, dim_model, rnn_layers=2):
        super().__init__()

        self.dim_model = dim_model

        self.rnn = nn.LSTM(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=rnn_layers,
            batch_first=True)

        self.mixer = nn.Sequential(
            nn.Conv2d(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(
                dim_model, dim_model, kernel_size=3, stride=1, padding=1))

    def forward(self, img_feature):
        n, c, h, w = img_feature.size()

        rnn_input = img_feature.permute(0, 2, 3, 1).contiguous()
        rnn_input = rnn_input.view(n * h, w, c)
        rnn_output, _ = self.rnn(rnn_input)
        rnn_output = rnn_output.view(n, h, w, c)
        rnn_output = rnn_output.permute(0, 3, 1, 2).contiguous()

        out = self.mixer(rnn_output)

        return out
