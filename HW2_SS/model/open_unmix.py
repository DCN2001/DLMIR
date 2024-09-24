from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class OpenUnmix(nn.Module):
    def __init__(
        self,
        nb_bins: int = 1025,
        nb_channels: int = 2,
        hidden_size: int = 512,
        nb_layers: int = 3,        #3
        unidirectional: bool = False,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
    ):
        super(OpenUnmix, self).__init__()

        self.nb_bins = nb_bins
        self.nb_output_bins = nb_bins
        self.hidden_size = hidden_size

        # Define layers
        self.fc1 = self._linear_layer(self.nb_bins * nb_channels, hidden_size)
        self.lstm = self._lstm_layer(hidden_size, nb_layers, unidirectional)
        self.fc2 = self._linear_layer(hidden_size * 2, hidden_size)
        self.fc3 = self._linear_layer(hidden_size, self.nb_output_bins * nb_channels)

        # Initialize parameters
        self.input_mean = self._init_parameter(input_mean, self.nb_bins, fill_value=0)
        self.input_scale = self._init_parameter(1.0 / input_scale if input_scale is not None else None, self.nb_bins, fill_value=1)
        self.output_scale = nn.Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = nn.Parameter(torch.ones(self.nb_output_bins).float())

    def _linear_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features)
        )

    def _lstm_layer(self, hidden_size, nb_layers, unidirectional):
        lstm_hidden_size = hidden_size if unidirectional else hidden_size // 2
        return nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

    def _init_parameter(self, array, size, fill_value):
        param = torch.from_numpy(array[:size]).float() if array is not None else torch.full((size,), fill_value, dtype=torch.float)
        return nn.Parameter(param)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(3, 0, 1, 2)  # Permute to (nb_frames, nb_samples, nb_channels, nb_bins)
        nb_frames, nb_samples, nb_channels, nb_bins = x.shape
        mix = x.detach().clone()

        x = x[..., :self.nb_bins]  # Crop to nb_bins
        x = (x + self.input_mean) * self.input_scale  # Normalize

        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))  # Linear + BatchNorm
        x = torch.tanh(x).reshape(nb_frames, nb_samples, self.hidden_size)

        lstm_out, _ = self.lstm(x)  # LSTM
        x = torch.cat([x, lstm_out], -1)  # Concatenate LSTM output

        x = self.fc2(x.reshape(-1, x.shape[-1]))  # Linear + BatchNorm
        x = F.relu(x)

        x = self.fc3(x)  # Final Linear layer
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x = (F.relu(x) * mix) * self.output_scale + self.output_mean  # Scale and shift output
        x = x.permute(1,2,3,0).contiguous()
        #print(x.shape)     #torch.Size([8, 2, 128, 216])
        return x  # Permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
