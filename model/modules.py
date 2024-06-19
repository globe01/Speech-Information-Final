import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 本文件主要是定义Variance Adaptor
# 其中主要包括Duration Predictor、Length Regulator、Pitch Predictor和Energy Predictor
# 完整Variance Adaptor
class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        # 设置pitch和energy的级别
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        # 设置pitch和energy的量化方式
        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        # 加载pitch和energy的正则化所需参数
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        # pitch和energy的嵌入层
        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    # 计算pitch嵌入层
    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)# pitch预测器预测的数值
        if target is not None:# target存在，训练过程，使用target计算embedding
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else: # target不存在，预测过程，使用prediction计算embedding
            prediction = prediction * control# control是用于控制的系数
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding# prediction用于训练过程计算损失，embedding与x相加进行后续计算

    # 计算energy嵌入层
    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)# energy预测器预测的数值
        if target is not None:# target存在，训练过程，使用target计算embedding
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:# target不存在，预测过程，使用prediction计算embedding
            prediction = prediction * control# control是用于控制的系数
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding# prediction用于训练过程计算损失，embedding与x相加进行后续计算

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)# 对音素序列预测的持续时间
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding # 累加pitch嵌入层
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding # 累加energy嵌入层

        if duration_target is not None:# duration_target，训练过程，使用duration_target计算
            x, mel_len = self.length_regulator(x, duration_target, max_len)# 使用duration_target调整x
            duration_rounded = duration_target
        else:# 预测过程
            # 基于log_duration_prediction构建duration_rounded，用于调整x
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)
        # 与phoneme_level一致, frame_level的pitch和energy也是在这里计算的
        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,# 经过Variance Adaptor处理后的x
            pitch_prediction,# pitch预测值
            energy_prediction,# energy预测值
            log_duration_prediction,# 音素序列预测的持续时间
            duration_rounded,# 基于log_duration_prediction构建的duration_rounded
            mel_len,# mel的长度
            mel_mask,# mel的mask
        )

# 长度调整器
class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    # 对输入的x进行调整，使其长度与duration_target一致
    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)# 对batch进行扩展
            output.append(expanded)
            mel_len.append(expanded.shape[0])# 记录扩展后的长度

        # 如果传入了max_len，对output进行padding
        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    # 对batch进行扩展
    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()# 预测的持续时间
            out.append(vec.expand(max(int(expand_size), 0), -1))# 对vec进行扩展
        out = torch.cat(out, 0)# 拼接整个音素序列

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]# 输入的维度
        self.filter_size = model_config["variance_predictor"]["filter_size"]# 卷积层的输出维度
        self.kernel = model_config["variance_predictor"]["kernel_size"]# 卷积核的大小
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]# 卷积层的输出维度
        self.dropout = model_config["variance_predictor"]["dropout"]# dropout的概率

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)# 卷积层
        out = self.linear_layer(out)# 线性层
        out = out.squeeze(-1)# 去掉最后一维

        if mask is not None:
            out = out.masked_fill(mask, 0.0)# mask处理

        return out

# 卷积层（1维）
class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
