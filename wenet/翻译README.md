# WeNet

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**路线图**](https://github.com/wenet-e2e/wenet/issues/1683)
| [**文档**](https://wenet-e2e.github.io/wenet)
| [**论文**](https://wenet-e2e.github.io/wenet/papers.html)
| [**运行时**](https://github.com/wenet-e2e/wenet/tree/main/runtime)
| [**预训练模型**](docs/pretrained_models.md)
| [**HuggingFace**](https://huggingface.co/spaces/wenet/wenet_demo)

**We** share **Net** together.

## 亮点

* **生产优先和生产就绪**：WeNet 的核心设计原则是提供完整的语音识别生产解决方案。
* **精准**：WeNet 在许多公共语音数据集上取得了最新的技术成果。
* **轻量级**：WeNet 易于安装、使用简单、设计良好且文档详尽。

## 安装

### 安装 Python 包

``` sh
pip install git+https://github.com/wenet-e2e/wenet.git
```

**命令行使用**（使用 `-h` 查看参数）：

``` sh
wenet --language chinese audio.wav
```

**Python 编程使用**：

``` python
import wenet

model = wenet.load_model('chinese')
result = model.transcribe('audio.wav')
print(result['text'])
```

请参阅 [python usage](docs/python_package.md) 了解更多命令行和 Python 编程使用。

### 安装用于训练和部署

- 克隆仓库

``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- 安装 Conda：请参阅 https://docs.conda.io/en/latest/miniconda.html
- 创建 Conda 环境：

``` sh
conda create -n wenet python=3.10
conda activate wenet
conda install conda-forge::sox
pip install -r requirements.txt
pre-commit install  # 用于保持代码清洁

# 如果遇到 sox 兼容性问题
RuntimeError: set_buffer_size requires sox extension which is not available.
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
# conda env
conda install  conda-forge::sox
```

**为部署构建**

如果你想使用 x86 运行时或语言模型 (LM)，你需要按如下步骤构建运行时。否则，你可以忽略这一步。

``` sh
# 构建运行时需要 cmake 3.14 或以上版本
cd runtime/libtorch
mkdir build && cd build && cmake -DGRAPH_TOOLS=ON .. && cmake --build .
```

请参阅 [文档](https://github.com/wenet-e2e/wenet/tree/main/runtime) 了解更多平台和操作系统上的运行时构建。

## 讨论和交流

你可以在 [Github Issues](https://github.com/wenet-e2e/wenet/issues) 上直接讨论。

对于中国用户，你也可以扫描左边的二维码关注 WeNet 官方账号。我们创建了一个微信讨论组以便更好地讨论和更快地响应。请扫描右边的个人二维码，我们将邀请你加入讨论组。

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/robin1001/qr/blob/master/binbin.jpeg" width="250px"> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

## 致谢

1. 我们借用了 [ESPnet](https://github.com/espnet/espnet) 的很多代码来进行基于 Transformer 的建模。
2. 我们借用了 [Kaldi](http://kaldi-asr.org/) 的很多代码来进行基于 WFST 的解码和 LM 集成。
3. 我们参考了 [EESEN](https://github.com/srvk/eesen) 来构建用于 LM 集成的 TLG 图。
4. 我们参考了 [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer/) 来进行 e2e 模型的 Python 批处理推理。

## 引用

``` bibtex
@inproceedings{yao2021wenet,
title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
  booktitle={Proc. Interspeech},
  year={2021},
  address={Brno, Czech Republic },
  organization={IEEE}
}

@article{zhang2022wenet,
  title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
  author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
  journal={arXiv preprint arXiv:2203.15455},
  year={2022}
}
```