import os
import glob
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import librosa
from tensorflow.keras.models import load_model

def load_wav(file_path):
    # 读取wav文件
    sr, wav_data = wavfile.read(file_path)
    return wav_data, sr

def preprocess_wav(wav_data, sr, n_fft=1024, hop_length=256, n_mels=257):
    # 使用librosa计算Mel频谱图
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav_data.astype(float), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # 转换为对数刻度（dB）
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # 标准化
    log_mel_spectrogram = (log_mel_spectrogram - np.mean(log_mel_spectrogram)) / np.std(log_mel_spectrogram)
    # 转置为（时间, Mel）
    log_mel_spectrogram = log_mel_spectrogram.T
    # 增加批次和通道维度
    mel_tensor = np.expand_dims(log_mel_spectrogram, axis=(0, -1))
    return mel_tensor

def evaluate_mos(wav_dir, model):
    # 获取目录下的所有wav文件
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    scores = []

    for wav_file in wav_files:
        # 读取和预处理每个wav文件
        wav_data, sr = load_wav(wav_file)
        wav_tensor = preprocess_wav(wav_data, sr)
        
        # 确保输入形状符合模型的预期输入
        if wav_tensor.shape[1] > 80000:  # 截断到80000样本作为示例
            wav_tensor = wav_tensor[:, :80000, :]

        # 预测MOS分数
        score = model.predict(wav_tensor)
        scores.append(score[0][0])

    # 计算平均MOS和标准差
    mean_mos = np.mean(scores)
    std_mos = np.std(scores)
    
    return mean_mos, std_mos

if __name__ == "__main__":
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--wav_dir", type=str, required=True, help="生成的wav文件的目录"
    )
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="预训练MOSNet模型的路径"
    )
    args = parser.parse_args()

    # 加载预训练模型
    model = load_model(args.model_path)

    # 评估MOS
    mean_mos, std_mos = evaluate_mos(args.wav_dir, model)
    print(f"平均MOS: {mean_mos:.4f}, 标准差: {std_mos:.4f}")
