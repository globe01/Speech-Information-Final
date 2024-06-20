import os
import glob
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import librosa
from tensorflow.keras.models import load_model

def load_wav(file_path):
    sr, wav_data = wavfile.read(file_path)
    return wav_data, sr

def preprocess_wav(wav_data, sr, n_fft=1024, hop_length=256, n_mels=257):
    # Compute Mel-spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav_data.astype(float), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # Normalize
    log_mel_spectrogram = (log_mel_spectrogram - np.mean(log_mel_spectrogram)) / np.std(log_mel_spectrogram)
    # Transpose to (time, mel)
    log_mel_spectrogram = log_mel_spectrogram.T
    # Add batch and channel dimensions
    mel_tensor = np.expand_dims(log_mel_spectrogram, axis=(0, -1))
    return mel_tensor

def evaluate_mos(wav_dir, model):
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    scores = []

    for wav_file in wav_files:
        wav_data, sr = load_wav(wav_file)
        wav_tensor = preprocess_wav(wav_data, sr)
        
        # Ensure the input shape matches the model's expected input
        if wav_tensor.shape[1] > 80000:  # Truncate to 80000 samples for example
            wav_tensor = wav_tensor[:, :80000, :]

        score = model.predict(wav_tensor)
        scores.append(score[0][0])

    mean_mos = np.mean(scores)
    std_mos = np.std(scores)
    
    return mean_mos, std_mos

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--wav_dir", type=str, required=True, help="directory of the generated wav files"
    )
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="path to the pretrained MOSNet model"
    )
    args = parser.parse_args()

    # Load the pretrained model
    model = load_model(args.model_path)

    # Evaluate MOS
    mean_mos, std_mos = evaluate_mos(args.wav_dir, model)
    print(f"Mean MOS: {mean_mos:.4f}, Standard Deviation: {std_mos:.4f}")
