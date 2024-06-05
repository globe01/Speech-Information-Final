import torchaudio
import torchaudio.backend.sox_io_backend

torchaudio.set_audio_backend("sox_io")
print(torchaudio.get_audio_backend())
