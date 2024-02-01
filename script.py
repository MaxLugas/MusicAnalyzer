import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
audio_path = 'music_folder/rammstein-auslander.mp3'
y, sr = librosa.load(audio_path)

# Extract and display spectrogram, MFCC, and chromagram in one plot
plt.figure(figsize=(12, 8))

# Spectrogram
plt.subplot(3, 1, 1)
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')

# MFCC
plt.subplot(3, 1, 2)
mfccs = librosa.feature.mfcc(y=y, sr=sr)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')

# Chromagram
plt.subplot(3, 1, 3)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')

plt.tight_layout()
plt.show()

# Tempo and beat analysis with limited decimal places
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(f'Tempo: {tempo:.2f} BPM')
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(f'Beat times: {beat_times.round(2)}')