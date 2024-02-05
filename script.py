import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Loading the audio file
audio_path = 'music_folder/rammstein-auslander.mp3'
y, sr = librosa.load(audio_path)

# Extracting onset strength
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# Tempo and beat analysis
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# Converting frame numbers to time instances
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# Extracting and displaying spectrogram, MFCC, chromagram, and onset strength in one window
plt.figure(figsize=(12, 10))

# MFCC
plt.subplot(4, 1, 1)
mfccs = librosa.feature.mfcc(y=y, sr=sr)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')

# Chromagram
plt.subplot(4, 1, 2)
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')

# Mel Spectrogram
plt.subplot(4, 1, 3)
hop_length = 512
times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                         y_axis='mel', x_axis='time', hop_length=hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')

# Onset strength and beats
plt.subplot(4, 1, 4)
plt.plot(times, librosa.util.normalize(onset_env), label='Onset Strength', alpha=0.8)
plt.vlines(beat_times, 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Onset Strength and Beats')
plt.legend()

plt.tight_layout()
plt.show()
