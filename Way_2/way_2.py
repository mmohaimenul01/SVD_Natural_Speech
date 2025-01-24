import librosa
import numpy as np
from numpy.linalg import svd
import soundfile as sf
import sounddevice as sd
import os
import matplotlib.pyplot as plt

# Parameters
target_sr = 16000  # Target sampling rate (e.g., 16 kHz)
n_fft = 1024       # FFT window size
hop_length = 256   # Hop length for STFT
max_frames = 128   # Number of time frames to standardize spectrograms

# Path to the folder containing vowel WAV files
vowel_folder = 'C:/Users/mohai/Desktop/dataforthesis/vowelsfromnet'

# Get list of WAV files in the folder, sorted sequentially by name
vowel_files = sorted([
    os.path.join(vowel_folder, file)
    for file in os.listdir(vowel_folder)
    if file.endswith(".wav")
], key=lambda x: int(os.path.basename(x).split('.')[0]))

# Preprocess and generate spectrograms
spectrograms = []

for file in vowel_files:
    # Load the WAV file
    audio, sr = librosa.load(file, sr=None)

    # Resample to target sampling rate
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Normalize audio to range [-1, 1]
    audio_normalized = audio_resampled / np.max(np.abs(audio_resampled))

    # Compute the spectrogram (magnitude)
    spectrogram = np.abs(librosa.stft(audio_normalized, n_fft=n_fft, hop_length=hop_length))

    # Standardize the number of time frames (truncate or pad)
    if spectrogram.shape[1] < max_frames:
        spectrogram_padded = np.pad(spectrogram, ((0, 0), (0, max_frames - spectrogram.shape[1])), mode="constant")
    else:
        spectrogram_padded = spectrogram[:, :max_frames]

    # Flatten the spectrogram and add to the list
    spectrograms.append(spectrogram_padded.flatten())

# Create matrix A (each column is the flattened spectrogram of one vowel)
A = np.stack(spectrograms, axis=1)  # Shape: (m_features x n_vowels)

# Apply Singular Value Decomposition (SVD)
U, S, Vt = svd(A, full_matrices=False)

# Convert Sigma vector to a diagonal matrix
Sigma = np.diag(S)

# Print shapes of the matrices
print(f"Matrix A shape: {A.shape}")
print(f"U shape: {U.shape}")      # Left singular vectors (frequency domain basis)
print(f"Sigma shape: {Sigma.shape}")  # Singular values as a diagonal matrix
print(f"V^T shape: {Vt.shape}")  # Right singular vectors (relationships among vowels)

# Plot the singular values
plt.plot(S)
plt.title("Singular Values")
plt.xlabel("Index")
plt.ylabel("Magnitude")
plt.show()

# Plot the temporal basis vectors (columns of U)
num_vectors_to_plot = 5
for i in range(num_vectors_to_plot):
    plt.plot(U[:, i], label=f'U[:, {i}]')
    plt.title(f'Temporal Basis Vector {i+1}')
    plt.xlabel('Time Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Reconstruct signal using top k components
k = 5  # Number of singular values to retain
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Reconstruct the spectrogram of a specific vowel
vowel_idx = 0  # Index of the vowel to reconstruct
reconstructed_spectrogram = A_k[:, vowel_idx].reshape((n_fft // 2 + 1, max_frames))

# Convert the spectrogram back to a time-domain signal
reconstructed_audio = librosa.istft(reconstructed_spectrogram, hop_length=hop_length)

# Normalize the reconstructed audio for playback
reconstructed_audio = reconstructed_audio / np.max(np.abs(reconstructed_audio))

# Plot the original and reconstructed spectrograms
original_spectrogram = A[:, vowel_idx].reshape((n_fft // 2 + 1, max_frames))
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Spectrogram")
librosa.display.specshow(librosa.amplitude_to_db(original_spectrogram, ref=np.max), sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format="%+2.0f dB")

plt.subplot(1, 2, 2)
plt.title("Reconstructed Spectrogram")
librosa.display.specshow(librosa.amplitude_to_db(reconstructed_spectrogram, ref=np.max), sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()

# Play the reconstructed audio
sd.play(reconstructed_audio, samplerate=target_sr)
sd.wait()

# Save the reconstructed audio as a WAV file
sf.write(f"reconstructed_vowel_{vowel_idx}.wav", reconstructed_audio, samplerate=target_sr)
