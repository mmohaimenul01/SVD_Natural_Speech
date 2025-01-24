import librosa
import numpy as np
from numpy.linalg import svd
import os
import soundfile as sf
import matplotlib.pyplot as plt

# Parameters
target_sr = 16000  # Target sampling rate (e.g., 16 kHz)
max_length = target_sr  # Fixed length for audio signals (1 second for 16 kHz)

# Path to the folder containing vowel WAV files
vowel_folder = 'C:/Users/mohai/Desktop/dataforthesis/vowelsfromnet'

# Get list of WAV files in the folder, sorted sequentially by name
vowel_files = sorted([
    os.path.join(vowel_folder, file)
    for file in os.listdir(vowel_folder)
    if file.endswith(".wav")
], key=lambda x: int(os.path.basename(x).split('.')[0]))

# Preprocess and load all audio files
time_data = []

for file in vowel_files:
    # Load the WAV file
    audio, sr = librosa.load(file, sr=None)  # Load with original sampling rate

    # Resample to target sampling rate
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Normalize audio to range [-1, 1]
    audio_normalized = audio_resampled / np.max(np.abs(audio_resampled))

    # Trim silence
    audio_trimmed, _ = librosa.effects.trim(audio_normalized, top_db=30)

    # Pad or truncate to fixed length
    if len(audio_trimmed) < max_length:
        audio_padded = np.pad(audio_trimmed, (0, max_length - len(audio_trimmed)), mode="constant")
    else:
        audio_padded = audio_trimmed[:max_length]

    # Append preprocessed audio to the list
    time_data.append(audio_padded)

# Create matrix A (each column is the time-domain data of one vowel)
A = np.stack(time_data, axis=1)  # Shape: (m_samples x n_vowels)

# Apply Singular Value Decomposition (SVD)
U, S, Vt = svd(A, full_matrices=False)

# Convert Sigma vector to a diagonal matrix
Sigma = np.diag(S)

# Print shapes of the matrices
print(f"Matrix A shape: {A.shape}")
print(f"U shape: {U.shape}")  # Left singular vectors (time-domain basis)
print(f"Sigma shape: {Sigma.shape}")  # Singular values as a diagonal matrix
print(f"V^T shape: {Vt.shape}")  # Right singular vectors (relationships among vowels)

# Display the matrices
print("\nMatrix U:")
print(U)
print("\nSigma (as a diagonal matrix):")
print(Sigma)
print("\nMatrix V^T:")
print(Vt)

# Plot singular values
plt.plot(S)
plt.title("Singular Values")
plt.xlabel("Component Index")
plt.ylabel("Magnitude")
plt.show()

# Cumulative energy plot
energy_contribution = (S ** 2) / np.sum(S ** 2)
cumulative_energy = np.cumsum(energy_contribution)
plt.plot(cumulative_energy)
plt.title("Cumulative Energy Contribution")
plt.xlabel("Number of Singular Values")
plt.ylabel("Cumulative Energy")
plt.show()

import matplotlib.pyplot as plt

# Plot the first few temporal basis vectors
num_vectors_to_plot = 5  # Number of basis vectors to visualize
for i in range(num_vectors_to_plot):
    plt.plot(U[:, i], label=f'U[:, {i}]')
    plt.title(f'Temporal Basis Vector {i+1}')
    plt.xlabel('Time Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

P = U.T @ A  # Projection of original data onto temporal basis

# Plot the contribution of the first few temporal basis vectors
num_vowels = A.shape[1]
for i in range(num_vectors_to_plot):
    plt.bar(range(num_vowels), P[i, :], label=f'U[:, {i}] contribution')
    plt.title(f'Contribution of Temporal Basis Vector {i+1}')
    plt.xlabel('Vowel Index')
    plt.ylabel('Contribution')
    plt.legend()
    plt.show()

k = 3  # Retain top 3 singular values
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Compare original and reconstructed signals for a vowel
vowel_idx = 0  # Index of vowel to compare
reconstructed_signal = A_k[:, vowel_idx]
# Normalize the reconstructed signal to range [-1, 1] for playback
reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))
plt.plot(A[:, vowel_idx], label='Original')
plt.plot(A_k[:, vowel_idx], label=f'Reconstructed with {k} components')
plt.title('Reconstruction of Vowel Time-Domain Signal')
plt.xlabel('Time Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
# Save the reconstructed signal as a WAV file
sf.write(f"reconstructed_vowel_{vowel_idx}.wav", reconstructed_signal, samplerate=target_sr)
