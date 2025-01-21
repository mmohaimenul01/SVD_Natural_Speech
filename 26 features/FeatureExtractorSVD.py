import os
import numpy as np
import pandas as pd
import librosa
from librosa.feature import mfcc
import matplotlib.pyplot as plt

# Function to extract features from a single audio file
def extract_features(file_path, n_mfcc=13):
    features = {}

    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Extract fundamental frequency (F0)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    f0 = np.mean([np.max(p[p > 0]) for p in pitches.T if np.any(p > 0)])
    features['F0'] = f0

    # Extract formants (F1, F2, F3) approximation via spectral peaks
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['Spectral Centroid'] = np.mean(spectral_centroid)

    # Formant-like spectral peaks (approximation using peaks in spectrum)
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    peaks = np.argsort(spectrum)[-3:][::-1]  # Top 3 peaks
    features['F1'] = freqs[peaks[0]]
    features['F2'] = freqs[peaks[1]]
    features['F3'] = freqs[peaks[2]]

    # Extract MFCCs
    mfccs = mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f'MFCC_{i+1}'] = np.mean(mfccs[i])

    # Extract energy features
    rms = librosa.feature.rms(y=y)[0]
    features['RMS Energy'] = np.mean(rms)
    total_energy = np.sum(y ** 2)
    features['Total Energy'] = total_energy

    # Extract duration
    duration = librosa.get_duration(y=y, sr=sr)
    features['Duration'] = duration

    # Harmonic-to-Noise Ratio (HNR) approximation
    harmonic, percussive = librosa.effects.hpss(y)
    hnr = np.mean(harmonic ** 2) / (np.mean(percussive ** 2) + 1e-10)
    features['HNR'] = 10 * np.log10(hnr)

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['Zero-Crossing Rate'] = np.mean(zcr)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['Spectral Bandwidth'] = np.mean(spectral_bandwidth)

    # Spectral Roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['Spectral Roll-off'] = np.mean(spectral_rolloff)

    return features

# Function to process all vowel files and create the feature matrix
def process_vowel_files(directory, n_mfcc=13):
    feature_matrix = []
    vowel_names = []

    # Custom sort based on the numerical prefix of the filenames
    def sort_key(file_name):
        try:
            # Extract the numeric prefix
            return int(file_name.split('.')[0])
        except ValueError:
            # Fallback for files without numeric prefixes
            return float('inf')

    # Iterate over all files in the directory sorted numerically
    for file_name in sorted(os.listdir(directory), key=sort_key):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            
            # Extract features for the file
            features = extract_features(file_path, n_mfcc)
            
            # Append to matrix
            feature_matrix.append(features)
            vowel_names.append(file_name)

    # Create a DataFrame
    feature_df = pd.DataFrame(feature_matrix)
    feature_df.index = vowel_names  # Set vowel file names as index

    return feature_df

# Directory containing vowel sound files (replace with your directory path)
vowel_directory = 'C:/Users/mohai/Desktop/dataforthesis/vowelsfromnet'

# Process files and create the feature matrix
feature_matrix = process_vowel_files(vowel_directory)

# Save the matrix to a CSV file
feature_matrix.to_csv('vowel_feature_matrix.csv')

# Display the matrix
print("Feature Matrix:")
print(feature_matrix)

# Load the feature matrix for full SVD
feature_matrix = pd.read_csv('vowel_feature_matrix.csv', index_col=0)

# Apply full SVD
U, Sigma, VT = np.linalg.svd(feature_matrix, full_matrices=True)

# Save the SVD components
np.savetxt('U_matrix.csv', U, delimiter=',')
np.savetxt('Sigma_vector.csv', Sigma, delimiter=',')
np.savetxt('VT_matrix.csv', VT, delimiter=',')

# Display results
print("U Matrix:")
print(U)
print("Sigma Vector:")
print(Sigma)
print("VT Matrix:")
print(VT)

# Visualize singular values
plt.figure(figsize=(8, 5))
plt.plot(Sigma, marker='o')
plt.title('Singular Values')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.grid()
plt.show()
