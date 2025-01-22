import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

# Define the folder containing vowel files
vowel_folder = 'C:/Users/mohai/Desktop/dataforthesis/vowelsfromnet'  # Replace with the actual path

# Get the list of files in the folder
vowel_files = sorted([f for f in os.listdir(vowel_folder) if f.endswith(".wav")])

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file

    # Extract features
    try:
        f0, _, _ = librosa.pyin(y, sr=sr, fmin=75, fmax=300)  # Pitch (F0)
    except:
        f0 = [np.nan]  # Handle pitch extraction errors
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)    # MFCCs
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # Spectral Centroid
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # Spectral Bandwidth
    rms_energy = librosa.feature.rms(y=y)  # Short-Time Energy
    spectral_flatness = librosa.feature.spectral_flatness(y=y)  # Spectral Flatness
    zcr = librosa.feature.zero_crossing_rate(y)  # Zero-Crossing Rate

    # Formants approximation using peaks in the spectrum
    fft_spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    magnitude = np.abs(fft_spectrum)
    peaks = np.argsort(magnitude)[-3:]  # Approximate formants as the top 3 peaks
    formants = freqs[peaks] if len(peaks) >= 3 else [np.nan] * 3

    # Placeholder for Harmonic-to-Noise Ratio (HNR)
    hnr = np.nan  # HNR calculation can be added if needed

    # Compute means for each feature
    feature_vector = [
        np.nanmean(f0),                  # Mean pitch
        *np.mean(mfccs, axis=1),        # Mean MFCCs (13 values)
        np.mean(spectral_centroid),     # Spectral Centroid
        np.mean(spectral_bandwidth),    # Spectral Bandwidth
        np.mean(rms_energy),            # RMS Energy
        np.mean(spectral_flatness),     # Spectral Flatness
        np.mean(zcr),                   # Zero-Crossing Rate
        *formants,                      # Formants (F1, F2, F3)
        hnr                             # Harmonic-to-Noise Ratio (placeholder)
    ]

    return feature_vector

# Initialize feature matrix
feature_matrix = []

# Extract features for all files in the folder
for file_name in vowel_files:
    file_path = os.path.join(vowel_folder, file_name)

    if os.path.exists(file_path):
        features = extract_features(file_path)
        feature_matrix.append(features)
    else:
        print(f"File not found: {file_path}")
        feature_matrix.append([np.nan] * 19)  # Adjust placeholder for missing files

# Create feature matrix DataFrame
columns = [
    "F0", 
    *[f"MFCC_{i}" for i in range(1, 14)], 
    "Spectral_Centroid", 
    "Spectral_Bandwidth", 
    "RMS_Energy", 
    "Spectral_Flatness", 
    "Zero_Crossing_Rate", 
    "F1", "F2", "F3", 
    "HNR"
]
feature_df = pd.DataFrame(feature_matrix, columns=columns, index=vowel_files)

# Save the feature matrix to a CSV file
output_file = "vowel_feature_matrix.csv"
feature_df.to_csv(output_file)
print(f"Feature matrix saved to {output_file}")

# Apply Eigen Component Decomposition
from numpy.linalg import eig

# Select numerical features (all columns are now numerical)
numerical_features = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

# Compute covariance matrix
cov_matrix = np.cov(numerical_features.T)

# Check for validity of covariance matrix
if not np.all(np.isfinite(cov_matrix)):
    raise ValueError("Covariance matrix contains invalid values.")

# Perform eigen decomposition
eigenvalues, eigenvectors = eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project the data onto the eigenvectors
projected_data = np.dot(numerical_features, eigenvectors)

# Save eigen decomposition results
np.savetxt("eigenvalues.csv", eigenvalues, delimiter=",", header="Eigenvalues")
np.savetxt("eigenvectors.csv", eigenvectors, delimiter=",", header="Eigenvectors")
pd.DataFrame(projected_data, index=numerical_features.index).to_csv("projected_data.csv")

print("Eigen Component Decomposition applied. Results saved to files.")

# Visualization
# 1. Scatter plot for first two eigenvectors
plt.figure(figsize=(10, 7))
plt.scatter(projected_data[:, 0], projected_data[:, 1], edgecolors='k', c='blue')
for i, label in enumerate(numerical_features.index):
    plt.annotate(label, (projected_data[i, 0], projected_data[i, 1]), fontsize=8)
plt.title('Projection onto First Two Eigenvectors')
plt.xlabel('Eigenvector 1')
plt.ylabel('Eigenvector 2')
plt.grid()
plt.show()

# 2. Heatmap of Eigenvectors
plt.figure(figsize=(12, 8))
sns.heatmap(eigenvectors[:, :10], annot=True, cmap='coolwarm', xticklabels=[f"Eigenvector {i+1}" for i in range(10)],
            yticklabels=numerical_features.columns)
plt.title('Heatmap of Eigenvectors')
plt.xlabel('Eigenvectors')
plt.ylabel('Features')
plt.show()

# 3. Bar plot of Eigenvalues
explained_variance = eigenvalues / np.sum(eigenvalues) * 100
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.title('Explained Variance by Eigenvalues')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Percentage of Explained Variance')
plt.show()
