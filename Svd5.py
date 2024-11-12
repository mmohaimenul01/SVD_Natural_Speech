import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.linalg import svd, hankel

def read_audio(file_path):
    # Read the audio file
    sampling_rate, data = wavfile.read(file_path)
    print(f"Audio sampling rate: {sampling_rate} Hz")
    return sampling_rate, data

def preprocess_audio(data):
    # If stereo, convert to mono by averaging channels
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    # Normalize data to range -1 to 1
    data = data / np.max(np.abs(data))
    return data

def create_hankel_matrix(data, window_length=1024):
    # Create a Hankel matrix using the signal data
    hankel_matrix = hankel(data[:window_length], data[window_length-1:])
    return hankel_matrix

def apply_svd(hankel_matrix):
    # Apply SVD on the Hankel matrix
    U, S, Vt = svd(hankel_matrix, full_matrices=False)
    return U, S, Vt

def plot_singular_values(S):
    # Plot only the first 10 singular values
    S = S[:10]
    plt.figure(figsize=(10, 4))
    plt.plot(S, 'o-', label='First 10 Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('First 10 Singular Values of the Hankel Matrix')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # Step 1: Read the audio file
    file_path = "output_signal.wav"
    sampling_rate, data = read_audio(file_path)

    # Step 2: Preprocess the audio signal
    data = preprocess_audio(data)

    # Step 3: Create the Hankel matrix
    window_length = 1024
    hankel_matrix = create_hankel_matrix(data, window_length)

    # Step 4: Apply SVD
    U, S, Vt = apply_svd(hankel_matrix)

    # Step 5: Plot the singular values
    plot_singular_values(S)

    
if __name__ == "__main__":
    main()
