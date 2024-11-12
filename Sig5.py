import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.io.wavfile import write

def generate_signal(frequencies=[1], amplitudes=[1], phases=[0], formant_bandwidths=[50], duration=1, sampling_rate=1000, window_type=None):
    # Time array from 0 to duration with step size based on the sampling rate
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    y = np.zeros_like(t)

    # Generate sine wave(s) based on the number of frequencies
    individual_waves = []
    for freq, amp, phase, bw in zip(frequencies, amplitudes, phases, formant_bandwidths):
        # Apply bandwidth effect as an exponential decay
        envelope = np.exp(-bw * t)
        wave = amp * np.sin(2 * np.pi * freq * t + phase) * envelope
        individual_waves.append(wave)
        y += wave
    
    # Apply window function if specified
    if window_type:
        window = get_window(window_type, len(t))
        y *= window
        individual_waves = [wave * window for wave in individual_waves]
    
    return t, y, individual_waves

def plot_signal(t, y, title="Signal"):
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)

if __name__ == "__main__":
    # User-defined parameters for signal
    duration = float(input("Enter the duration (seconds): "))
    sampling_rate = int(input("Enter the sampling rate (samples per second): "))
    num_waves = int(input("Enter the number of sine waves: "))
    frequencies = [float(input(f"Enter frequency {i+1} (Hz): ")) for i in range(num_waves)]
    amplitudes = [float(input(f"Enter amplitude {i+1}: ")) for i in range(num_waves)]
    phases = [float(input(f"Enter phase {i+1} (radians): ")) for i in range(num_waves)]
    formant_bandwidths = [float(input(f"Enter formant bandwidth {i+1} (Hz): ")) for i in range(num_waves)]
    window_type = input("Enter window type (e.g., hann, hamming, blackman) or leave blank for none: ").strip().lower()
    window_type = window_type if window_type else None

    # Generate and plot signal
    t, y, individual_waves = generate_signal(frequencies=frequencies, amplitudes=amplitudes, phases=phases, formant_bandwidths=formant_bandwidths, duration=duration, sampling_rate=sampling_rate, window_type=window_type)
    
    if len(frequencies) == 1:
        # Plot single sine wave
        plt.figure(figsize=(10, 4))
        plot_signal(t, y, title="Single Sine Wave")
        plt.show()
    else:
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot sum of all sine waves
        plt.sca(axes[0])
        plot_signal(t, y, title="Sum of Multiple Sine Waves")
        
        # Plot individual sine waves
        plt.sca(axes[1])
        for i, wave in enumerate(individual_waves):
            plt.plot(t, wave, label=f"Frequency {frequencies[i]} Hz, Bandwidth {formant_bandwidths[i]} Hz")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Individual Sine Waves')
        plt.legend()
        plt.grid(True)
        
        # Show both plots
        plt.tight_layout()
        plt.show()

    # Save the generated signal to a WAV file
    bit_depth = int(input("Enter bit depth for WAV file (e.g., 16, 24, 32): "))
    if bit_depth == 16:
        y_normalized = np.int16(y / np.max(np.abs(y)) * 32767)  # Normalize to 16-bit PCM
    elif bit_depth == 24:
        y_normalized = np.int32(y / np.max(np.abs(y)) * 8388607)  # Normalize to 24-bit PCM
    elif bit_depth == 32:
        y_normalized = np.int32(y / np.max(np.abs(y)) * 2147483647)  # Normalize to 32-bit PCM
    else:
        raise ValueError("Unsupported bit depth. Please enter 16, 24, or 32.")
    
    write("output_signal.wav", sampling_rate, y_normalized)
    print(f"Signal saved to 'output_signal.wav' with {bit_depth}-bit depth and {sampling_rate} Hz sampling rate")
