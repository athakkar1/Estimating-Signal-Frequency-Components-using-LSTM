import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def generate_sin_wave(freqs, duration, sampling_rate=60, amplitude=1, noise_level=0.1, phases=None):
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Generate random phases if not provided
    if phases is None:
        phases = [np.random.uniform(0, 2 * np.pi) for _ in freqs]
    
    # Generate each sine wave with the specified phase
    sin_waves = [amplitude * np.sin(2 * np.pi * freq * t + phase) for freq, phase in zip(freqs, phases)]
    
    # Sum all sine waves
    combined_wave = np.sum(sin_waves, axis=0)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, len(t))
    noisy_combined_wave = combined_wave + noise
    
    return t, noisy_combined_wave

def create_dataset(num_samples, noise_level=0.1):
    dataset = []

    for _ in range(num_samples):
        # Determine if it's a single-frequency or mixed-frequency sinusoid
        if random.choice([True, False]):
            # Single-frequency sinusoid
            freqs = [int(round(np.random.uniform(1, 10)))]
            amplitude = np.random.uniform(20, 100)
            phases = None  # No specified phases for single-frequency
        else:
            # Mixed-frequency sinusoid (random between 2 and 5 frequencies)
            num_frequencies = random.randint(2, 5)
            freqs = [int(round(f)) for f in np.random.uniform(1, 10, size=num_frequencies)]
            amplitude = np.random.uniform(20, 100)
            phases = None  # No specified phases for mixed-frequency
            
        # Generate sinusoid with 20000 points and add noise
        duration = 1  # 20000 points with a sampling rate of 20000 Hz
        t, sin_wave = generate_sin_wave(freqs, duration, amplitude=amplitude, noise_level=noise_level, phases=phases)

        # Store the string of numbers representing the noisy sinusoid
        feature = ' '.join(map(str, sin_wave))

        # Store the label as a vector of frequencies
        label = freqs

        dataset.append((feature, label))

    return dataset

# Example: Create a dataset with 100 samples and introduce noise
num_samples = 20000
noise_level = 0.1
dataset = create_dataset(num_samples, noise_level=noise_level)

# Plot 10 samples from the dataset with line graph
for i in range(10):
    t, sin_wave = generate_sin_wave(dataset[i][1], 1, amplitude=1, noise_level=noise_level, phases=None)
    
    plt.figure(figsize=(10, 3))
    
    # Plot each individual sine wave
    for freq, phase in zip(dataset[i][1], [0]*len(dataset[i][1])):  # Set phase to 0 for clarity
        individual_sin_wave = generate_sin_wave([freq], 1, amplitude=1, noise_level=0, phases=[phase])[1]
        plt.plot(t, individual_sin_wave, label=f'Frequency: {freq} Hz', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.plot(t, sin_wave, label=f'Combined', linestyle='-', linewidth=2, color='black')
    plt.title(f'Sample {i + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Convert the dataset to a DataFrame
df = pd.DataFrame(dataset, columns=['Feature', 'Label'])

# Save the DataFrame to a CSV file
csv_filename = 'sinusoid_dataset.csv'
df.to_csv(csv_filename, index=False)

print(f"Dataset saved to {csv_filename}")