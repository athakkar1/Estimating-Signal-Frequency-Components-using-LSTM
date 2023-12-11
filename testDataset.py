import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def generate_sin_wave(freqs, duration, sampling_rate=299, amplitude=1, noise_level=0.1, phases=None):
    t = np.arange(0, duration, 1/sampling_rate)
    if phases is None:
        phases = [np.random.uniform(0, 2 * np.pi) for _ in freqs]
    sin_waves = [amplitude * np.sin(2 * np.pi * freq * t + phase) for freq, phase in zip(freqs, phases)]
    combined_wave = np.sum(sin_waves, axis=0)
    noise = np.random.normal(0, noise_level, len(t))*amplitude
    noisy_combined_wave = combined_wave + noise
    return t, noisy_combined_wave

def create_dataset(num_samples, noise_level=0.1):
    dataset = []

    for _ in range(num_samples):
        if random.choice([True, False]):
            freqs = [int(round(np.random.uniform(1, 100)))]
            amplitude = np.random.uniform(20, 100)
            phases = None
        else:
            num_frequencies = random.randint(2, 3)
            freqs = random.sample(range(1, 101), num_frequencies)
            amplitude = np.random.uniform(20, 100)
            phases = None
        #freqs = [int(round(np.random.uniform(1, 100)))]
        #amplitude = np.random.uniform(20, 100)
        #phases = None  # No specified phases for single-frequency
        duration = 1
        t, sin_wave = generate_sin_wave(freqs, duration, amplitude=amplitude, noise_level=noise_level, phases=phases)
        feature = ' '.join(map(str, sin_wave))
        label = freqs

        dataset.append((feature, label))

    return dataset
num_samples = 200000
noise_level = 0.1
dataset = create_dataset(num_samples, noise_level=noise_level)
for i in range(10):
    t, sin_wave = generate_sin_wave(dataset[i][1], 1, amplitude=1, noise_level=noise_level, phases=None)
    plt.figure(figsize=(10, 3))
    for freq, phase in zip(dataset[i][1], [0]*len(dataset[i][1])):  # Set phase to 0 for clarity
        individual_sin_wave = generate_sin_wave([freq], 1, amplitude=1, noise_level=0, phases=[phase])[1]
        plt.plot(t, individual_sin_wave, label=f'Frequency: {freq} Hz', linestyle='-', linewidth=1, alpha=0.5)
    plt.plot(t, sin_wave, label=f'Combined', linestyle='-', linewidth=2, color='black')
    plt.title(f'Sample {i + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

df = pd.DataFrame(dataset, columns=['Feature', 'Label'])

csv_filename = 'sinusoid_dataset.csv'
df.to_csv(csv_filename, index=False)

print(f"Dataset saved to {csv_filename}")