import numpy as np
import pandas as pd
import random

# Define parameters
num_samples = 1000
num_frequencies = random.randint(2, 5)
min_frequency = 1
max_frequency = 5000
num_points = 20000
sampling_frequency = 20000
min_amplitude = 20
max_amplitude = 100
noise_std = 5

# Create empty dataset
dataset = []

# Generate dataset
for _ in range(num_samples):
    frequencies = []
    amplitudes = []
    sinusoid = np.zeros(num_points)
    
    for _ in range(num_frequencies):
        frequency = random.uniform(min_frequency, max_frequency)
        amplitude = random.uniform(min_amplitude, max_amplitude)
        frequencies.append(frequency)
        amplitudes.append(amplitude)
        time = np.linspace(0, (num_points-1)/sampling_frequency, num_points)
        sinusoid += amplitude * np.sin(2 * np.pi * frequency * time)
    
    sinusoid += np.random.normal(0, noise_std, num_points)
    dataset.append([sinusoid] + frequencies)

# Convert dataset to DataFrame
df = pd.DataFrame(dataset, columns=['Signal'] + [f'Frequency_{i+1}' for i in range(num_frequencies)])

# Export DataFrame to CSV
df.to_csv('dataset.csv', index=False)
