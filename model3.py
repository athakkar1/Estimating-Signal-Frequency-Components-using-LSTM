import keras as keras
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.layers import Bidirectional
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
from keras.layers import Input, LSTM, Dense, Concatenate
from keras.models import Model
from itertools import permutations
import keras.backend as K
import pandas as pd
import numpy as np

def permutation_invariant_loss(y_true, y_pred):
    # Sort the true and predicted values along the last axis
    y_true_sorted = tf.sort(y_true, axis=-1)
    y_pred_sorted = tf.sort(y_pred, axis=-1)

    # Calculate the mean absolute error between the sorted true and predicted values
    return K.mean(K.abs(y_true_sorted - y_pred_sorted), axis=-1)

csv_file_path = 'sinusoid_dataset.csv'
df = pd.read_csv(csv_file_path)
features = df['Feature'].apply(lambda x: np.array([float(val.strip("[]")) for val in x.split()])).values
labels = df['Label'].apply(lambda x: np.array([float(val.strip("[]")) for val in x.split(',')])).values

print(features.shape)
print(labels.shape)

padded_array = pad_sequences(labels, padding='post', maxlen=3)
print(padded_array.shape)

labels = np.vstack(padded_array)
features = np.vstack(features)

#scaler = MinMaxScaler()
#features = scaler.fit_transform(features)
#labels = scaler.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train_categorical shape:", y_train.shape)
print("y_test_categorical shape:", y_test.shape)

# Define three LSTM branches
inputs = keras.Input(shape=(299,1))
# Branch 1
lstm_branch_1 = LSTM(units=64, return_sequences=True)(inputs)
flatten_1 = layers.Flatten()(lstm_branch_1)
output_branch_1 = Dense(units=1, activation='linear', name='output_branch_1')(flatten_1)

# Branch 2
lstm_branch_2 = LSTM(units=64, return_sequences=True)(lstm_branch_1)  # Pass the output of branch 1
flatten_2 = layers.Flatten()(lstm_branch_2)
output_branch_2 = Dense(units=1, activation='linear', name='output_branch_2')(flatten_2)

# Branch 3
lstm_branch_3 = LSTM(units=64, return_sequences=True)(lstm_branch_2)  # Pass the output of branch 2
flatten_3 = layers.Flatten()(lstm_branch_3)
output_branch_3 = Dense(units=1, activation='linear', name='output_branch_3')(flatten_3)

# Concatenate the outputs of the three branches
merged_output = Concatenate(axis=-1)([output_branch_1, output_branch_2, output_branch_3])

# Define the model
model = Model(inputs=inputs, outputs=merged_output)
print(model.summary())

model.compile(optimizer='adam',
              loss=permutation_invariant_loss, 
              metrics=['mae'])

model.fit(X_train_tensor, y_train_tensor, epochs=15, batch_size=256, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test_tensor, y_test_tensor)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
csv_file_path = 'samplesreal.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

# Extract features and labels
features = df['Feature'].apply(lambda x: np.array([float(val.strip("[]")) for val in x.split(',')])).values

# Convert the string of up to 5 numbers to a NumPy array of floats
labels = df['Label'].values
labels = np.vstack(labels)
features = np.vstack(features)
print(labels.shape)
print(features.shape)

means = np.mean(features, axis=1, keepdims=True)
print(means)
features = features - means
#scaler = MinMaxScaler()
#features = scaler.fit_transform(features)
#change to 50
features = features * 50
print("Real Sinusoid Tests:\n")
for i in range(8):
  print(labels[i:i+1])
  print(model.predict(features[i:i+1]))

print("Synthetic Sinusoid Tests:\n")
for i in range(100):
  print(y_test[i:i+1])
  print(model.predict(X_test[i:i+1]))