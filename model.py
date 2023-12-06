
import keras as keras
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.layers import Bidirectional
from keras.utils import pad_sequences
import pandas as pd
import numpy as np

# Load CSV into a Pandas DataFrame
csv_file_path = 'sinusoid_dataset.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

# Extract features and labels
features = df['Feature'].apply(lambda x: np.array([float(val.strip("[]")) for val in x.split()])).values

# Convert the string of up to 5 numbers to a NumPy array of floats
labels = df['Label'].apply(lambda x: np.array([float(val.strip("[]")) for val in x.split(',')])).values

print(features.shape)
print(labels.shape)

# Preprocess the data
# Reshape the features array to have one column for each data point
# Find the maximum length of the sublists
# Create a 2D array and pad with zeros
padded_array = pad_sequences(labels, padding='post', maxlen=3)
print(padded_array.shape)
# Convert to a true 2D NumPy array
labels = np.vstack(padded_array)
features = np.vstack(features)

#scaler = MinMaxScaler()
#features = scaler.fit_transform(features)
#labels = scaler.fit_transform(labels)

# Standardize the features (optional but can be beneficial for some models)
#scaler = StandardScaler()
#features = scaler.fit_transform(features)

# Split the data into training and testing sets
# Adjust the test_size and random_state according to your requirements
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Now X_train, X_test, y_train_categorical, and y_test_categorical are ready for input to a Keras model

# Print the shapes of the processed data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train_categorical shape:", y_train.shape)
print("y_test_categorical shape:", y_test.shape)

model = keras.Sequential()
model.add(layers.LSTM(299, input_shape=(299, 1), return_sequences=True))
model.add(layers.BatchNormalization())  # Batch normalization can be helpful
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.BatchNormalization())
model.add(layers.LSTM(64, return_sequences=False))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
# Dense layer with ReLU activation for each time step
model.add(layers.Dense(32, activation='tanh'))

# Fully connected output layer with up to 5 nodes (adjust as needed)
model.add(layers.Dense(3))
print(model.summary())

model.compile(optimizer='adam',  # You can choose other optimizers like 'sgd', 'rmsprop', etc.
              loss='mean_absolute_error',  # Adjust the loss based on your problem (categorical_crossentropy for classification)
              metrics=['mae'])  # You can add more metrics as needed

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
#change to 50
features = features * 1
print("Real Sinusoid Tests:\n")
for i in range(8):
  print(labels[i:i+1])
  print(model.predict(features[i:i+1]))
print("Synthetic Sinusoid Tests:\n")
for i in range(100):
  print(y_test[i:i+1])
  print(model.predict(X_test[i:i+1]))
