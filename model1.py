import keras as keras
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.utils import pad_sequences
from keras import Model, Input
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
padded_array = pad_sequences(labels, padding='post', maxlen=5)
print(padded_array.shape)
# Convert to a true 2D NumPy array
labels = np.vstack(padded_array)
features = np.vstack(features)

scaler = MinMaxScaler()
features = scaler.fit_transform(features)
labels = scaler.fit_transform(labels)

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

input_layer = Input(shape=(60, 1))
query = layers.LSTM(64, return_sequences=True)(input_layer)
query = layers.BatchNormalization()(query)
key = layers.LSTM(24, return_sequences=True)(query)
key = layers.BatchNormalization()(key)
attention = layers.MultiHeadAttention(num_heads=2, key_dim=2)([query, key])
x = layers.LSTM(8, return_sequences=False)(attention)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
output_layer = layers.Dense(5)(x)

model = Model(input_layer, output_layer)

model.compile(optimizer='adam',  # You can choose other optimizers like 'sgd', 'rmsprop', etc.
              loss='mean_squared_error',  # Adjust the loss based on your problem (categorical_crossentropy for classification)
              metrics=['mae', 'mse', 'mape'])  # You can add more metrics as needed

model.fit(X_train_tensor, y_train_tensor, epochs=5, batch_size=32, validation_split=0.2)

test_metrics = model.evaluate(X_test_tensor, y_test_tensor)
test_loss, test_mae, test_mse, test_mape = test_metrics
print(f'Test Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')
print(f'Test MSE: {test_mse:.4f}')
print(f'Test MAPE: {test_mape:.4f}')