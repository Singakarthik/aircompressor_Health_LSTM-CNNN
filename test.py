#10
import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from sklearn.metrics import confusion_matrix


# Define a function to calculate specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# Function to load audio files and preprocess them
def load_and_preprocess_data(data_path, sampling_rate=16000, duration=2):
    data = []
    labels = []
    
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):  # Check if it's a directory
            continue
        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path, filename)
            # Check if it's a WAV file
            if not file_path.endswith('.wav'):
                continue
            # Load audio file and extract features
            y, sr = librosa.load(file_path, sr=sampling_rate, duration=duration, mono=True)
            # Perform feature extraction (e.g., MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            # Pad or truncate sequences to ensure uniform length
            if mfccs.shape[1] < 40:  # Adjust the sequence length as needed
                pad_width = 40 - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :40]
            data.append(mfccs.T)  # Transpose to match LSTM input shape
            labels.append(label)
    
    return np.array(data), np.array(labels)

# Define paths to the dataset
data_path = r"\Air_compressor\AirDataset\AirCompressorDataset"

# Load and preprocess data
X, y = load_and_preprocess_data(data_path)

# Convert labels to categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(LSTM(units=180, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

# Compile the model with precision and recall as additional metrics
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', Precision(), Recall(), specificity])

# Display model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# Evaluate the model
loss, accuracy, precision, recall, specificity = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test Specificity: {specificity:.4f}')
print(f'Validation Loss: {val_loss[-1]:.4f}')  # Print the last validation loss
print(f'Validation Accuracy: {val_accuracy[-1]:.4f}')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = label_encoder.classes_
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

plt.show()


# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['loss'], label='Training Loss')  # Add training loss to the plot
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.legend()
plt.show()

# Function to preprocess new audio file
def preprocess_new_audio(file_path, sampling_rate=16000, duration=2):
    # Load audio file and extract features
    y, sr = librosa.load(file_path, sr=sampling_rate, duration=duration, mono=True)
    # Perform feature extraction (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Pad or truncate sequences to ensure uniform length
    if mfccs.shape[1] < 40:  # Adjust the sequence length as needed
        pad_width = 40 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :40]
    return np.expand_dims(mfccs.T, axis=0)  # Transpose to match LSTM input shape and add batch dimension

# Path to new WAV file
new_file_path = r"D:\Air_compressor\AirDataset\AirCompressorTest\Healthy\preprocess_Reading200.wav"

# Preprocess new audio file
new_data = preprocess_new_audio(new_file_path)

# Make prediction
prediction = model.predict(new_data)
print("Shape of prediction:", prediction.shape)  # Print shape for debugging
if prediction.shape[0] == 0:  # Check if prediction is empty
    print("Prediction is empty. Check your model or input data.")
else:
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
    print(f'The Status of Air Compressor Predicted using LSTM:  {predicted_label}')
    print()
    print()
    print()
