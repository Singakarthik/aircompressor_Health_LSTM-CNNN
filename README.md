
# Air Compressor Health Monitoring Using LSTM

## Overview

This project implements a machine learning-based solution to monitor the health of air compressors using audio signal analysis. The system leverages Long Short-Term Memory (LSTM) neural networks to classify the status of air compressors as "Healthy" or "Faulty" based on processed audio features.

## Features

- **Audio Signal Processing**: Extracts MFCC features from audio recordings of air compressors.
- **LSTM Model**: Uses a neural network to classify compressor health status.
- **Real-Time Prediction**: Supports new audio file preprocessing and status prediction.
- **Visualization Tools**: Includes confusion matrix and training history plots for model evaluation.

## Prerequisites

- Python 3.7 or later
- Required Python libraries:
  - `numpy`
  - `librosa`
  - `scikit-learn`
  - `keras` / `tensorflow`
  - `matplotlib`

Install the libraries using:
```bash
pip install numpy librosa scikit-learn tensorflow keras matplotlib
```

## Dataset Structure

Organize your dataset into labeled folders within the specified directory. For example:
```
AirCompressorDataset/
├── Healthy/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── Faulty/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
```

## How It Works

1. **Data Preprocessing**:
   - Extracts 20 Mel-Frequency Cepstral Coefficients (MFCC) from audio files.
   - Pads or truncates features to ensure uniform sequence length.
2. **LSTM Model**:
   - A neural network trained on processed MFCC features to classify health status.
3. **Evaluation**:
   - Model performance assessed using accuracy, precision, recall, and specificity.
   - Outputs a confusion matrix and training-validation accuracy/loss plots.
4. **Real-Time Prediction**:
   - Processes new audio files and predicts their status.

## Usage

### Step 1: Data Preprocessing
- Update the dataset path in the script:
  ```python
  data_path = r"D:\Air_compressor\AirDataset\AirCompressorDataset"
  ```
- Run the script to preprocess data and train the model.

### Step 2: Train the Model
- Train the LSTM model:
  ```python
  history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))
  ```

### Step 3: Evaluate the Model
- Review the model performance metrics:
  ```python
  loss, accuracy, precision, recall, specificity = model.evaluate(X_test, y_test)
  ```

### Step 4: Predict New Audio Files
- Provide the path to a new `.wav` file:
  ```python
  new_file_path = r"D:\Air_compressor\AirDataset\AirCompressorTest\Healthy\preprocess_Reading200.wav"
  ```
- Get the prediction:
  ```python
  predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
  print(f'The Status of Air Compressor Predicted using LSTM:  {predicted_label}')
  ```

## Results and Visualizations

- **Confusion Matrix**: Displays classification performance.
- **Training History**: Includes accuracy and loss plots for training and validation data.

![Confusion Matrix Placeholder](confusion_matrix.png)
![Training History Placeholder](training_history.png)

## Contributing

Contributions are welcome! Please feel free to fork the repository, open issues, or submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
