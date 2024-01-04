# Deep-Learning-Based-Pyrolysis-Yield-Predictor
# Pyrolysis Yield Prediction using Deep Neural Networks
In the provided code, the deep neural network model is constructed in the `build_ann` function. Here are the details of the architecture and hyperparameters used in the model:

### Neural Network Architecture:

1. **Input Layer:**
   - Shape: The input shape is determined by the number of features in the dataset (`X_train.shape[1]`).

2. **Hidden Layers:**
   - Batch Normalization: Applied before each dense layer.
   - Dense Layer 1:
     - Units: 20
     - Activation: ReLU (Rectified Linear Unit)
   - Batch Normalization
   - Dense Layer 2:
     - Units: 20
     - Activation: ReLU (Rectified Linear Unit)

3. **Output Layer:**
   - Dense Layer:
     - Units: 1 (since it's a regression problem predicting a single continuous value)
     - Activation: None (linear activation for regression)

### Hyperparameters:

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Batch Size:** 16
- **Epochs:** 150 (for the initial training) and 200 (for the training with validation data)
- **Learning Rate (not explicitly specified):** Adam optimizer uses an adaptive learning rate.

### Model Compilation:

The model is compiled using the Adam optimizer and Mean Squared Error as the loss function:

```python
model.compile(optimizer="adam", loss="mean_squared_error")
```

### Training:

The model is trained using the training data (`X_train`, `y_train`) with a batch size of 16 and for 150 epochs:

```python
model.fit(X_train, y_train, batch_size=16, epochs=150)
```

A second training phase is conducted with validation data for 200 epochs:

```python
history = model.fit(X_train, y_train, batch_size=16, epochs=200, validation_data=(X_test, y_test))
```

### Note:

The specific choice of architecture and hyperparameters can be further tuned based on the characteristics of your dataset and the problem at hand. Adjustments to the number of layers, units, learning rate, and other hyperparameters may be necessary for optimal performance. Experimenting with different configurations is recommended to find the best model for your specific use case.

## Installation

Before running the code, make sure to install the required dependencies:

```bash
pip install tensorflow
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/pyrolysis-yield-prediction.git
cd pyrolysis-yield-prediction
```

2. Run the code:

```bash
python main.py
```

The `main.py` script reads a dataset from an Excel file (`Dataset.xlsx`), preprocesses the data, builds and trains a deep neural network model, and finally evaluates the model's performance.

## Code Structure

- `main.py`: The main script to execute the pyrolysis yield prediction.
- `utils.py`: Contains utility functions for data preprocessing and model evaluation.
- `Dataset.xlsx`: Input dataset in Excel format.

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Workflow

1. **Data Loading**: The dataset is loaded from the provided Excel file.

2. **Data Preprocessing**: The data undergoes preprocessing, which includes one-hot encoding categorical variables and handling missing values.

3. **Data Splitting**: The dataset is split into training and testing sets.

4. **Model Building**: A deep neural network model is constructed using TensorFlow and Keras.

5. **Model Training**: The model is trained on the training dataset.

6. **Model Evaluation**: The model's performance is evaluated on the test dataset, and predictions are compared with the actual values.

7. **Mean Squared Error (MSE) Plotting**: The Mean Squared Error during training and validation is plotted for performance analysis.
