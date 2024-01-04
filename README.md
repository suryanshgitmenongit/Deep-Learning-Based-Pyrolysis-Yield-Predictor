# Deep-Learning-Based-Pyrolysis-Yield-Predictor
# Pyrolysis Yield Prediction using Deep Neural Networks

This repository contains Python code for predicting pyrolysis yield using deep neural networks. The project leverages TensorFlow and Keras for building and training the neural network model.

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
