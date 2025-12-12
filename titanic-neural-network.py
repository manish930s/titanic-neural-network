"""
Titanic Survival Prediction using Neural Networks
==================================================
This script builds and trains a simple neural network to predict 
survival on the Titanic dataset using TensorFlow/Keras.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# ============================================================================
# CONSTANTS
# ============================================================================
DATASET_PATH = r'/content/titanic.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 88
BATCH_SIZE = 32
HIDDEN_UNITS = 10


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(filepath):
    """Load the Titanic dataset from CSV file."""
    return pd.read_csv(filepath)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def preprocess_data(df):
    """
    Preprocess the Titanic dataset.
    
    Steps:
    1. Drop unnecessary columns
    2. Convert categorical variables to dummy variables
    3. Separate features and target variable
    
    Args:
        df: Raw dataset
        
    Returns:
        X: Feature matrix
        y: Target variable
    """
    # Drop columns with missing values or not needed for modeling
    df.drop(['age', 'embarked'], axis=1, inplace=True)
    
    # Convert categorical variables to dummy variables (one-hot encoding)
    df = pd.get_dummies(
        df, 
        columns=['sex', 'class', 'who', 'deck'], 
        drop_first=True
    )
    
    # Separate features and target
    columns_to_drop = ['survived', 'alive', 'embark_town', 'adult_male', 'alone']
    X = df.drop(columns_to_drop, axis=1)
    y = df['survived']
    
    return X, y


def split_and_scale_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into train/test sets and standardize features.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Scaled train and test sets
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# MODEL BUILDING
# ============================================================================
def build_model(input_shape, hidden_units=HIDDEN_UNITS):
    """
    Build a simple neural network for binary classification.
    
    Architecture:
    - Input layer: Dense layer with ReLU activation
    - Output layer: Dense layer with Sigmoid activation
    
    Args:
        input_shape: Shape of input features
        hidden_units: Number of units in hidden layer
        
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # Input layer with ReLU activation
        tf.keras.layers.Dense(
            hidden_units, 
            activation='relu', 
            input_shape=(input_shape,),
            name='input_layer'
        ),
        
        # Output layer with Sigmoid activation for binary classification
        tf.keras.layers.Dense(
            1, 
            activation='sigmoid',
            name='output_layer'
        )
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_model(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train the neural network model.
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Training history
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return history


# ============================================================================
# MODEL EVALUATION
# ============================================================================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f'Test Loss:     {loss:.4f}')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print("="*50)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    # Load data
    print("Loading dataset...")
    dataset = load_data(DATASET_PATH)
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data(dataset)
    
    # Split and scale data
    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)
    
    # Build model
    print("Building model...")
    model = build_model(input_shape=X_train.shape[1])
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
