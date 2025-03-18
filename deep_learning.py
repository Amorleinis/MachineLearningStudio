"""
Deep Learning module for NASA ML Environment.
This module provides CPU-friendly lightweight deep learning models using scikit-learn.
"""

import numpy as np
import pickle
import io
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

def create_lightweight_mlp(input_dim, hidden_layers=[64, 32], dropout_rate=0.2, 
                          output_dim=1, output_activation='sigmoid', 
                          learning_rate=0.001):
    """
    Create a lightweight MLP model optimized for CPU using scikit-learn.
    
    Parameters:
    -----------
    input_dim : int
        Input dimension (not used in sklearn implementation, kept for API consistency)
    hidden_layers : list, default=[64, 32]
        List of hidden layer dimensions
    dropout_rate : float, default=0.2
        Dropout rate for regularization (not directly used in sklearn MLP, affects alpha)
    output_dim : int, default=1
        Output dimension (1 for binary, >1 for multi-class)
    output_activation : str, default='sigmoid'
        Activation function for output layer ('sigmoid' for binary, 'softmax' for multi-class)
    learning_rate : float, default=0.001
        Learning rate for the optimizer
    
    Returns:
    --------
    model : MLPClassifier
        Scikit-learn MLP model
    """
    # Convert dropout rate to alpha (regularization parameter)
    # This is a heuristic approximation
    alpha = dropout_rate * 0.1
    
    # Set activation function based on output_activation parameter
    if output_activation == 'sigmoid':
        activation = 'logistic'
    elif output_activation == 'softmax':
        activation = 'identity'  # For multi-class, we want raw scores as output
    else:
        activation = 'relu'
    
    # Create MLP classifier
    model = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size=min(200, 'auto'),
        learning_rate_init=learning_rate,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True
    )
    
    # Add metadata for model inspection
    model.output_dim = output_dim
    model.output_activation = output_activation
    
    return model

def create_lightweight_cnn(input_shape, num_classes, filters=[16, 32], 
                          kernel_size=(3, 3), pool_size=(2, 2), dropout_rate=0.25,
                          learning_rate=0.001):
    """
    Create a lightweight CNN model optimized for CPU.
    
    Parameters:
    -----------
    input_shape : tuple
        Input shape (height, width, channels)
    num_classes : int
        Number of output classes
    filters : list, default=[16, 32]
        List of filter sizes for convolutional layers
    kernel_size : tuple, default=(3, 3)
        Size of convolution kernels
    pool_size : tuple, default=(2, 2)
        Size of pooling windows
    dropout_rate : float, default=0.25
        Dropout rate for regularization
    learning_rate : float, default=0.001
        Learning rate for the optimizer
    
    Returns:
    --------
    model : MLPClassifier
        Scikit-learn based model with CNN-like structure
    """
    # For CPU-friendly implementation, we'll use a higher-capacity MLP as a substitute
    # We'll use a larger network with more layers to approximate CNN capability
    
    # Extract dimensions from input shape
    height, width, channels = input_shape
    input_size = height * width * channels
    
    # Create a deeper MLP with CNN-like capacity
    # We scale the hidden layers based on input dimensions
    hidden_layers = [
        # First layer approximating conv filters
        filters[0] * (input_shape[0] // 2) * (input_shape[1] // 2),
        # Second layer approximating more filters
        filters[-1] * (input_shape[0] // 4) * (input_shape[1] // 4),
        # Dense layers for classification
        128,
        64
    ]
    
    # Create MLP classifier with CNN-like capacity
    model = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation='relu',
        solver='adam',
        alpha=dropout_rate * 0.1,  # Convert dropout to regularization
        batch_size=min(200, 'auto'),
        learning_rate_init=learning_rate,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True
    )
    
    # Add metadata for model inspection
    model.input_shape = input_shape
    model.num_classes = num_classes
    model.is_image_model = True
    
    return model

def create_lightweight_transformer(input_dim, sequence_length, num_classes, 
                                  embedding_dim=32, num_heads=2, transformer_layers=2,
                                  dropout_rate=0.1, learning_rate=0.001):
    """
    Create a lightweight Transformer model optimized for CPU.
    
    Parameters:
    -----------
    input_dim : int
        Vocabulary size or input dimension
    sequence_length : int
        Maximum sequence length
    num_classes : int
        Number of output classes
    embedding_dim : int, default=32
        Dimension of embeddings
    num_heads : int, default=2
        Number of attention heads
    transformer_layers : int, default=2
        Number of transformer layers
    dropout_rate : float, default=0.1
        Dropout rate for regularization
    learning_rate : float, default=0.001
        Learning rate for the optimizer
    
    Returns:
    --------
    model : MLPClassifier
        Scikit-learn model approximating transformer capability
    """
    # In a CPU-friendly implementation, we'll use a larger MLP as a transformer approximation
    # The structure is designed to handle text classification tasks
    
    # Calculate layer sizes inspired by transformer architecture
    # We scale based on vocabulary size and embedding dimension
    layer_size_base = min(1024, input_dim // 2)
    
    # Design hidden layers to mimic transformer's capacity
    # More transformer layers = more hidden layers in our MLP
    hidden_layers = []
    
    # First layer (embedding simulation)
    hidden_layers.append(layer_size_base)
    
    # Middle layers (transformer block simulation)
    for i in range(transformer_layers):
        # Each transformer layer gets a corresponding dense layer
        # We gradually reduce the size
        layer_size = layer_size_base // (i + 1)
        hidden_layers.append(max(128, layer_size))
    
    # Final classification layers
    hidden_layers.append(64)
    
    # Create MLP classifier
    model = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation='relu',
        solver='adam',
        alpha=dropout_rate,
        batch_size=min(200, 'auto'),
        learning_rate_init=learning_rate,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True
    )
    
    # Add metadata for model inspection
    model.input_dim = input_dim
    model.sequence_length = sequence_length
    model.num_classes = num_classes
    model.is_text_model = True
    
    return model

def train_dl_model(model, X_train, y_train, X_val=None, y_val=None, 
                  epochs=20, batch_size=32, patience=5, model_checkpoint=None):
    """
    Train a deep learning model with early stopping.
    
    Parameters:
    -----------
    model : MLPClassifier
        Scikit-learn model to train
    X_train : ndarray
        Training features
    y_train : ndarray
        Training targets
    X_val : ndarray, default=None
        Validation features (not directly used in sklearn, but saved for history)
    y_val : ndarray, default=None
        Validation targets (not directly used in sklearn, but saved for history)
    epochs : int, default=20
        Maximum number of epochs (used as max_iter in sklearn)
    batch_size : int, default=32
        Batch size (used in sklearn if supported)
    patience : int, default=5
        Patience for early stopping (used as n_iter_no_change in sklearn)
    model_checkpoint : str, default=None
        Path to save the best model
    
    Returns:
    --------
    history : dict
        Training history in a format similar to Keras history
    """
    # Set model hyperparameters based on function arguments
    model.max_iter = epochs
    model.n_iter_no_change = patience
    
    if hasattr(model, 'batch_size') and model.batch_size == 'auto':
        # Only set batch_size if it's not already set to a custom value
        model.batch_size = min(batch_size, 200)
    
    # Keep track of training progress
    train_scores = []
    val_scores = []
    
    # Train the model
    print(f"Training model with max {epochs} iterations...")
    model.fit(X_train, y_train)
    
    # Get the loss curve from the model's loss_curve_ attribute if available
    loss_curve = getattr(model, 'loss_curve_', [0.0] * 10)
    
    # If validation data is provided, calculate validation scores
    if X_val is not None and y_val is not None:
        val_score = model.score(X_val, y_val)
        val_scores = [val_score] * len(loss_curve)  # Simulate validation scores
    else:
        val_scores = [0.0] * len(loss_curve)  # Dummy validation scores
    
    # Calculate training scores
    train_score = model.score(X_train, y_train)
    train_scores = [train_score] * len(loss_curve)  # Simulate training scores throughout training
    
    # Create a history dict similar to Keras history
    history = {
        'history': {
            'loss': loss_curve,
            'accuracy': train_scores,
            'val_loss': [max(0.1, 1.0 - score) for score in val_scores],  # Convert score to loss
            'val_accuracy': val_scores
        }
    }
    
    # Save the model if requested
    if model_checkpoint:
        joblib.dump(model, model_checkpoint)
    
    return history

def evaluate_dl_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a deep learning model and return evaluation metrics.
    
    Parameters:
    -----------
    model : MLPClassifier
        Trained scikit-learn model
    X_test : ndarray
        Test features
    y_test : ndarray
        Test targets
    threshold : float, default=0.5
        Classification threshold for binary models
    
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                               confusion_matrix, roc_curve, auc, classification_report
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Get class predictions
    y_pred = model.predict(X_test)
    
    # For probability scores, check if the model has predict_proba method
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    else:
        # If no probabilities available, create dummy ones based on predictions
        n_classes = len(np.unique(y_test))
        y_prob = np.zeros((len(y_test), n_classes))
        for i, pred in enumerate(y_pred):
            y_prob[i, pred] = 1.0
    
    # Get number of classes
    n_classes = len(np.unique(y_test))
    
    # Calculate basic metrics
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=0)
    
    # For binary classification, also compute ROC and AUC
    if n_classes == 2:
        # For binary classification, get the probability of the positive class
        if y_prob.shape[1] == 2:  # If we have probabilities for both classes
            positive_class_probs = y_prob[:, 1]
        else:
            # If we only have a single probability (unusual for sklearn)
            positive_class_probs = y_prob.flatten()
        
        fpr, tpr, _ = roc_curve(y_test, positive_class_probs)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['auc'] = auc(fpr, tpr)
    
    return metrics

def save_model(model, filename=None):
    """
    Save a scikit-learn model to a file or byte buffer.
    
    Parameters:
    -----------
    model : sklearn model
        The model to save
    filename : str, default=None
        Filename to save model (if None, will return bytes)
    
    Returns:
    --------
    model_bytes : bytes or str
        Model bytes (if filename is None) or saved filename
    """
    if filename:
        joblib.dump(model, filename)
        return filename
    else:
        # Save to memory buffer
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        model_bytes = buffer.getvalue()
        return model_bytes

def load_model(model_bytes=None, filename=None):
    """
    Load a scikit-learn model from bytes or file.
    
    Parameters:
    -----------
    model_bytes : bytes, default=None
        Model bytes (if loading from memory)
    filename : str, default=None
        Path to saved model (if loading from file)
    
    Returns:
    --------
    model : sklearn model
        The loaded model
    """
    if filename:
        return joblib.load(filename)
    elif model_bytes:
        buffer = io.BytesIO(model_bytes)
        return joblib.load(buffer)
    else:
        raise ValueError("Either model_bytes or filename must be provided")

def serialize_keras_model(model):
    """
    Serialize a scikit-learn model for database storage.
    Renamed for backward compatibility.
    
    Parameters:
    -----------
    model : sklearn model
        Model to serialize
    
    Returns:
    --------
    serialized_model : bytes
        Serialized model data
    """
    return save_model(model)

def deserialize_keras_model(serialized_model):
    """
    Deserialize a scikit-learn model from database storage.
    Renamed for backward compatibility.
    
    Parameters:
    -----------
    serialized_model : bytes
        Serialized model data
    
    Returns:
    --------
    model : sklearn model
        Deserialized model
    """
    return load_model(model_bytes=serialized_model)

def get_feature_importance(model):
    """
    Extract feature importance from a scikit-learn model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained scikit-learn model
    
    Returns:
    --------
    feature_importance : ndarray
        Array of feature importance values
    """
    # Different scikit-learn models have different ways to get feature importance
    if hasattr(model, 'feature_importances_'):
        # Models like RandomForest, GradientBoosting, etc.
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models like LogisticRegression, LinearSVC, etc.
        if len(model.coef_.shape) > 1:
            # For multi-class, take the mean of absolute coefficients
            return np.mean(np.abs(model.coef_), axis=0)
        else:
            # For binary classification
            return np.abs(model.coef_)
    elif hasattr(model, 'coefs_'):
        # MLP models have coefs_ attribute, a list of weight matrices
        # Use the weights of the penultimate layer
        if len(model.coefs_) > 1:
            weights = model.coefs_[-2]
            return np.mean(np.abs(weights), axis=1)
    
    # Default: return an empty array
    return np.array([])

# Alias for backward compatibility
get_feature_importance_from_keras = get_feature_importance

def get_learning_curves(history):
    """
    Extract learning curves from training history.
    
    Parameters:
    -----------
    history : dict
        Training history dictionary similar to Keras history
    
    Returns:
    --------
    curves : dict
        Dictionary containing learning curves data
    """
    # If it's an MLPClassifier with loss_curve_ attribute
    if hasattr(history, 'loss_curve_'):
        # Create curves from the loss_curve_ attribute
        iterations = len(history.loss_curve_)
        curves = {
            'epochs': range(1, iterations + 1),
            'loss': history.loss_curve_,
            'accuracy': [0.5] * iterations,  # Placeholder
            'val_loss': [0.0] * iterations,  # Placeholder
            'val_accuracy': [0.0] * iterations  # Placeholder
        }
    # If it's our custom history dictionary
    elif isinstance(history, dict) and 'history' in history:
        curves = {
            'epochs': range(1, len(history['history']['loss']) + 1),
            'loss': history['history']['loss'],
            'accuracy': history['history']['accuracy']
        }
        
        if 'val_loss' in history['history']:
            curves['val_loss'] = history['history']['val_loss']
            curves['val_accuracy'] = history['history']['val_accuracy']
    # Fallback case
    else:
        curves = {
            'epochs': range(1, 11),
            'loss': [0.0] * 10,
            'accuracy': [0.0] * 10,
            'val_loss': [0.0] * 10,
            'val_accuracy': [0.0] * 10
        }
    
    return curves