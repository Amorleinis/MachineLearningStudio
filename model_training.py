import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import cross_val_score

def train_nlp_model(X_train, X_test, y_train, y_test, model_type, hyperparams):
    """
    Train an NLP model with the given hyperparameters.
    
    Parameters:
    -----------
    X_train : scipy sparse matrix
        Training features
    X_test : scipy sparse matrix
        Testing features
    y_train : array-like
        Training labels
    y_test : array-like
        Testing labels
    model_type : str
        Type of model to train
    hyperparams : dict
        Hyperparameters for the model
    
    Returns:
    --------
    model : sklearn estimator
        Trained model
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Initialize the model based on the specified type
    if model_type == "Naive Bayes":
        model = MultinomialNB(**hyperparams)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(**hyperparams, random_state=42)
    elif model_type == "Support Vector Machine":
        model = SVC(**hyperparams, probability=True, random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(**hyperparams, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate evaluation metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    # Handle multi-class differently for some metrics
    if len(np.unique(y_test)) > 2:
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
    else:
        metrics['precision'] = precision_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1'] = f1_score(y_test, y_pred)
    
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve and AUC if binary classification
    if len(np.unique(y_test)) == 2:
        # Get the index of the positive class
        pos_idx = list(model.classes_).index(1)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, pos_idx])
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['auc'] = auc(fpr, tpr)
    
    # Cross-validation with safety checks
    # Count samples per class to ensure we can do cross-validation
    classes, counts = np.unique(y_train, return_counts=True)
    min_samples_per_class = np.min(counts)
    
    # Default to 5-fold cross-validation
    cv_folds = 5
    
    # Adjust cv_folds if needed to prevent "n_splits > samples per class" error
    actual_cv_folds = min(cv_folds, min_samples_per_class)
    if actual_cv_folds < 2:
        # If too few samples, skip cross-validation
        metrics['cv_accuracy_mean'] = metrics['accuracy']
        metrics['cv_accuracy_std'] = 0.0
    else:
        # Perform cross-validation with adjusted folds
        cv_scores = cross_val_score(model, X_train, y_train, cv=actual_cv_folds, scoring='accuracy')
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
    
    return model, metrics


def train_classification_model(X_train, X_test, y_train, y_test, model_type, hyperparams, cv_folds=5):
    """
    Train a classification model with the given hyperparameters.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Training labels
    y_test : array-like
        Testing labels
    model_type : str
        Type of model to train
    hyperparams : dict
        Hyperparameters for the model
    cv_folds : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    model : sklearn estimator
        Trained model
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Initialize the model based on the specified type
    if model_type == "Logistic Regression":
        model = LogisticRegression(**hyperparams, random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(**hyperparams, random_state=42)
    elif model_type == "Support Vector Machine":
        model = SVC(**hyperparams, probability=True, random_state=42)
    elif model_type == "K-Nearest Neighbors":
        model = KNeighborsClassifier(**hyperparams)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(**hyperparams, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Try to get probabilities (some models might not support predict_proba)
    try:
        y_prob = model.predict_proba(X_test)
    except:
        y_prob = None
    
    # Calculate evaluation metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    # Handle multi-class differently for some metrics
    if len(np.unique(y_test)) > 2:
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
    else:
        metrics['precision'] = precision_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1'] = f1_score(y_test, y_pred)
    
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve and AUC if binary classification and probabilities are available
    if len(np.unique(y_test)) == 2 and y_prob is not None:
        # Get the index of the positive class
        pos_idx = list(model.classes_).index(1) if hasattr(model, 'classes_') else 1
        
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, pos_idx])
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['auc'] = auc(fpr, tpr)
    
    # Cross-validation with safety checks
    # Count samples per class to ensure we can do cross-validation
    classes, counts = np.unique(y_train, return_counts=True)
    min_samples_per_class = np.min(counts)
    
    # Adjust cv_folds if needed to prevent "n_splits > samples per class" error
    actual_cv_folds = min(cv_folds, min_samples_per_class)
    if actual_cv_folds < 2:
        # If too few samples, skip cross-validation
        metrics['cv_accuracy_mean'] = metrics['accuracy']
        metrics['cv_accuracy_std'] = 0.0
    else:
        # Perform cross-validation with adjusted folds
        cv_scores = cross_val_score(model, X_train, y_train, cv=actual_cv_folds, scoring='accuracy')
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
    
    return model, metrics


def train_image_classification_model(X_train, X_test, y_train, y_test, model_type, hyperparams):
    """
    Train an image classification model.
    
    This is a placeholder function. In a real application, you would use
    deep learning frameworks like TensorFlow/Keras or PyTorch for this task.
    
    Parameters:
    -----------
    X_train : array-like
        Training images
    X_test : array-like
        Testing images
    y_train : array-like
        Training labels
    y_test : array-like
        Testing labels
    model_type : str
        Type of model to train
    hyperparams : dict
        Hyperparameters for the model
    
    Returns:
    --------
    model : object
        Trained model
    metrics : dict
        Dictionary of evaluation metrics
    """
    # This is a placeholder function
    # In a real application, you would implement CNN training here
    
    # For demonstration purposes, return dummy model and metrics
    dummy_model = {"model_type": model_type, "hyperparams": hyperparams}
    
    dummy_metrics = {
        "accuracy": 0.85,
        "precision": 0.84,
        "recall": 0.83,
        "f1": 0.83,
        "confusion_matrix": np.array([[40, 10], [10, 40]])
    }
    
    return dummy_model, dummy_metrics


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : array-like
        Testing features
    y_test : array-like
        Testing labels
    
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Try to get probabilities (some models might not support predict_proba)
    try:
        y_prob = model.predict_proba(X_test)
    except:
        y_prob = None
    
    # Calculate evaluation metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    # Handle multi-class differently for some metrics
    if len(np.unique(y_test)) > 2:
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
    else:
        metrics['precision'] = precision_score(y_test, y_pred)
        metrics['recall'] = recall_score(y_test, y_pred)
        metrics['f1'] = f1_score(y_test, y_pred)
    
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve and AUC if binary classification and probabilities are available
    if len(np.unique(y_test)) == 2 and y_prob is not None:
        # Get the index of the positive class
        pos_idx = list(model.classes_).index(1) if hasattr(model, 'classes_') else 1
        
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, pos_idx])
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['auc'] = auc(fpr, tpr)
    
    return metrics
