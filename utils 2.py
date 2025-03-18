import os
import pickle
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io

def save_model(model, preprocessor=None, preprocessing_options=None):
    """
    Save a trained model and its preprocessing pipeline.
    
    Parameters:
    -----------
    model : trained model
        The model to save
    preprocessor : object, default=None
        Preprocessing pipeline or vectorizer
    preprocessing_options : list, default=None
        List of preprocessing options used
    
    Returns:
    --------
    filename : str
        The filename where the model was saved
    """
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get model type
    if hasattr(model, '__class__'):
        model_type = model.__class__.__name__
    else:
        model_type = "model"
    
    # Create filename
    filename = f"{model_type}_{timestamp}.pkl"
    
    # Create a dictionary with all components to save
    save_dict = {
        "model": model,
        "preprocessor": preprocessor,
        "preprocessing_options": preprocessing_options,
        "timestamp": timestamp
    }
    
    # Save the model
    with open(filename, 'wb') as f:
        pickle.dump(save_dict, f)
    
    return filename


def load_model(filename):
    """
    Load a saved model and its preprocessing pipeline.
    
    Parameters:
    -----------
    filename : str
        Path to the saved model file
    
    Returns:
    --------
    model : trained model
        The loaded model
    preprocessor : object
        The loaded preprocessor
    preprocessing_options : list
        The preprocessing options used
    """
    with open(filename, 'rb') as f:
        save_dict = pickle.load(f)
    
    model = save_dict["model"]
    preprocessor = save_dict["preprocessor"]
    preprocessing_options = save_dict["preprocessing_options"]
    
    return model, preprocessor, preprocessing_options


def export_results_to_csv(results_df, filename=None):
    """
    Export results to a CSV file.
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        DataFrame containing the results
    filename : str, default=None
        Filename for the CSV file
    
    Returns:
    --------
    csv_data : bytes
        CSV data for download in Streamlit
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.csv"
    
    # Convert DataFrame to CSV
    csv = results_df.to_csv(index=False)
    
    # Return CSV as bytes for Streamlit download
    return csv.encode()


def create_model_report(model, metrics, dataset_name, feature_names=None):
    """
    Create a comprehensive model report.
    
    Parameters:
    -----------
    model : trained model
        The model for the report
    metrics : dict
        Dictionary of evaluation metrics
    dataset_name : str
        Name of the dataset used
    feature_names : list, default=None
        Names of features used
    
    Returns:
    --------
    report_html : str
        HTML report content
    """
    # Get model type
    if hasattr(model, '__class__'):
        model_type = model.__class__.__name__
    else:
        model_type = "Unknown Model"
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the report
    report = ["<html><head><style>",
              "body { font-family: Arial, sans-serif; margin: 20px; }",
              "h1, h2 { color: #2c3e50; }",
              "table { border-collapse: collapse; width: 100%; }",
              "th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }",
              "th { background-color: #f2f2f2; }",
              ".metric-card { display: inline-block; background-color: #f8f9fa; border-radius: 5px;",
              "               padding: 10px; margin: 10px; width: 200px; text-align: center; }",
              ".metric-value { font-size: 24px; font-weight: bold; color: #3498db; }",
              ".metric-name { font-size: 14px; color: #7f8c8d; }",
              "</style></head><body>"]
    
    # Header
    report.append(f"<h1>Model Performance Report</h1>")
    report.append(f"<p>Generated on {timestamp}</p>")
    
    # Model information
    report.append("<h2>Model Information</h2>")
    report.append("<table>")
    report.append(f"<tr><th>Model Type</th><td>{model_type}</td></tr>")
    report.append(f"<tr><th>Dataset</th><td>{dataset_name}</td></tr>")
    report.append("</table>")
    
    # Performance metrics
    report.append("<h2>Performance Metrics</h2>")
    report.append("<div style='display: flex; flex-wrap: wrap;'>")
    
    # Add metric cards
    metrics_to_display = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics_to_display:
        if metric in metrics:
            report.append("<div class='metric-card'>")
            report.append(f"<div class='metric-value'>{metrics[metric]:.4f}</div>")
            report.append(f"<div class='metric-name'>{metric.replace('_', ' ').title()}</div>")
            report.append("</div>")
    
    report.append("</div>")
    
    # Cross-validation results if available
    if 'cv_accuracy_mean' in metrics:
        report.append("<h2>Cross-Validation Results</h2>")
        report.append("<table>")
        report.append(f"<tr><th>Mean CV Accuracy</th><td>{metrics['cv_accuracy_mean']:.4f}</td></tr>")
        report.append(f"<tr><th>Standard Deviation</th><td>{metrics['cv_accuracy_std']:.4f}</td></tr>")
        report.append("</table>")
    
    # Feature importance if available
    if feature_names is not None and (hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')):
        report.append("<h2>Feature Importance</h2>")
        report.append("<p>This would include a feature importance plot in a real report.</p>")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        report.append("<h2>Confusion Matrix</h2>")
        report.append("<p>This would include a confusion matrix visualization in a real report.</p>")
    
    # ROC curve
    if 'auc' in metrics:
        report.append("<h2>ROC Curve</h2>")
        report.append(f"<p>AUC: {metrics['auc']:.4f}</p>")
        report.append("<p>This would include an ROC curve visualization in a real report.</p>")
    
    # Close the HTML
    report.append("</body></html>")
    
    # Join all parts of the report
    report_html = "\n".join(report)
    
    return report_html


def generate_sample_predictions(model, preprocessor, X_samples, feature_names=None):
    """
    Generate predictions for sample data points.
    
    Parameters:
    -----------
    model : trained model
        The model to use for predictions
    preprocessor : object
        Preprocessing pipeline or vectorizer
    X_samples : array-like
        Sample data points
    feature_names : list, default=None
        Names of features
    
    Returns:
    --------
    predictions_df : pandas DataFrame
        DataFrame containing input features and predictions
    """
    # Preprocess the samples
    if preprocessor is not None:
        X_samples_processed = preprocessor.transform(X_samples)
    else:
        X_samples_processed = X_samples
    
    # Make predictions
    y_pred = model.predict(X_samples_processed)
    
    # Try to get prediction probabilities
    try:
        y_prob = model.predict_proba(X_samples_processed)
        has_proba = True
    except:
        has_proba = False
    
    # Create a DataFrame with the samples and predictions
    if isinstance(X_samples, pd.DataFrame):
        predictions_df = X_samples.copy()
    else:
        if feature_names is not None:
            predictions_df = pd.DataFrame(X_samples, columns=feature_names)
        else:
            predictions_df = pd.DataFrame(X_samples, columns=[f'Feature {i+1}' for i in range(X_samples.shape[1])])
    
    # Add predictions
    predictions_df['Prediction'] = y_pred
    
    # Add probabilities if available
    if has_proba:
        class_names = model.classes_ if hasattr(model, 'classes_') else [f'Class {i}' for i in range(y_prob.shape[1])]
        for i, class_name in enumerate(class_names):
            predictions_df[f'Probability_{class_name}'] = y_prob[:, i]
    
    return predictions_df
