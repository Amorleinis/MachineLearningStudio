import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve
from wordcloud import WordCloud
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from itertools import cycle

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list
        List of class names
    """
    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='d', ax=ax)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    st.pyplot(fig)


def plot_roc_curve(fpr, tpr, auc_value):
    """
    Plot an ROC curve.
    
    Parameters:
    -----------
    fpr : array-like
        False positive rates
    tpr : array-like
        True positive rates
    auc_value : float
        Area under the ROC curve
    """
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {auc_value:.3f})')
    )
    
    # Add diagonal line (random classifier)
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray'))
    )
    
    # Update layout
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700,
        height=500
    )
    
    st.plotly_chart(fig)


def plot_feature_importance(model, feature_names, top_n=20, is_coef=False):
    """
    Plot feature importance or coefficients.
    
    Parameters:
    -----------
    model : trained model
        The model to extract feature importance from
    feature_names : list or array
        Names of the features
    top_n : int, default=20
        Number of top features to display
    is_coef : bool, default=False
        If True, plot coefficients instead of feature importance
    """
    # Get feature importance or coefficients
    if is_coef:
        # For linear models with coefficients
        importance = model.coef_[0]
        title = 'Feature Coefficients'
    else:
        # For tree-based models with feature_importances_
        importance = model.feature_importances_
        title = 'Feature Importance'
    
    # Create a DataFrame of features and their importance
    if len(feature_names) == len(importance):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
    else:
        # If dimensions don't match (e.g., after one-hot encoding)
        st.warning("Feature names and importance dimensions don't match. Displaying without feature names.")
        feature_importance = pd.DataFrame({
            'Feature': [f"Feature {i}" for i in range(len(importance))],
            'Importance': importance
        })
    
    # Sort by importance and take top N
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title=title
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig)


def plot_learning_curve(estimator, X, y, cv=5, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Plot a learning curve.
    
    Parameters:
    -----------
    estimator : estimator instance
        The estimator to use for the learning curve
    X : array-like
        Training data
    y : array-like
        Training target
    cv : int, cross-validation generator or iterable, default=5
        Cross-validation strategy
    n_jobs : int or None, default=None
        Number of jobs to run in parallel
    train_sizes : array-like, default=np.linspace(0.1, 1.0, 5)
        Relative or absolute numbers of training examples to use
    """
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    
    # Calculate mean and standard deviation
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'Training Size': np.repeat(train_sizes, 2),
        'Score': np.concatenate([train_scores_mean, test_scores_mean]),
        'Error': np.concatenate([train_scores_std, test_scores_std]),
        'Set': ['Training'] * len(train_sizes) + ['Validation'] * len(train_sizes)
    })
    
    # Create the plot
    fig = px.line(
        df,
        x='Training Size',
        y='Score',
        color='Set',
        error_y='Error',
        title='Learning Curve',
        labels={'Score': 'Accuracy', 'Training Size': 'Training Examples'}
    )
    
    fig.update_layout(
        xaxis_title='Training Examples',
        yaxis_title='Accuracy',
        legend_title='Dataset'
    )
    
    st.plotly_chart(fig)


def plot_text_wordcloud(text_data, max_words=100):
    """
    Plot a word cloud from text data.
    
    Parameters:
    -----------
    text_data : list or pandas Series
        Text data to generate word cloud from
    max_words : int, default=100
        Maximum number of words in the word cloud
    """
    # Combine all text
    if isinstance(text_data, pd.Series):
        all_text = ' '.join(text_data.astype(str).values)
    else:
        all_text = ' '.join(text_data)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color='white',
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue'
    ).generate(all_text)
    
    # Display the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)


def plot_image_grid(images, labels=None, n_cols=5, fig_size=(15, 10)):
    """
    Plot a grid of images.
    
    Parameters:
    -----------
    images : list or array
        List of images to display
    labels : list, default=None
        Labels for the images
    n_cols : int, default=5
        Number of columns in the grid
    fig_size : tuple, default=(15, 10)
        Figure size
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n_images:
            ax.imshow(images[i])
            if labels is not None:
                ax.set_title(str(labels[i]))
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)


def plot_prediction_distribution(predictions, class_names):
    """
    Plot the distribution of predictions.
    
    Parameters:
    -----------
    predictions : array-like
        Predicted labels
    class_names : list
        List of class names
    """
    # Count the occurrences of each class
    counts = pd.Series(predictions).value_counts().sort_index()
    
    # Create a bar chart
    fig = px.bar(
        x=[class_names[i] if i < len(class_names) else f'Class {i}' for i in counts.index],
        y=counts.values,
        labels={'x': 'Class', 'y': 'Count'},
        title='Distribution of Predictions'
    )
    
    st.plotly_chart(fig)


def plot_scatter_with_classes(X, y, feature_names, class_names=None):
    """
    Create a scatter plot of data points, colored by class.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    feature_names : list
        Names of features
    class_names : list, default=None
        Names of classes
    """
    # If no class names are provided, use default labels
    if class_names is None:
        class_names = [str(c) for c in np.unique(y)]
    
    # If there are more than 2 features, we'll use PCA for visualization
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        feature_1 = 'PCA Component 1'
        feature_2 = 'PCA Component 2'
        explained_var = pca.explained_variance_ratio_
        title = f'PCA Scatter Plot (Explained Variance: {explained_var[0]:.2f}, {explained_var[1]:.2f})'
    else:
        X_2d = X
        feature_1 = feature_names[0]
        feature_2 = feature_names[1]
        title = 'Feature Scatter Plot'
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'class': [class_names[c] if c < len(class_names) else f'Class {c}' for c in y]
    })
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='class',
        title=title,
        labels={'x': feature_1, 'y': feature_2}
    )
    
    st.plotly_chart(fig)
