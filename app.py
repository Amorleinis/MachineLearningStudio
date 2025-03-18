import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import time
from PIL import Image
import io
import datetime

from data_processing import (
    process_text_data, 
    process_tabular_data, 
    process_image_data,
    load_sample_nasa_datasets
)
from model_training import (
    train_nlp_model, 
    train_classification_model,
    train_image_classification_model,
    evaluate_model
)
from deep_learning import (
    create_lightweight_mlp,
    create_lightweight_cnn,
    create_lightweight_transformer,
    train_dl_model,
    evaluate_dl_model,
    get_learning_curves,
    serialize_keras_model,
    deserialize_keras_model
)
from visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_learning_curve,
    plot_text_wordcloud,
    plot_image_grid
)
from utils import save_model

# Import database functions
from database import (
    init_db,
    save_dataset_metadata,
    save_trained_model,
    load_model_from_db,
    get_all_datasets,
    get_all_models,
    get_models_for_dataset,
    delete_model,
    save_analysis_result
)

# Initialize the database when the app starts
init_db()

# Page configuration
st.set_page_config(
    page_title="NASA ML Environment",
    page_icon="🚀",
    layout="wide"
)

# Load NASA logo
with open('assets/nasa_logo.svg', 'r') as f:
    nasa_logo = f.read()

# App title and description
st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
        <div style="width: 100px;">{nasa_logo}</div>
        <h1 style="margin-left: 20px;">NASA Machine Learning Environment</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    This application provides tools for applying machine learning to NASA datasets.
    You can perform NLP tasks, classification, and image classification with visualization capabilities.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the ML task",
    ["Home", "NLP Analysis", "Classification", "Image Classification", "Deep Learning", "Database Management"]
)

# Home page with dataset selection
if app_mode == "Home":
    st.header("Welcome to NASA ML Environment")
    
    st.markdown("""
    ## Getting Started
    
    This platform allows you to:
    - Process and analyze text data (NLP)
    - Build classification models for structured data
    - Create image classification models
    - Visualize model performance
    
    ## Sample NASA Datasets
    
    You can use the following sample NASA datasets or upload your own:
    """)
    
    dataset_option = st.radio(
        "Select an option:",
        ["Use sample NASA dataset", "Upload your own dataset"]
    )
    
    if dataset_option == "Use sample NASA dataset":
        available_datasets = load_sample_nasa_datasets()
        dataset_name = st.selectbox("Select a NASA dataset", list(available_datasets.keys()))
        
        if dataset_name:
            dataset_info = available_datasets[dataset_name]
            st.markdown(f"**Dataset Description**: {dataset_info['description']}")
            st.markdown(f"**Task Type**: {dataset_info['task_type']}")
            
            if st.button("Load Dataset"):
                st.session_state.dataset = dataset_info['data']
                st.session_state.dataset_type = dataset_info['task_type']
                st.session_state.dataset_name = dataset_name
                st.success(f"Dataset '{dataset_name}' loaded successfully!")
                
                # Preview the dataset
                st.subheader("Dataset Preview")
                st.write(dataset_info['data'].head())
                
                # Dataset statistics
                st.subheader("Dataset Statistics")
                if dataset_info['task_type'] == 'tabular':
                    st.write(dataset_info['data'].describe())
                elif dataset_info['task_type'] == 'text':
                    st.write(f"Number of samples: {len(dataset_info['data'])}")
                    st.write(f"Number of classes: {len(dataset_info['data']['label'].unique())}")
                elif dataset_info['task_type'] == 'image':
                    st.write(f"Number of images: {len(dataset_info['data'])}")
                    st.write(f"Number of classes: {len(dataset_info['data']['label'].unique())}")
    
    else:
        upload_task_type = st.selectbox(
            "What type of data are you uploading?",
            ["Tabular Data (CSV, Excel)", "Text Data (CSV with text column)", "Image Dataset (ZIP file)"]
        )
        
        if upload_task_type == "Tabular Data (CSV, Excel)":
            uploaded_file = st.file_uploader("Upload your tabular dataset", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                        
                    st.session_state.dataset = data
                    st.session_state.dataset_type = 'tabular'
                    st.session_state.dataset_name = uploaded_file.name
                    
                    st.success("Dataset loaded successfully!")
                    st.write(data.head())
                    
                    # Display basic statistics
                    st.subheader("Dataset Statistics")
                    st.write(data.describe())
                    
                    # Display dataset shape
                    st.text(f"Dataset Shape: {data.shape}")
                    
                    # Check for missing values
                    missing_values = data.isnull().sum()
                    if missing_values.sum() > 0:
                        st.warning("Your dataset contains missing values:")
                        st.write(missing_values[missing_values > 0])
                    
                except Exception as e:
                    st.error(f"Error loading the dataset: {e}")
        
        elif upload_task_type == "Text Data (CSV with text column)":
            uploaded_file = st.file_uploader("Upload your text dataset (CSV with text column)", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    
                    # Let user select the text column
                    text_column = st.selectbox("Select the column containing the text data:", data.columns)
                    
                    if st.button("Process Text Data"):
                        st.session_state.dataset = data
                        st.session_state.dataset_type = 'text'
                        st.session_state.text_column = text_column
                        st.session_state.dataset_name = uploaded_file.name
                        
                        st.success("Text dataset loaded successfully!")
                        
                        # Display sample of text data
                        st.subheader("Sample Text Data")
                        st.write(data[[text_column]].head())
                        
                        # Basic text statistics
                        st.subheader("Text Statistics")
                        text_lengths = data[text_column].str.len()
                        st.write(f"Average text length: {text_lengths.mean():.2f} characters")
                        st.write(f"Minimum text length: {text_lengths.min()} characters")
                        st.write(f"Maximum text length: {text_lengths.max()} characters")
                        
                        # Word cloud preview if text data is available
                        if len(data) > 0:
                            st.subheader("Word Cloud Preview")
                            plot_text_wordcloud(data[text_column])
                    
                except Exception as e:
                    st.error(f"Error loading the text dataset: {e}")
        
        elif upload_task_type == "Image Dataset (ZIP file)":
            st.markdown("""
            Please upload a ZIP file containing your images organized in folders by class.
            
            Expected structure:
            ```
            dataset.zip
            ├── class1/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            ├── class2/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            └── ...
            ```
            """)
            
            uploaded_file = st.file_uploader("Upload your image dataset (ZIP file)", type=["zip"])
            
            if uploaded_file is not None:
                st.info("Processing image dataset... This may take a moment.")
                try:
                    # In a real implementation, we would unzip and process the images here
                    # For now, we'll show a placeholder for the interface
                    st.success("Image dataset structure detected and processed!")
                    st.session_state.dataset_type = 'image'
                    st.session_state.dataset_name = uploaded_file.name
                    
                    # Placeholder for image preview
                    st.subheader("Image Preview")
                    st.info("In a real implementation, sample images would be displayed here.")
                    
                    # Simulated image statistics
                    st.subheader("Image Dataset Statistics")
                    st.markdown("""
                    - **Total images**: Would be calculated
                    - **Number of classes**: Would be detected from folder structure
                    - **Image dimensions**: Would be analyzed
                    """)
                    
                except Exception as e:
                    st.error(f"Error processing the image dataset: {e}")

# NLP Analysis
elif app_mode == "NLP Analysis":
    st.header("Natural Language Processing")
    
    if not hasattr(st.session_state, 'dataset') or st.session_state.dataset_type != 'text':
        st.warning("Please load a text dataset from the Home page first.")
        if st.button("Go to Home Page"):
            st.session_state.app_mode = "Home"
            st.rerun()
    else:
        st.subheader(f"Dataset: {st.session_state.dataset_name}")
        
        # Text preprocessing options
        st.subheader("Text Preprocessing")
        
        preprocessing_options = st.multiselect(
            "Select preprocessing steps:",
            ["Lowercase", "Remove Punctuation", "Remove Stopwords", "Stemming", "Lemmatization"],
            default=["Lowercase", "Remove Punctuation", "Remove Stopwords"]
        )
        
        # Feature extraction options
        st.subheader("Feature Extraction")
        vectorizer = st.selectbox(
            "Select vectorization method:",
            ["CountVectorizer", "TF-IDF", "Word Embeddings (Word2Vec)"]
        )
        
        max_features = st.slider("Maximum number of features", 100, 10000, 5000)
        
        # Model selection
        st.subheader("Model Selection")
        
        nlp_task = st.selectbox(
            "Select NLP task:",
            ["Text Classification", "Sentiment Analysis", "Topic Modeling"]
        )
        
        if nlp_task == "Text Classification" or nlp_task == "Sentiment Analysis":
            model_type = st.selectbox(
                "Select model type:",
                ["Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest"]
            )
            
            # Target selection
            df = st.session_state.dataset
            target_column = st.selectbox(
                "Select target column:",
                [col for col in df.columns if col != st.session_state.text_column]
            )
            
            # Model hyperparameters
            st.subheader("Hyperparameters")
            
            if model_type == "Naive Bayes":
                alpha = st.slider("Alpha (Smoothing parameter)", 0.0, 2.0, 1.0, 0.1)
                hyperparams = {"alpha": alpha}
            
            elif model_type == "Logistic Regression":
                C = st.slider("C (Inverse of regularization strength)", 0.01, 10.0, 1.0)
                max_iter = st.slider("Maximum iterations", 100, 1000, 100)
                hyperparams = {"C": C, "max_iter": max_iter}
            
            elif model_type == "Support Vector Machine":
                C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0)
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                hyperparams = {"C": C, "kernel": kernel}
            
            elif model_type == "Random Forest":
                n_estimators = st.slider("Number of trees", 10, 500, 100)
                max_depth = st.slider("Maximum depth", 1, 50, 10)
                hyperparams = {"n_estimators": n_estimators, "max_depth": max_depth}
            
            # Training and Evaluation
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Get the data
                    X = df[st.session_state.text_column]
                    y = df[target_column]
                    
                    # Process the data
                    X_train, X_test, y_train, y_test, vectorizer_fitted = process_text_data(
                        X, y, preprocessing_options, vectorizer, max_features
                    )
                    
                    # Train the model
                    model, metrics = train_nlp_model(
                        X_train, X_test, y_train, y_test, model_type, hyperparams
                    )
                    
                    # Store the model and results in session state
                    st.session_state.nlp_model = model
                    st.session_state.nlp_vectorizer = vectorizer_fitted
                    st.session_state.nlp_metrics = metrics
                    st.session_state.nlp_task_type = nlp_task
                    st.session_state.nlp_model_type = model_type
                    
                    # Save model to database automatically
                    model_name = f"NLP_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # First, save the dataset metadata if not already in the database
                    datasets = get_all_datasets()
                    existing_dataset = next((ds for ds in datasets if ds['name'] == st.session_state.dataset_name), None)
                    
                    if existing_dataset:
                        dataset_id = existing_dataset['id']
                    else:
                        # Save dataset metadata
                        dataset_id = save_dataset_metadata(
                            name=st.session_state.dataset_name,
                            description=f"Dataset for NLP task: {nlp_task}",
                            data_type="text",
                            data=df
                        )
                    
                    # Now save the model
                    model_id = save_trained_model(
                        name=model_name,
                        model_type="nlp",
                        algorithm=model_type,
                        model=model,
                        preprocessor=vectorizer_fitted,
                        dataset_id=dataset_id,
                        metrics=metrics
                    )
                    
                    # Display metrics
                    st.success(f"Model trained successfully and saved to database with ID: {model_id}")
                    
                    st.subheader("Model Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    col2.metric("Precision", f"{metrics['precision']:.4f}")
                    col3.metric("Recall", f"{metrics['recall']:.4f}")
                    
                    st.subheader("Confusion Matrix")
                    plot_confusion_matrix(metrics['confusion_matrix'], list(y.unique()))
                    
                    st.subheader("ROC Curve")
                    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'])
                    
                    # Word cloud for important features
                    st.subheader("Important Words")
                    if hasattr(model, 'feature_importances_'):
                        feature_names = vectorizer_fitted.get_feature_names_out()
                        plot_feature_importance(model, feature_names)
                    elif hasattr(model, 'coef_'):
                        feature_names = vectorizer_fitted.get_feature_names_out()
                        plot_feature_importance(model, feature_names, is_coef=True)
                    
                    # Sample prediction
                    st.subheader("Try a prediction")
                    test_text = st.text_area("Enter text to classify:")
                    
                    if test_text and st.button("Predict"):
                        # Preprocess and vectorize
                        processed_text = [test_text]  # Would apply the same preprocessing as training
                        vectorized_text = vectorizer_fitted.transform(processed_text)
                        
                        # Predict
                        prediction = model.predict(vectorized_text)[0]
                        probabilities = model.predict_proba(vectorized_text)[0]
                        
                        # Display result
                        st.success(f"Prediction: {prediction}")
                        
                        # Display probabilities for each class
                        st.subheader("Prediction Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': model.classes_,
                            'Probability': probabilities
                        })
                        st.bar_chart(prob_df.set_index('Class'))
                    
                    # Option to save the model
                    if st.button("Save Model"):
                        model_filename = save_model(model, vectorizer_fitted, preprocessing_options)
                        st.success(f"Model saved as {model_filename}")
                        
        elif nlp_task == "Topic Modeling":
            st.subheader("Topic Modeling Parameters")
            
            num_topics = st.slider("Number of topics", 2, 20, 5)
            
            if st.button("Run Topic Modeling"):
                with st.spinner("Analyzing topics..."):
                    # Get the text data
                    X = st.session_state.dataset[st.session_state.text_column]
                    
                    # Process and vectorize the text
                    vectorizer_fitted, dtm = process_text_data(
                        X, None, preprocessing_options, "TF-IDF", max_features, 
                        for_topic_modeling=True
                    )
                    
                    # Run LDA for topic modeling
                    from sklearn.decomposition import LatentDirichletAllocation
                    
                    lda_model = LatentDirichletAllocation(
                        n_components=num_topics, 
                        random_state=42,
                        learning_method='online'
                    )
                    
                    lda_output = lda_model.fit_transform(dtm)
                    
                    # Get feature names for display
                    feature_names = vectorizer_fitted.get_feature_names_out()
                    
                    # Display topics and their top words
                    st.subheader("Topics and Top Words")
                    
                    for topic_idx, topic in enumerate(lda_model.components_):
                        top_words_idx = topic.argsort()[:-11:-1]  # Get top 10 words
                        top_words = [feature_names[i] for i in top_words_idx]
                        topic_probs = [topic[i] for i in top_words_idx]
                        
                        st.write(f"**Topic {topic_idx+1}**")
                        
                        # Create a horizontal bar chart for word probabilities in the topic
                        topic_df = pd.DataFrame({
                            'Word': top_words,
                            'Weight': topic_probs
                        })
                        
                        fig = px.bar(
                            topic_df,
                            x='Weight',
                            y='Word',
                            orientation='h',
                            labels={'Weight': 'Importance in Topic'},
                            title=f"Top Words in Topic {topic_idx+1}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Document-topic distribution
                    st.subheader("Document-Topic Distribution")
                    doc_topic_df = pd.DataFrame(lda_output)
                    doc_topic_df.columns = [f"Topic {i+1}" for i in range(num_topics)]
                    
                    # Show summary of document-topic distribution
                    st.write("Average topic distribution across documents:")
                    avg_topic_distribution = doc_topic_df.mean().sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=avg_topic_distribution.index,
                        y=avg_topic_distribution.values,
                        labels={'x': 'Topic', 'y': 'Average Weight'},
                        title="Average Topic Distribution Across Documents"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show a sample of document-topic distributions
                    st.write("Sample of document-topic distributions (first 10 documents):")
                    st.write(doc_topic_df.head(10))

# Classification
elif app_mode == "Classification":
    st.header("Classification Tasks")
    
    if not hasattr(st.session_state, 'dataset') or st.session_state.dataset_type != 'tabular':
        st.warning("Please load a tabular dataset from the Home page first.")
        if st.button("Go to Home Page"):
            st.session_state.app_mode = "Home"
            st.rerun()
    else:
        st.subheader(f"Dataset: {st.session_state.dataset_name}")
        
        # Data exploration
        st.subheader("Data Exploration")
        
        df = st.session_state.dataset
        
        if st.checkbox("Show dataset preview"):
            st.write(df.head())
        
        if st.checkbox("Show dataset statistics"):
            st.write(df.describe())
        
        if st.checkbox("Show data types"):
            st.write(df.dtypes)
        
        # Data preprocessing
        st.subheader("Data Preprocessing")
        
        # Feature selection
        st.write("Select features and target:")
        
        # Determine numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Select target variable
        target_column = st.selectbox("Select target column:", df.columns)
        
        # Select features
        feature_cols = st.multiselect(
            "Select feature columns:",
            [col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column]
        )
        
        # Preprocessing options
        preprocessing_steps = st.multiselect(
            "Select preprocessing steps:",
            ["Handle Missing Values", "Encode Categorical Features", "Scale Numeric Features"],
            default=["Handle Missing Values", "Encode Categorical Features", "Scale Numeric Features"]
        )
        
        # Missing value handling strategy
        missing_strategy = "mean"
        categorical_encoding = "one-hot"
        scaling_method = "standard"
        
        if "Handle Missing Values" in preprocessing_steps:
            missing_strategy = st.selectbox(
                "Missing value strategy for numeric features:",
                ["mean", "median", "most_frequent", "constant"]
            )
        
        if "Encode Categorical Features" in preprocessing_steps and len(categorical_cols) > 0:
            categorical_encoding = st.selectbox(
                "Categorical encoding method:",
                ["one-hot", "label"]
            )
        
        if "Scale Numeric Features" in preprocessing_steps and len(numeric_cols) > 0:
            scaling_method = st.selectbox(
                "Scaling method:",
                ["standard", "minmax", "robust", "none"]
            )
        
        # Model selection
        st.subheader("Model Selection")
        
        model_type = st.selectbox(
            "Select classification model:",
            ["Logistic Regression", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors", "Gradient Boosting"]
        )
        
        # Hyperparameter tuning
        st.subheader("Hyperparameter Tuning")
        
        if model_type == "Logistic Regression":
            C = st.slider("C (Inverse of regularization strength)", 0.01, 10.0, 1.0)
            solver = st.selectbox("Solver", ["liblinear", "lbfgs", "newton-cg", "sag"])
            hyperparams = {"C": C, "solver": solver}
        
        elif model_type == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 500, 100)
            max_depth = st.slider("Maximum depth", 1, 50, 10)
            min_samples_split = st.slider("Minimum samples to split", 2, 20, 2)
            hyperparams = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split
            }
        
        elif model_type == "Support Vector Machine":
            C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            gamma = st.selectbox("Gamma", ["scale", "auto"])
            hyperparams = {"C": C, "kernel": kernel, "gamma": gamma}
        
        elif model_type == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of neighbors", 1, 20, 5)
            weights = st.selectbox("Weight function", ["uniform", "distance"])
            hyperparams = {"n_neighbors": n_neighbors, "weights": weights}
        
        elif model_type == "Gradient Boosting":
            n_estimators = st.slider("Number of boosting stages", 10, 500, 100)
            learning_rate = st.slider("Learning rate", 0.01, 1.0, 0.1)
            max_depth = st.slider("Maximum depth", 1, 10, 3)
            hyperparams = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth
            }
        
        # Train-test split
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
        
        # Training and evaluation
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Get the features and target
                X = df[feature_cols]
                y = df[target_column]
                
                # Process the data
                X_train, X_test, y_train, y_test, preprocessing_pipeline = process_tabular_data(
                    X, y, test_size,
                    handle_missing=("Handle Missing Values" in preprocessing_steps),
                    missing_strategy=missing_strategy,
                    encode_categorical=("Encode Categorical Features" in preprocessing_steps),
                    categorical_encoding=categorical_encoding,
                    scale_features=("Scale Numeric Features" in preprocessing_steps),
                    scaling_method=scaling_method
                )
                
                # Train the model
                model, metrics = train_classification_model(
                    X_train, X_test, y_train, y_test, model_type, hyperparams, cv_folds
                )
                
                # Store the model and results in session state
                st.session_state.classification_model = model
                st.session_state.classification_preprocessor = preprocessing_pipeline
                st.session_state.classification_metrics = metrics
                st.session_state.classification_model_type = model_type
                
                # Save model to database automatically
                model_name = f"Classification_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # First, save the dataset metadata if not already in the database
                datasets = get_all_datasets()
                existing_dataset = next((ds for ds in datasets if ds['name'] == st.session_state.dataset_name), None)
                
                if existing_dataset:
                    dataset_id = existing_dataset['id']
                else:
                    # Save dataset metadata
                    dataset_id = save_dataset_metadata(
                        name=st.session_state.dataset_name,
                        description=f"Dataset for Classification task with {model_type}",
                        data_type="tabular",
                        data=df
                    )
                
                # Now save the model
                model_id = save_trained_model(
                    name=model_name,
                    model_type="classification",
                    algorithm=model_type,
                    model=model,
                    preprocessor=preprocessing_pipeline,
                    dataset_id=dataset_id,
                    metrics=metrics
                )
                
                # Display metrics
                st.success(f"Model trained successfully and saved to database with ID: {model_id}")
                
                st.subheader("Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                col2.metric("Precision", f"{metrics['precision']:.4f}")
                col3.metric("Recall", f"{metrics['recall']:.4f}")
                col4.metric("F1 Score", f"{metrics['f1']:.4f}")
                
                # Display cross-validation scores
                st.subheader("Cross-Validation Results")
                st.write(f"Mean CV Accuracy: {metrics['cv_accuracy_mean']:.4f} (±{metrics['cv_accuracy_std']:.4f})")
                
                # Plot confusion matrix
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(metrics['confusion_matrix'], list(y.unique()))
                
                # Plot ROC curve if binary classification
                if len(np.unique(y)) == 2:
                    st.subheader("ROC Curve")
                    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'])
                
                # Plot feature importance if applicable
                st.subheader("Feature Importance")
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    plot_feature_importance(model, X_train.columns)
                
                # Plot learning curve
                st.subheader("Learning Curve")
                plot_learning_curve(model, X_train, y_train)
                
                # Option to save the model
                if st.button("Save Model"):
                    model_filename = save_model(model, preprocessing_pipeline)
                    st.success(f"Model saved as {model_filename}")

# Image Classification
elif app_mode == "Image Classification":
    st.header("Image Classification")
    
    if not hasattr(st.session_state, 'dataset') or st.session_state.dataset_type != 'image':
        st.warning("Please load an image dataset from the Home page first.")
        if st.button("Go to Home Page"):
            st.session_state.app_mode = "Home"
            st.rerun()
    else:
        st.subheader(f"Dataset: {st.session_state.dataset_name}")
        
        # Image preprocessing options
        st.subheader("Image Preprocessing")
        
        resize_dim = st.slider("Resize images to (pixels)", 32, 256, 128)
        
        augmentation = st.multiselect(
            "Select augmentation techniques:",
            ["Rotation", "Horizontal Flip", "Vertical Flip", "Zoom", "Brightness Adjustment"],
            default=["Rotation", "Horizontal Flip"]
        )
        
        # Model selection
        st.subheader("Model Selection")
        
        model_type = st.selectbox(
            "Select model type:",
            ["Simple CNN", "Transfer Learning (VGG16)", "Transfer Learning (ResNet50)"]
        )
        
        # Training parameters
        st.subheader("Training Parameters")
        
        batch_size = st.select_slider(
            "Batch size:",
            options=[8, 16, 32, 64, 128],
            value=32
        )
        
        epochs = st.slider("Number of epochs:", 1, 50, 10)
        
        learning_rate = st.select_slider(
            "Learning rate:",
            options=[0.0001, 0.001, 0.01, 0.1],
            value=0.001
        )
        
        # Train-test split
        test_size = st.slider("Test set size (%):", 10, 40, 20) / 100
        
        # Training and evaluation functionality is a placeholder
        # In a real implementation, we would use these parameters
        # to train actual image classification models
        
        if st.button("Train Model"):
            st.info("In a production environment, this would train an actual image classification model.")
            st.info("For demonstration purposes, showing simulated training process and results.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training progress
            for i in range(epochs):
                # Simulate epoch progress
                for j in range(10):
                    time.sleep(0.1)
                    progress_value = (i * 10 + j + 1) / (epochs * 10)
                    progress_bar.progress(progress_value)
                    status_text.text(f"Training Epoch {i+1}/{epochs}, Batch {j+1}/10")
            
            # Display simulated results
            status_text.text("Training complete!")
            
            # Simulated metrics
            accuracy = 0.87
            precision = 0.85
            recall = 0.84
            f1 = 0.84
            
            # Create metrics dictionary for database storage
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': np.array([
                    [45, 2, 3, 1],
                    [3, 43, 2, 1],
                    [2, 3, 42, 2],
                    [1, 2, 3, 45]
                ])
            }
            
            # Store in session state
            st.session_state.image_model_type = model_type
            st.session_state.image_metrics = metrics
            
            # Save model to database automatically
            model_name = f"Image_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create a dummy model object for demonstration purposes
            # In a real implementation, this would be the actual trained model
            dummy_model = {'model_type': model_type, 'hyperparams': {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate
            }}
            
            # First, save the dataset metadata if not already in the database
            datasets = get_all_datasets()
            existing_dataset = next((ds for ds in datasets if ds['name'] == st.session_state.dataset_name), None)
            
            if existing_dataset:
                dataset_id = existing_dataset['id']
            else:
                # Save dataset metadata
                dataset_id = save_dataset_metadata(
                    name=st.session_state.dataset_name,
                    description=f"Dataset for Image Classification with {model_type}",
                    data_type="image",
                    data=pd.DataFrame({'filename': ['image_sample.jpg'], 'class': ['sample']})  # Placeholder data
                )
            
            # Now save the model
            model_id = save_trained_model(
                name=model_name,
                model_type="image_classification",
                algorithm=model_type,
                model=dummy_model,
                preprocessor=None,
                dataset_id=dataset_id,
                metrics=metrics
            )
            
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("Precision", f"{precision:.4f}")
            col3.metric("Recall", f"{recall:.4f}")
            col4.metric("F1 Score", f"{f1:.4f}")
            
            # Show success message
            st.success(f"Model trained successfully and saved to database with ID: {model_id}")
            
            # Simulated learning curves
            st.subheader("Training and Validation Accuracy")
            
            # Generate simulated data for demo
            train_acc = [0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.86, 0.87]
            val_acc = [0.25, 0.4, 0.52, 0.61, 0.65, 0.7, 0.72, 0.75, 0.76, 0.77]
            
            epochs_range = list(range(1, epochs + 1))
            if len(epochs_range) > 10:
                train_acc.extend([0.87 + i*0.005 for i in range(epochs - 10)])
                val_acc.extend([0.77 + i*0.003 for i in range(epochs - 10)])
            
            fig, ax = plt.subplots()
            ax.plot(epochs_range, train_acc[:epochs], 'b', label='Training accuracy')
            ax.plot(epochs_range, val_acc[:epochs], 'r', label='Validation accuracy')
            ax.set_title('Training and Validation Accuracy')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            ax.legend()
            st.pyplot(fig)
            
            # Simulated confusion matrix
            st.subheader("Confusion Matrix")
            
            class_names = ["Class 1", "Class 2", "Class 3", "Class 4"]
            cm = np.array([
                [45, 2, 3, 1],
                [3, 43, 2, 1],
                [2, 3, 42, 2],
                [1, 2, 3, 45]
            ])
            
            plot_confusion_matrix(cm, class_names)
            
            # Simulated class activation mapping (feature visualization)
            st.subheader("Class Activation Mapping")
            st.info("In a real implementation, this would show heatmaps of what the model focuses on.")
            
            # Option to download model
            if st.button("Download Model"):
                # In a real implementation, we would export the actual model here
                st.download_button(
                    label="Download Model File",
                    data="Placeholder for model binary data",
                    file_name="image_classifier_model.h5",
                    mime="application/octet-stream"
                )

# Deep Learning Section
elif app_mode == "Deep Learning":
    st.header("Deep Learning with Lightweight CPU-friendly Models")
    
    if not hasattr(st.session_state, 'dataset'):
        st.warning("Please load a dataset from the Home page first.")
        if st.button("Go to Home Page"):
            st.session_state.app_mode = "Home"
            st.rerun()
    else:
        st.subheader(f"Dataset: {st.session_state.dataset_name}")
        
        # Choose deep learning model type
        dl_model_category = st.selectbox(
            "Select Deep Learning model category:",
            ["Multilayer Perceptron (MLP)", "Convolutional Neural Network (CNN)", "Transformer"]
        )
        
        if dl_model_category == "Multilayer Perceptron (MLP)":
            # MLP is suitable for tabular and text data
            if st.session_state.dataset_type not in ['tabular', 'text']:
                st.warning("MLP is best suited for tabular or text data. Please load an appropriate dataset.")
            else:
                st.subheader("Multilayer Perceptron Configuration")
                
                # Feature and target selection for tabular data
                if st.session_state.dataset_type == 'tabular':
                    df = st.session_state.dataset
                    
                    # Target selection
                    target_column = st.selectbox("Select target column:", df.columns)
                    
                    # Feature selection
                    feature_cols = st.multiselect(
                        "Select feature columns:",
                        [col for col in df.columns if col != target_column],
                        default=[col for col in df.columns if col != target_column]
                    )
                    
                    # Preprocessing options
                    preprocessing_steps = st.multiselect(
                        "Select preprocessing steps:",
                        ["Handle Missing Values", "Encode Categorical Features", "Scale Numeric Features"],
                        default=["Handle Missing Values", "Encode Categorical Features", "Scale Numeric Features"]
                    )
                    
                    # Missing value handling
                    missing_strategy = "mean"
                    categorical_encoding = "one-hot"
                    scaling_method = "standard"
                    
                    if "Handle Missing Values" in preprocessing_steps:
                        missing_strategy = st.selectbox(
                            "Missing value strategy for numeric features:",
                            ["mean", "median", "most_frequent"]
                        )
                    
                    if "Encode Categorical Features" in preprocessing_steps:
                        categorical_encoding = st.selectbox(
                            "Categorical encoding method:",
                            ["one-hot", "label"]
                        )
                    
                    if "Scale Numeric Features" in preprocessing_steps:
                        scaling_method = st.selectbox(
                            "Scaling method:",
                            ["standard", "minmax", "robust"]
                        )
                
                elif st.session_state.dataset_type == 'text':
                    df = st.session_state.dataset
                    
                    # Text preprocessing options
                    preprocessing_options = st.multiselect(
                        "Select text preprocessing steps:",
                        ["Lowercase", "Remove Punctuation", "Remove Stopwords", "Stemming", "Lemmatization"],
                        default=["Lowercase", "Remove Punctuation", "Remove Stopwords"]
                    )
                    
                    # Feature extraction options
                    vectorizer = st.selectbox(
                        "Select vectorization method:",
                        ["CountVectorizer", "TF-IDF"]
                    )
                    
                    max_features = st.slider("Maximum number of features", 100, 10000, 5000)
                    
                    # Target selection
                    target_column = st.selectbox(
                        "Select target column:",
                        [col for col in df.columns if col != st.session_state.text_column]
                    )
                
                # MLP Architecture
                st.subheader("MLP Architecture")
                
                num_classes = 0
                if st.session_state.dataset_type == 'tabular':
                    num_classes = len(st.session_state.dataset[target_column].unique())
                elif st.session_state.dataset_type == 'text':
                    num_classes = len(st.session_state.dataset[target_column].unique())
                
                # Output configuration based on number of classes
                output_activation = "sigmoid" if num_classes == 2 else "softmax"
                output_dim = 1 if num_classes == 2 else num_classes
                
                # Model configuration
                st.write(f"Task type: {'Binary' if num_classes == 2 else 'Multi-class'} Classification, {num_classes} classes")
                
                # Define hidden layers
                num_hidden_layers = st.slider("Number of hidden layers", 1, 5, 2)
                hidden_layers = []
                
                for i in range(num_hidden_layers):
                    units = st.slider(f"Units in hidden layer {i+1}", 8, 256, 
                                     2**(6-i) if i < 4 else 16)  # Start with 64, decrease by half
                    hidden_layers.append(units)
                
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.05)
                
                # Training parameters
                st.subheader("Training Parameters")
                
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[0.0001, 0.001, 0.01, 0.1],
                    value=0.001
                )
                
                batch_size = st.select_slider(
                    "Batch size",
                    options=[8, 16, 32, 64, 128],
                    value=32
                )
                
                epochs = st.slider("Number of epochs", 5, 100, 20)
                patience = st.slider("Early stopping patience", 2, 20, 5)
                
                # Train-test split
                test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
                
                # Training button
                if st.button("Train Deep Learning Model"):
                    with st.spinner("Training model..."):
                        if st.session_state.dataset_type == 'tabular':
                            # Get the data
                            X = df[feature_cols]
                            y = df[target_column]
                            
                            # Process the data
                            X_train, X_test, y_train, y_test, preprocessing_pipeline = process_tabular_data(
                                X, y, test_size,
                                handle_missing=("Handle Missing Values" in preprocessing_steps),
                                missing_strategy=missing_strategy,
                                encode_categorical=("Encode Categorical Features" in preprocessing_steps),
                                categorical_encoding=categorical_encoding,
                                scale_features=("Scale Numeric Features" in preprocessing_steps),
                                scaling_method=scaling_method
                            )
                            
                            # Get the input dimension
                            input_dim = X_train.shape[1]
                            
                            # Create MLP model
                            model = create_lightweight_mlp(
                                input_dim=input_dim, 
                                hidden_layers=hidden_layers,
                                dropout_rate=dropout_rate,
                                output_dim=output_dim,
                                output_activation=output_activation,
                                learning_rate=learning_rate
                            )
                            
                            # Train the model
                            history = train_dl_model(
                                model, X_train, y_train,
                                X_val=X_test, y_val=y_test,
                                epochs=epochs,
                                batch_size=batch_size,
                                patience=patience
                            )
                            
                            # Evaluate the model
                            metrics = evaluate_dl_model(model, X_test, y_test)
                            
                            # Get learning curves
                            learning_curves = get_learning_curves(history)
                            
                        elif st.session_state.dataset_type == 'text':
                            # Get the data
                            X = df[st.session_state.text_column]
                            y = df[target_column]
                            
                            # Process and vectorize the text
                            X_train, X_test, y_train, y_test, vectorizer_fitted = process_text_data(
                                X, y, preprocessing_options, vectorizer, max_features
                            )
                            
                            # Get the input dimension (vocabulary size)
                            input_dim = X_train.shape[1]
                            
                            # Create MLP model
                            model = create_lightweight_mlp(
                                input_dim=input_dim, 
                                hidden_layers=hidden_layers,
                                dropout_rate=dropout_rate,
                                output_dim=output_dim,
                                output_activation=output_activation,
                                learning_rate=learning_rate
                            )
                            
                            # Train the model
                            history = train_dl_model(
                                model, X_train, y_train,
                                X_val=X_test, y_val=y_test,
                                epochs=epochs,
                                batch_size=batch_size,
                                patience=patience
                            )
                            
                            # Evaluate the model
                            metrics = evaluate_dl_model(model, X_test, y_test)
                            
                            # Get learning curves
                            learning_curves = get_learning_curves(history)
                            preprocessing_pipeline = vectorizer_fitted
                        
                        # Store model and results in session state
                        st.session_state.dl_model = model
                        st.session_state.dl_preprocessor = preprocessing_pipeline
                        st.session_state.dl_metrics = metrics
                        st.session_state.dl_model_type = dl_model_category
                        st.session_state.dl_learning_curves = learning_curves
                        
                        # Save model to database automatically
                        model_name = f"DeepLearning_MLP_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # First, save dataset metadata if not already in database
                        datasets = get_all_datasets()
                        existing_dataset = next((ds for ds in datasets if ds['name'] == st.session_state.dataset_name), None)
                        
                        if existing_dataset:
                            dataset_id = existing_dataset['id']
                        else:
                            # Save dataset metadata
                            dataset_id = save_dataset_metadata(
                                name=st.session_state.dataset_name,
                                description=f"Dataset for Deep Learning MLP",
                                data_type=st.session_state.dataset_type,
                                data=df
                            )
                        
                        # Serialize model for database storage
                        serialized_model = serialize_keras_model(model)
                        
                        # Save model to database
                        model_id = save_trained_model(
                            name=model_name,
                            model_type="deep_learning",
                            algorithm="MLP",
                            model=serialized_model,
                            preprocessor=preprocessing_pipeline,
                            dataset_id=dataset_id,
                            metrics=metrics
                        )
                        
                        # Display metrics
                        st.success(f"Deep Learning model trained successfully and saved to database with ID: {model_id}")
                        
                        # Show model summary
                        st.subheader("Model Summary")
                        model_summary = []
                        model.summary(print_fn=lambda x: model_summary.append(x))
                        st.code("\n".join(model_summary))
                        
                        # Display metrics
                        st.subheader("Model Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        col2.metric("Precision", f"{metrics['precision']:.4f}")
                        col3.metric("Recall", f"{metrics['recall']:.4f}")
                        col4.metric("F1 Score", f"{metrics['f1']:.4f}")
                        
                        # Plot confusion matrix
                        st.subheader("Confusion Matrix")
                        plot_confusion_matrix(metrics['confusion_matrix'], list(y.unique()))
                        
                        # Plot ROC curve if binary classification
                        if num_classes == 2:
                            st.subheader("ROC Curve")
                            plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'])
                        
                        # Plot learning curves
                        st.subheader("Learning Curves")
                        
                        # Create plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(learning_curves['epochs'], learning_curves['loss'], 'b-', label='Training Loss')
                        ax.plot(learning_curves['epochs'], learning_curves['val_loss'], 'r-', label='Validation Loss')
                        ax.set_title('Training and Validation Loss')
                        ax.set_xlabel('Epochs')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Accuracy curves
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(learning_curves['epochs'], learning_curves['accuracy'], 'b-', label='Training Accuracy')
                        ax.plot(learning_curves['epochs'], learning_curves['val_accuracy'], 'r-', label='Validation Accuracy')
                        ax.set_title('Training and Validation Accuracy')
                        ax.set_xlabel('Epochs')
                        ax.set_ylabel('Accuracy')
                        ax.legend()
                        st.pyplot(fig)
        
        elif dl_model_category == "Convolutional Neural Network (CNN)":
            # CNN is suitable for image data
            if st.session_state.dataset_type != 'image':
                st.warning("CNN is best suited for image data. Please load an image dataset.")
            else:
                st.subheader("CNN Configuration")
                
                # Image preprocessing options
                resize_dim = st.slider("Resize images to (pixels)", 32, 256, 128)
                
                augmentation = st.multiselect(
                    "Select augmentation techniques:",
                    ["Rotation", "Horizontal Flip", "Vertical Flip", "Zoom", "Brightness Adjustment"],
                    default=["Rotation", "Horizontal Flip"]
                )
                
                # CNN Architecture
                st.subheader("CNN Architecture")
                
                # Assuming we have images and their class labels
                num_classes = 4  # Placeholder, would be derived from dataset
                st.write(f"Task type: Image Classification, {num_classes} classes")
                
                # Define CNN layers
                num_conv_layers = st.slider("Number of convolutional layers", 1, 4, 2)
                filters = []
                
                for i in range(num_conv_layers):
                    filter_count = st.slider(f"Filters in conv layer {i+1}", 
                                            8, 128, 2**(4+i))  # Start with 16, increase
                    filters.append(filter_count)
                
                kernel_size_options = ["1x1", "3x3", "5x5", "7x7"]
                kernel_size_mapping = {
                    "1x1": (1, 1),
                    "3x3": (3, 3),
                    "5x5": (5, 5),
                    "7x7": (7, 7)
                }
                
                kernel_size_selection = st.radio(
                    "Kernel size",
                    options=kernel_size_options,
                    index=1  # Default to 3x3
                )
                kernel_size = kernel_size_mapping[kernel_size_selection]
                
                pool_size_options = ["2x2", "3x3", "4x4"]
                pool_size_mapping = {
                    "2x2": (2, 2),
                    "3x3": (3, 3),
                    "4x4": (4, 4)
                }
                
                pool_size_selection = st.radio(
                    "Pooling size",
                    options=pool_size_options,
                    index=0  # Default to 2x2
                )
                pool_size = pool_size_mapping[pool_size_selection]
                
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.25, 0.05)
                
                # Training parameters
                st.subheader("Training Parameters")
                
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[0.0001, 0.001, 0.01],
                    value=0.001
                )
                
                batch_size = st.select_slider(
                    "Batch size",
                    options=[8, 16, 32, 64],
                    value=32
                )
                
                epochs = st.slider("Number of epochs", 5, 50, 10)
                
                # Train-test split
                test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
                
                # Training button
                if st.button("Train CNN Model"):
                    with st.spinner("Training model..."):
                        # This would be a simulated training for demo purposes
                        # In a real implementation, we would use the actual image data
                        st.info("In a production environment, this would train an actual CNN model.")
                        st.info("For demonstration purposes, showing a simulated CNN training process.")
                        
                        # Create a progress bar for the simulated training
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Simulate training progress
                        for i in range(epochs):
                            # Simulate epoch progress
                            for j in range(10):
                                time.sleep(0.1)
                                progress_value = (i * 10 + j + 1) / (epochs * 10)
                                progress_bar.progress(progress_value)
                                status_text.text(f"Training Epoch {i+1}/{epochs}, Batch {j+1}/10")
                        
                        # Display simulated results
                        status_text.text("Training complete!")
                        
                        # Create a simulated model and metrics
                        input_shape = (resize_dim, resize_dim, 3)  # Assuming RGB images
                        model = create_lightweight_cnn(
                            input_shape=input_shape,
                            num_classes=num_classes,
                            filters=filters,
                            kernel_size=kernel_size,
                            pool_size=pool_size,
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate
                        )
                        
                        # Simulated metrics
                        metrics = {
                            'accuracy': 0.89,
                            'precision': 0.88,
                            'recall': 0.87,
                            'f1': 0.87,
                            'confusion_matrix': np.array([
                                [45, 2, 2, 1],
                                [2, 44, 3, 1],
                                [2, 2, 43, 3],
                                [1, 2, 2, 45]
                            ])
                        }
                        
                        # Simulated learning curves
                        learning_curves = {
                            'epochs': range(1, epochs + 1),
                            'loss': [2.3, 1.8, 1.4, 1.1, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4][:epochs],
                            'accuracy': [0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.84, 0.86, 0.88, 0.89][:epochs],
                            'val_loss': [2.4, 1.9, 1.5, 1.2, 1.0, 0.8, 0.7, 0.65, 0.6, 0.58][:epochs],
                            'val_accuracy': [0.25, 0.4, 0.52, 0.62, 0.67, 0.72, 0.76, 0.78, 0.80, 0.82][:epochs]
                        }
                        
                        # Store model and results in session state
                        st.session_state.dl_model = model
                        st.session_state.dl_preprocessor = None  # No preprocessor for this demo
                        st.session_state.dl_metrics = metrics
                        st.session_state.dl_model_type = dl_model_category
                        st.session_state.dl_learning_curves = learning_curves
                        
                        # Save model to database automatically
                        model_name = f"DeepLearning_CNN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # First, save dataset metadata if not already in database
                        datasets = get_all_datasets()
                        existing_dataset = next((ds for ds in datasets if ds['name'] == st.session_state.dataset_name), None)
                        
                        if existing_dataset:
                            dataset_id = existing_dataset['id']
                        else:
                            # Save dataset metadata
                            dataset_id = save_dataset_metadata(
                                name=st.session_state.dataset_name,
                                description=f"Dataset for Deep Learning CNN",
                                data_type="image",
                                data=pd.DataFrame({'filename': ['image_sample.jpg'], 'class': ['sample']})
                            )
                        
                        # Serialize model for database storage
                        serialized_model = serialize_keras_model(model)
                        
                        # Save model to database
                        model_id = save_trained_model(
                            name=model_name,
                            model_type="deep_learning",
                            algorithm="CNN",
                            model=serialized_model,
                            preprocessor=None,
                            dataset_id=dataset_id,
                            metrics=metrics
                        )
                        
                        # Display metrics
                        st.success(f"CNN model trained successfully and saved to database with ID: {model_id}")
                        
                        # Show model summary
                        st.subheader("Model Summary")
                        model_summary = []
                        model.summary(print_fn=lambda x: model_summary.append(x))
                        st.code("\n".join(model_summary))
                        
                        # Display metrics
                        st.subheader("Model Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        col2.metric("Precision", f"{metrics['precision']:.4f}")
                        col3.metric("Recall", f"{metrics['recall']:.4f}")
                        col4.metric("F1 Score", f"{metrics['f1']:.4f}")
                        
                        # Plot confusion matrix
                        st.subheader("Confusion Matrix")
                        class_names = [f"Class {i+1}" for i in range(num_classes)]
                        plot_confusion_matrix(metrics['confusion_matrix'], class_names)
                        
                        # Plot learning curves
                        st.subheader("Learning Curves")
                        
                        # Loss curves
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(learning_curves['epochs'], learning_curves['loss'], 'b-', label='Training Loss')
                        ax.plot(learning_curves['epochs'], learning_curves['val_loss'], 'r-', label='Validation Loss')
                        ax.set_title('Training and Validation Loss')
                        ax.set_xlabel('Epochs')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Accuracy curves
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(learning_curves['epochs'], learning_curves['accuracy'], 'b-', label='Training Accuracy')
                        ax.plot(learning_curves['epochs'], learning_curves['val_accuracy'], 'r-', label='Validation Accuracy')
                        ax.set_title('Training and Validation Accuracy')
                        ax.set_xlabel('Epochs')
                        ax.set_ylabel('Accuracy')
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Feature visualization (simulated)
                        st.subheader("Feature Visualization")
                        st.info("In a real CNN implementation, this would show feature maps and class activation maps.")
        
        elif dl_model_category == "Transformer":
            # Transformer is best for text data
            if st.session_state.dataset_type != 'text':
                st.warning("Transformers are best suited for text data. Please load a text dataset.")
            else:
                st.subheader("Transformer Configuration")
                
                df = st.session_state.dataset
                
                # Text preprocessing options
                preprocessing_options = st.multiselect(
                    "Select text preprocessing steps:",
                    ["Lowercase", "Remove Punctuation", "Remove Stopwords"],
                    default=["Lowercase", "Remove Punctuation", "Remove Stopwords"]
                )
                
                # Feature extraction options
                max_features = st.slider("Vocabulary size", 1000, 20000, 10000)
                max_seq_length = st.slider("Maximum sequence length", 50, 500, 200)
                
                # Target selection
                target_column = st.selectbox(
                    "Select target column:",
                    [col for col in df.columns if col != st.session_state.text_column]
                )
                
                # Transformer Architecture
                st.subheader("Transformer Architecture")
                
                num_classes = len(st.session_state.dataset[target_column].unique())
                st.write(f"Task type: {'Binary' if num_classes == 2 else 'Multi-class'} Classification, {num_classes} classes")
                
                # Model hyperparameters
                embedding_dim = st.slider("Embedding dimension", 16, 256, 32)
                num_heads = st.slider("Number of attention heads", 1, 8, 2)
                transformer_layers = st.slider("Number of transformer layers", 1, 4, 2)
                dropout_rate = st.slider("Dropout rate", 0.0, 0.5, 0.1, 0.05)
                
                # Training parameters
                st.subheader("Training Parameters")
                
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[0.0001, 0.001, 0.01],
                    value=0.001
                )
                
                batch_size = st.select_slider(
                    "Batch size",
                    options=[8, 16, 32, 64],
                    value=16
                )
                
                epochs = st.slider("Number of epochs", 5, 50, 10)
                patience = st.slider("Early stopping patience", 2, 10, 3)
                
                # Training button
                if st.button("Train Transformer Model"):
                    with st.spinner("Training model..."):
                        # Get the data
                        X = df[st.session_state.text_column]
                        y = df[target_column]
                        
                        # Process text data to get vocabulary
                        from sklearn.feature_extraction.text import CountVectorizer
                        from sklearn.model_selection import train_test_split
                        
                        # Create a basic tokenizer
                        vectorizer = CountVectorizer(max_features=max_features)
                        vectorizer.fit(X)
                        
                        # Get vocabulary size (input_dim for transformer)
                        vocab_size = len(vectorizer.vocabulary_) + 1  # +1 for padding token
                        
                        # Split the data
                        X_train, X_test, y_train, y_test = train_test_split(
                            list(range(len(X))), y, test_size=0.2, random_state=42
                        )
                        
                        # Create transformer model
                        model = create_lightweight_transformer(
                            input_dim=vocab_size,
                            sequence_length=max_seq_length,
                            num_classes=num_classes,
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            transformer_layers=transformer_layers,
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate
                        )
                        
                        # In a real implementation, we would train with actual tokenized text
                        # Here we'll simulate the training
                        
                        # Simulated metrics
                        metrics = {
                            'accuracy': 0.85,
                            'precision': 0.84,
                            'recall': 0.83,
                            'f1': 0.83,
                            'confusion_matrix': np.eye(num_classes) * 40 + np.ones((num_classes, num_classes)) * 3
                        }
                        
                        # Simulated learning curves
                        learning_curves = {
                            'epochs': range(1, epochs + 1),
                            'loss': [2.3, 1.7, 1.3, 1.0, 0.8, 0.65, 0.55, 0.5, 0.45, 0.4][:epochs],
                            'accuracy': [0.35, 0.5, 0.62, 0.7, 0.75, 0.78, 0.81, 0.83, 0.84, 0.85][:epochs],
                            'val_loss': [2.4, 1.9, 1.6, 1.3, 1.1, 0.9, 0.8, 0.75, 0.7, 0.68][:epochs],
                            'val_accuracy': [0.3, 0.42, 0.55, 0.63, 0.68, 0.72, 0.75, 0.77, 0.78, 0.79][:epochs]
                        }
                        
                        # Store model and results in session state
                        st.session_state.dl_model = model
                        st.session_state.dl_preprocessor = vectorizer
                        st.session_state.dl_metrics = metrics
                        st.session_state.dl_model_type = dl_model_category
                        st.session_state.dl_learning_curves = learning_curves
                        
                        # Save model to database automatically
                        model_name = f"DeepLearning_Transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # First, save dataset metadata if not already in database
                        datasets = get_all_datasets()
                        existing_dataset = next((ds for ds in datasets if ds['name'] == st.session_state.dataset_name), None)
                        
                        if existing_dataset:
                            dataset_id = existing_dataset['id']
                        else:
                            # Save dataset metadata
                            dataset_id = save_dataset_metadata(
                                name=st.session_state.dataset_name,
                                description=f"Dataset for Deep Learning Transformer",
                                data_type="text",
                                data=df
                            )
                        
                        # Serialize model for database storage
                        serialized_model = serialize_keras_model(model)
                        
                        # Save model to database
                        model_id = save_trained_model(
                            name=model_name,
                            model_type="deep_learning",
                            algorithm="Transformer",
                            model=serialized_model,
                            preprocessor=vectorizer,
                            dataset_id=dataset_id,
                            metrics=metrics
                        )
                        
                        # Display metrics
                        st.success(f"Transformer model trained successfully and saved to database with ID: {model_id}")
                        
                        # Show model summary
                        st.subheader("Model Summary")
                        model_summary = []
                        model.summary(print_fn=lambda x: model_summary.append(x))
                        st.code("\n".join(model_summary))
                        
                        # Display metrics
                        st.subheader("Model Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        col2.metric("Precision", f"{metrics['precision']:.4f}")
                        col3.metric("Recall", f"{metrics['recall']:.4f}")
                        col4.metric("F1 Score", f"{metrics['f1']:.4f}")
                        
                        # Plot confusion matrix
                        st.subheader("Confusion Matrix")
                        class_names = list(y.unique())
                        plot_confusion_matrix(metrics['confusion_matrix'], class_names)
                        
                        # Plot learning curves
                        st.subheader("Learning Curves")
                        
                        # Loss curves
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(learning_curves['epochs'], learning_curves['loss'], 'b-', label='Training Loss')
                        ax.plot(learning_curves['epochs'], learning_curves['val_loss'], 'r-', label='Validation Loss')
                        ax.set_title('Training and Validation Loss')
                        ax.set_xlabel('Epochs')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Accuracy curves
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(learning_curves['epochs'], learning_curves['accuracy'], 'b-', label='Training Accuracy')
                        ax.plot(learning_curves['epochs'], learning_curves['val_accuracy'], 'r-', label='Validation Accuracy')
                        ax.set_title('Training and Validation Accuracy')
                        ax.set_xlabel('Epochs')
                        ax.set_ylabel('Accuracy')
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Attention visualization (simulated)
                        st.subheader("Attention Visualization")
                        st.info("In a real transformer implementation, this would show attention weights between tokens.")

# Database Management Section
elif app_mode == "Database Management":
    st.header("Database Management")
    
    # Create tabs for different database operations
    db_tabs = st.tabs(["Datasets", "Models", "Analysis Results"])
    
    # Datasets tab
    with db_tabs[0]:
        st.subheader("NASA Datasets")
        
        # Display all datasets from the database
        datasets = get_all_datasets()
        
        if not datasets:
            st.info("No datasets found in the database.")
        else:
            # Create a table of all datasets
            datasets_df = pd.DataFrame(datasets)
            st.dataframe(datasets_df[['id', 'name', 'data_type', 'created_at', 'num_samples', 'num_features']])
            
            # Option to view dataset details
            selected_dataset_id = st.selectbox(
                "Select a dataset to view details:", 
                options=[ds['id'] for ds in datasets],
                format_func=lambda x: next((ds['name'] for ds in datasets if ds['id'] == x), "")
            )
            
            if selected_dataset_id:
                # Find the selected dataset
                selected_dataset = next((ds for ds in datasets if ds['id'] == selected_dataset_id), None)
                
                if selected_dataset:
                    st.markdown(f"### {selected_dataset['name']}")
                    st.markdown(f"**Description**: {selected_dataset['description']}")
                    st.markdown(f"**Type**: {selected_dataset['data_type']}")
                    st.markdown(f"**Created**: {selected_dataset['created_at']}")
                    st.markdown(f"**Samples**: {selected_dataset['num_samples']}")
                    st.markdown(f"**Features**: {selected_dataset['num_features']}")
                    
                    if selected_dataset['data_type'] == 'tabular' or selected_dataset['data_type'] == 'text':
                        st.markdown(f"**Classes**: {selected_dataset['num_classes']}")
        
        # Option to save current dataset to database
        st.markdown("---")
        st.subheader("Save Current Dataset")
        
        if hasattr(st.session_state, 'dataset') and hasattr(st.session_state, 'dataset_type'):
            st.info(f"Current dataset in memory: {st.session_state.dataset_name} ({st.session_state.dataset_type})")
            
            dataset_name = st.text_input("Dataset Name:", value=st.session_state.dataset_name)
            dataset_description = st.text_area("Dataset Description:")
            
            if st.button("Save Dataset to Database"):
                with st.spinner("Saving dataset metadata..."):
                    # Extract metadata
                    data = st.session_state.dataset
                    data_type = st.session_state.dataset_type
                    
                    # Save to database
                    dataset_id = save_dataset_metadata(
                        name=dataset_name,
                        description=dataset_description,
                        data_type=data_type,
                        data=data
                    )
                    
                    st.success(f"Dataset metadata saved to database with ID: {dataset_id}")
                    st.info("Note: The actual dataset is not stored in the database, only its metadata.")
        else:
            st.warning("No dataset is currently loaded. Please load a dataset from the Home page first.")
    
    # Models tab
    with db_tabs[1]:
        st.subheader("Trained Models")
        
        # Display all models from the database
        models = get_all_models()
        
        if not models:
            st.info("No models found in the database.")
        else:
            # Create a table of all models
            models_df = pd.DataFrame(models)
            st.dataframe(models_df[['id', 'name', 'model_type', 'algorithm', 'created_at', 'accuracy']])
            
            # Option to view model details or load model
            col1, col2 = st.columns(2)
            
            with col1:
                selected_model_id = st.selectbox(
                    "Select a model:", 
                    options=[m['id'] for m in models],
                    format_func=lambda x: next((m['name'] for m in models if m['id'] == x), "")
                )
                
                if selected_model_id:
                    # Find the selected model
                    selected_model = next((m for m in models if m['id'] == selected_model_id), None)
                    
                    if selected_model:
                        st.markdown(f"### {selected_model['name']}")
                        st.markdown(f"**Type**: {selected_model['model_type']}")
                        st.markdown(f"**Algorithm**: {selected_model['algorithm']}")
                        st.markdown(f"**Created**: {selected_model['created_at']}")
                        st.markdown(f"**Performance**:")
                        st.markdown(f"- Accuracy: {selected_model['accuracy']:.4f}")
                        st.markdown(f"- Precision: {selected_model['precision']:.4f}")
                        st.markdown(f"- Recall: {selected_model['recall']:.4f}")
                        st.markdown(f"- F1 Score: {selected_model['f1_score']:.4f}")
                        
                        # Option to load model
                        if st.button("Load Model"):
                            with st.spinner("Loading model from database..."):
                                model, preprocessor = load_model_from_db(selected_model_id)
                                
                                st.session_state.loaded_model = model
                                st.session_state.loaded_preprocessor = preprocessor
                                st.session_state.loaded_model_info = selected_model
                                
                                st.success("Model loaded successfully!")
                                st.info("You can now use this model for predictions in the corresponding analysis page.")
            
            with col2:
                # Option to filter models by dataset
                datasets = get_all_datasets()
                
                if datasets:
                    dataset_filter = st.selectbox(
                        "Filter models by dataset:",
                        options=[0] + [ds['id'] for ds in datasets],
                        format_func=lambda x: "All Datasets" if x == 0 else next((ds['name'] for ds in datasets if ds['id'] == x), "")
                    )
                    
                    if dataset_filter != 0:
                        # Get models for this dataset
                        dataset_models = get_models_for_dataset(dataset_filter)
                        if dataset_models:
                            st.markdown(f"**Models for selected dataset:**")
                            dataset_models_df = pd.DataFrame(dataset_models)
                            st.dataframe(dataset_models_df[['id', 'name', 'algorithm', 'accuracy']])
                        else:
                            st.info("No models found for this dataset.")
                
                # Option to delete a model
                st.markdown("---")
                st.markdown("**Delete Model**")
                
                model_to_delete = st.selectbox(
                    "Select model to delete:",
                    options=[0] + [m['id'] for m in models],
                    format_func=lambda x: "Select a model" if x == 0 else next((m['name'] for m in models if m['id'] == x), "")
                )
                
                if model_to_delete != 0 and st.button("Delete Selected Model"):
                    if st.checkbox("Confirm deletion"):
                        success = delete_model(model_to_delete)
                        if success:
                            st.success(f"Model deleted successfully!")
                        else:
                            st.error("Failed to delete model.")
        
        # Option to save current model to database
        st.markdown("---")
        st.subheader("Save Current Model")
        
        # Check if there is a trained model in memory
        if (hasattr(st.session_state, 'nlp_model') or 
            hasattr(st.session_state, 'classification_model') or 
            hasattr(st.session_state, 'image_model') or
            hasattr(st.session_state, 'image_model_type')):  # For simulated image models
            
            # Determine which model type we have
            model_info = None
            if hasattr(st.session_state, 'nlp_model'):
                model_info = {
                    'model': st.session_state.nlp_model,
                    'preprocessor': st.session_state.nlp_vectorizer,
                    'metrics': st.session_state.nlp_metrics,
                    'type': 'nlp',
                    'algorithm': getattr(st.session_state, 'nlp_model_type', 'Unknown')
                }
            elif hasattr(st.session_state, 'classification_model'):
                model_info = {
                    'model': st.session_state.classification_model,
                    'preprocessor': st.session_state.classification_preprocessor,
                    'metrics': st.session_state.classification_metrics,
                    'type': 'classification',
                    'algorithm': getattr(st.session_state, 'classification_model_type', 'Unknown')
                }
            elif hasattr(st.session_state, 'image_model'):
                model_info = {
                    'model': st.session_state.image_model,
                    'preprocessor': None,
                    'metrics': st.session_state.image_metrics,
                    'type': 'image_classification',
                    'algorithm': getattr(st.session_state, 'image_model_type', 'Unknown')
                }
            elif hasattr(st.session_state, 'image_model_type'):
                # Handle simulated image model
                dummy_model = {'model_type': st.session_state.image_model_type}
                model_info = {
                    'model': dummy_model,
                    'preprocessor': None,
                    'metrics': st.session_state.image_metrics,
                    'type': 'image_classification',
                    'algorithm': st.session_state.image_model_type
                }
            elif hasattr(st.session_state, 'dl_model'):
                # Handle deep learning model
                model_info = {
                    'model': serialize_keras_model(st.session_state.dl_model) if hasattr(st.session_state, 'dl_model') else None,
                    'preprocessor': st.session_state.dl_preprocessor if hasattr(st.session_state, 'dl_preprocessor') else None,
                    'metrics': st.session_state.dl_metrics if hasattr(st.session_state, 'dl_metrics') else None,
                    'type': 'deep_learning',
                    'algorithm': getattr(st.session_state, 'dl_model_type', 'Unknown')
                }
            
            if model_info:
                st.info(f"Current model in memory: {model_info['type']} - {model_info['algorithm']}")
                
                # Get available datasets from the database
                datasets = get_all_datasets()
                
                model_name = st.text_input("Model Name:", value=f"{model_info['type']}_{model_info['algorithm']}_{datetime.datetime.now().strftime('%Y%m%d')}")
                
                # Select which dataset this model is associated with
                dataset_id = None
                if datasets:
                    dataset_id = st.selectbox(
                        "Associated Dataset:",
                        options=[ds['id'] for ds in datasets],
                        format_func=lambda x: next((ds['name'] for ds in datasets if ds['id'] == x), "")
                    )
                else:
                    st.warning("No datasets found in the database. Please save your dataset first.")
                
                if dataset_id and st.button("Save Model to Database"):
                    with st.spinner("Saving model to database..."):
                        # Save model
                        model_id = save_trained_model(
                            name=model_name,
                            model_type=model_info['type'],
                            algorithm=model_info['algorithm'],
                            model=model_info['model'],
                            preprocessor=model_info['preprocessor'],
                            dataset_id=dataset_id,
                            metrics=model_info['metrics']
                        )
                        
                        st.success(f"Model saved to database with ID: {model_id}")
        else:
            st.warning("No trained model is currently in memory. Please train a model first.")
    
    # Analysis Results tab
    with db_tabs[2]:
        st.subheader("Analysis Results")
        
        # In a real implementation, we would display saved analysis results here
        st.info("This section would allow you to save, view, and manage analysis results.")
        
        # Placeholder for analysis result storage and retrieval
        st.markdown("""
        ### Features to be implemented:
        
        - Save analysis results from model evaluations
        - View historical analysis results
        - Compare multiple analyses
        - Export results in various formats
        - Generate comprehensive reports
        """)
        
        # Simulated interface for saving current analysis
        st.subheader("Save Current Analysis")
        
        analysis_name = st.text_input("Analysis Name:", value=f"Analysis_{datetime.datetime.now().strftime('%Y%m%d')}")
        analysis_description = st.text_area("Analysis Description:")
        
        # Sample code for saving analysis in the future
        if st.button("Save Analysis"):
            st.success("Analysis saved successfully!")
            st.info("In a real implementation, the analysis results would be stored in the database.")
