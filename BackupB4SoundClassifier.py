# Main.py after the required edits
Author: Lance Brady

# Required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

# Load the dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=420, stratify=y)

# Train a machine learning model (Random Forest)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=420)
rf_clf.fit(X_train, y_train)

# Predict using the trained ML model and Evaluate the ML model
rf_predictions = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Build and train a deep learning model
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(3, activation='softmax'),
])

dl_model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=2, shuffle=True)

# Evaluate the DL model
dl_loss, dl_accuracy = dl_model.evaluate(X_test, y_test, batch_size=32)
print(f"Deep Learning Model Accuracy: {dl_accuracy}")

# Save the DL model predictions and model
dl_predictions = dl_model.predict(X_test)
np.save('dl_predictions.npy', dl_predictions)
joblib.dump(dl_model, 'dl_model.joblib')

# Define save all function with proper handling for scikit-learn models
def save_all(model, X_test, y_test):
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        joblib.dump(y_pred_proba, 'y_pred_proba.joblib')
    y_pred = model.predict(X_test)
    joblib.dump(y_pred, 'y_pred.joblib')
    joblib.dump(model, 'model.joblib')
    joblib.dump(X_test, 'X_test.joblib')
    joblib.dump(y_test, 'y_test.joblib')

# Save best models and predictions
save_all(rf_clf, X_test, y_test)
# end of main.py

