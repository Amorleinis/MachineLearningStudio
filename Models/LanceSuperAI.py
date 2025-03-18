import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate a dataset
def generate_dataset(num_samples, num_features, num_classes):
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(num_classes, size=num_samples)
    return X, y

# Example usage
X, y = generate_dataset(1000, 20, 5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up GPU configuration
def gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPUs available")

# Data Preprocessing
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Model Architecture
def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Batch Training and Learning Rate Schedule
def train_model_with_schedule(model, X_train, y_train, X_test, y_test):
    # Use Adam optimizer with a learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test))
    return model

if __name__ == "__main__":
    # Example usage 
    X, y = generate_dataset(1000, 20, 5)   # Example dataset generation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = preprocess_data(X_train, y_train)   # Preprocess data
    model = create_model(X_train.shape[1], len(np.unique(y_train)))   # Create model
    model = train_model_with_schedule(model, X_train, y_train, X_test, y_test)   # Train model with learning rate schedule
    print(model.summary())   # Print model summary 
# Save the models
    model.save('my_model')

# Load your data
    model = tf.keras.models.load_model('my_model')

# Evaluate your models
    model.evaluate(X_test, y_test)


    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    gpu_config()
    model = create_model(input_shape=X_train.shape[1], num_classes=10)
    trained_model = train_model_with_schedule(model, X_train, y_train, X_test, y_test)
    # Save the trained model
    trained_model.save('trained_model.h5')
    # Load the trained model
    trained_model = tf.keras.models.load_model('trained_model.h5')
    # Evaluate the trained model
    trained_model.evaluate(X_test, y_test)
    # Save the trained model
    trained_model.save('trained_model.h5')
    # Generate several experiments to teset the hyperparameter tuning
    X, y = generate_dataset(1000, 20, 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocess_data(X_train, y_train)
    model = create_model(input_shape=X_train.shape[1], num_classes=10)
    trained_model = train_model_with_schedule(model, X_train, y_train, X_test, y_test)
    # Save the trained model as LanceSuperAI.h5
    trained_model.save('LanceSuperAI.h5')

















