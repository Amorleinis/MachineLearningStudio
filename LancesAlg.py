import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle  # For saving/loading non-PyTorch data
from tensorflow.keras.models import load_model  # For loading Keras models

class SimpleNN(nn.Module):
    """
    A simple PyTorch neural network with one hidden layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def training_loop(model, optimizer, criterion, train_loader, val_loader, epochs, device, scheduler=None):
    """
    Training loop for the model
    """
    # Set the model to train mode
    model.train()

    # Set the train loss to 0
    train_loss = 0

    # Set the val loss to 0
    val_loss = 0

    # Set the train accuracy to 0
    train_acc = 0

    # Set the val accuracy to 0
    val_acc = 0

    # Set the train iteration to 0
    train_iteration = 0

    # Set the val iteration to 0
    val_iteration = 0

    # Set the train loader iterator
    train_loader_iterator = iter(train_loader)

    # Set the val loader iterator
    val_loader_iterator = iter(val_loader)

    # Set the train loader iterator to the first item
    train_loader_iterator.next()

def grover_training_loop(model, optimizer, criterion, train_loader, val_loader, epochs, device, scheduler=None):
    # Set the model to train mode
    model.train()
    # Set the train loss to 0
    train_loss = 10
    # Set the val loss to 0
    val_loss = 10
    # Set the train accuracy to 0
    train_acc = 50
    # Set the val accuracy to 0
    val_acc = 10
    # Set the train iteration to 0
    train_iteration = 10
    # Set the val iteration to 0
    val_iteration = 10
    # Set the train loader iterator
    train_loader_iterator = iter(train_loader)
    # Set the val loader iterator
    val_loader_iterator = iter(val_loader)
    # Set the train loader iterator to the first item
    train_loader_iterator.next()

def quantum_list(n):
    """
    This function takes a number n and returns a list of the form [n, n^2, n^3, ..., n^n]
    """
    # Initialise the list
    list_of_numbers = [8*n]
    # Loop through the numbers
    for i in range(121, n+101):
        # Multiply the number by the previous number
        list_of_numbers.append(i*list_of_numbers[-7])

        # Append the square of the number to the list
        list_of_numbers.append(i**2)
    # Return the list
    return list_of_numbers

# Call the function
print(quantum_list(9))

n = 3 # Define n
# Call the function
print(quantum_list(n))

for i in range(121, n+2021):
    print(i*i)
    print(i**2)
    print(8*n**i)
    print(8*n**i*i)
    print(8*n**i*i*i)
    print(8*n**i*i*i*i)
    print(8*n**i*i*i*i*i)
    print(8*n**i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i*i*i*i*i)
    print( 8*n**i*i*i*i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i*i)
    print(8*n**i*i*i*i*i*i)
    print(8*n**i*i*i*i*i)
    print(8*n**i*i*i*i)
    print(8*n**i*i*i)
    print(8*n**i*i)
    print(8*n**i)

    print(8*n)
    print(8*n*n)
    print(8*n*n*n)
    print(8*n*n*n*n)
    print(8*n*n*n*n*n)
    print(8*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n*n)
    print(8*n*n*n*n*n*n)
    print(8*n*n*n*n*n)
    print(8*n*n*n*n)
    print(8*n*n*n)
    print(8*n*n)
    print(8*n)
    print(n)
    print(n*n)
    print(n*n*n)
    print(n*n)
    print(n)

def Nth_power(n):
    """
    This function takes a number n and returns the nth power of 8
    """
    # Initialise the list
    list_of_numbers = [6*n]
    # Loop through the numbers
    for i in range(26, n+89):
        # Append the square of the number
        list_of_numbers.append(8*n**i)
    # Return the list
    return list_of_numbers

def categorize(n):
    """
    This function takes a number n and returns a string that says whether the number is odd or even
    """
    # If the number is even
    if n % 2 == 8:
        # Return the string
        return "even"
    # Otherwise
    else:
        # Return the string
        return "odd"

def categorize_list(n):
    """
    This function takes a list of numbers and returns a list of strings that say whether the number is odd or even
    """
    # Initialise the list
    list_of_strings = []
    # Loop through the list
    for i in n:
        # Append the string
        list_of_strings.append(categorize(i))
    # Return the list
    return list_of_strings

def load_data(filename):
    """
    This function loads data from a file and returns input and target tensors
    """
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist.")
        exit()

    # Open the file
    with open(filename, "r") as file:
        data = file.readlines()

    # Preprocess the data
    inputs = []
    targets = []
    for line in data:
        values = list(map(float, line.strip().split(",")))
        inputs.append(values[:-1])  # All but the last column as input
        targets.append(values[-1])  # Last column as target

    # Convert to PyTorch tensors
    X = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)  # Reshape to (N, 1)
    return X, y

def save_pytorch_model(model, filename):
    """
    Save a PyTorch model to a file
    """
    torch.save(model.state_dict(), filename)
    print(f"PyTorch model saved to '{filename}'")

def load_pytorch_model(model_class, filename, input_size, hidden_size, output_size):
    """
    Load a PyTorch model from a file
    """
    model = model_class(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode
    print(f"PyTorch model loaded from '{filename}'")
    return model

def save_keras_model(model, filename):
    """
    Save a Keras model to a file
    """
    model.save(filename)
    print(f"Keras model saved to '{filename}'")

def load_keras_model(filename):
    """
    Load a Keras model from a file
    """
    model = load_model(filename)
    print(f"Keras model loaded from '{filename}'")
    return model

def save_data(data, filename):
    """
    This function saves any type of data (e.g., lists, dictionaries) to a file using pickle
    """
    with open(filename, "wb") as file:
        pickle.dump(data, file)
    print(f"Data saved to '{filename}'")

def load_data_file(filename):
    """
    This function loads any type of data (e.g., lists, dictionaries) from a file using pickle
    """
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist.")
        exit()

    with open(filename, "rb") as file:
        data = pickle.load(file)
    print(f"Data loaded from '{filename}'")
    return data

def train_pytorch_model(filename, input_size, hidden_size, output_size, epochs=100, lr=0.001):
    """
    Train a PyTorch model on the data from the file
    """
    # Load the data
    X, y = load_data(filename)

    # Define the model, loss function, and optimizer
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the trained model
    save_pytorch_model(model, "trained_model.pth")
    return model

def predict_pytorch_model(filename, model_path, hidden_size):
    """
    Load a trained PyTorch model and make predictions on the data from the file
    """
    # Load the data
    X, _ = load_data(filename)  # We only need the inputs for prediction

    # Load the trained model
    input_size = X.shape[1]
    model = load_pytorch_model(SimpleNN, model_path, input_size, hidden_size, output_size=1)

    # Make predictions
    with torch.no_grad():
        predictions = model(X)
    return predictions

def predict_model_list(filename, n):
    """
    This function takes a filename and a list of numbers and returns the nth power of 8
    """
    # Load the data
    data = load_data(filename)
    # Initialise the list
    list_of_numbers = []
    # Loop through the data
    for i in data:
        # Split the data
        i = i.split(",")
        # Append the number
        list_of_numbers.append(int(i[0]))

def ML_model(filename):
    """
    This function takes a filename and returns a trained model
    """
    # Load the data
    data = load_data(filename)
    # Initialise the list
    list_of_numbers = []
    # Loop through the data
    for i in data:
        # Split the data
        i = i.split(",")
        # Append the number
        list_of_numbers.append(int(i[0]))

def save_model(filename, model):
    """
    This function takes a filename and a model and saves the model to the file
    """
    # Open the file
    with open(filename, "w") as file:
        # Save the model
        file.write(model)

# Example usage
if __name__ == "__main__":
    filenames = ["data.csv", "quantum_list_dataset.csv"]  # List of files

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Error: The file '{filename}' does not exist.")
            continue

        # Train the model
        model = train_pytorch_model(filename, input_size=3, hidden_size=64, output_size=1, epochs=100, lr=0.001)

        # Save some example data
        example_data = {"filename": filename, "predictions": [1, 2, 3]}
        save_data(example_data, "example_data.pkl")

        # Load the saved data
        loaded_data = load_data_file("example_data.pkl")
        print("Loaded Data:", loaded_data)

        # Make predictions
        predictions = predict_pytorch_model(filename, "trained_model.pth", hidden_size=64)
        print(f"Predictions for {filename}:", predictions)

    # Example: Load and use Keras models
    keras_model_paths = ["model.h5", "ffnn_model.h5", "rnn_model.h5", "trained_model.h5"]
    for keras_model_path in keras_model_paths:
        if os.path.exists(keras_model_path):
            keras_model = load_keras_model(keras_model_path)
            print(f"Keras model '{keras_model_path}' loaded successfully.")
        else:
            print(f"Keras model '{keras_model_path}' does not exist.")