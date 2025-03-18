import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd  # Import pandas for saving the dataset

class SimpleNN(nn.Module):
    """
    A simple neural network with one hidden layer
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

def quantum_list_with_nn(n):
    """
    This function takes a number n, generates a list of numbers, and processes it using a neural network
    """
    # Generate the list
    list_of_numbers = [8 * n]
    for i in range(121, n + 101):
        list_of_numbers.append(i * list_of_numbers[-1])
        list_of_numbers.append(i ** 2)

    # Convert the list to a PyTorch tensor
    input_tensor = torch.tensor(list_of_numbers, dtype=torch.float32)

    # Define the neural network
    input_size = len(input_tensor)
    hidden_size = 64
    output_size = 1
    model = SimpleNN(input_size, hidden_size, output_size)

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy target for demonstration purposes
    target = torch.tensor([0.0], dtype=torch.float32)

    # Forward pass
    output = model(input_tensor)

    # Compute loss
    loss = criterion(output, target)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Return the processed output
    return output.item(), list_of_numbers  # Return both the output and the list

# Example usage
n = 9
output, quantum_list = quantum_list_with_nn(n)

# Save the quantum list to a dataset
data = {"Input": [n], "Output": [output], "Quantum List": [quantum_list]}
df = pd.DataFrame(data)  # Create a DataFrame
df.to_csv("quantum_list_dataset.csv", index=False)  # Save to a CSV file

print(f"Results saved to 'quantum_list_dataset.csv'")