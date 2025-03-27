"""
Steps:
1. Define Encoder & Decoder (dimension, ...)
2. Forward propagation
3. Back propagation (compute error at decoder & encoder)
4. Train
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def MSE(x, y):
    return np.mean((x-y)**2)

def sigmoid_derivative(x):
    return x * (1-x)

class MyAutoencoder:
    def __init__(self, input_dim=4, hidden_dim=2, output_dim=4, learning_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

    def randomize_data(self, n_samples, n_features):
        np.random.seed(42)
        # Input data (rand: Uniform distribution)
        self.X = np.random.rand(n_samples, n_features)
        # Weights & biases for encoder (randn: standard normal (Gaussian) distribution)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.01 
        self.b1 = np.zeros((1,self.hidden_dim)) 
        # Weights & biases for decoder
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.01 
        self.b2 = np.zeros((1,self.output_dim))

    

    def train(self, epochs):
        for epoch in range(epochs):
            # Forward pass
            # Apply non-linear activation function for a linear transformation inside
            hidden = sigmoid((self.X @ self.W1) + self.b1)
            # output
            Y = sigmoid((hidden @ self.W2) + self.b2)

            # Compute loss (MSE)
            loss = MSE(self.X, Y)

            #  Back propagation
            output_error = self.X - Y # reconstruct error
            output_delta = output_error * sigmoid_derivative(Y)

            hidden_error = output_delta.dot(self.W2.T)  # Propagate error back
            hidden_delta = hidden_error * sigmoid_derivative(hidden)  # Gradient at encoder

            # Update weights using gradient descent
            self.W2 += hidden.T.dot(output_delta) * self.learning_rate
            self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

            self.W1 += self.X.T.dot(hidden_delta) * self.learning_rate
            self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

             # Print loss
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
            if epoch == epochs-1:
                print(f"Epoch {epoch}, Final Loss: {loss:.4f}")
                

myAutoencoder = MyAutoencoder(4,2,4,0.001)
myAutoencoder.randomize_data(1000,4)
myAutoencoder.train(1000)