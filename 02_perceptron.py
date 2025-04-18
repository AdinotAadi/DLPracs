import numpy as np

def activation_function(x):
    return 1 if x >= 0 else 0

def predict(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return activation_function(weighted_sum)

def train(X, y, learning_rate=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0
    for epoch in range(epochs):
        for i in range(len(X)):
            y_pred = predict(X[i], weights, bias)
            error = y[i] - y_pred
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
    return weights, bias

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

AND_y = np.array([0, 0, 0, 1])
OR_y = np.array([0, 1, 1, 1])
XOR_y = np.array([0, 1, 1, 0])

and_weights, and_bias = train(X, AND_y)
print("AND Gate:")
for input_data in X:
    print(f"Input: {input_data} -> Output: {predict(input_data, and_weights, and_bias)}")

or_weights, or_bias = train(X, OR_y)
print("\nOR Gate:")
for input_data in X:
    print(f"Input: {input_data} -> Output: {predict(input_data, or_weights, or_bias)}")

xor_weights, xor_bias = train(X, XOR_y)
print("\nXOR Gate (Note: Perceptron cannot model XOR correctly):")
for input_data in X:
    print(f"Input: {input_data} -> Output: {predict(input_data, xor_weights, xor_bias)}")