# Define a generic gate function
def gate(inputs, weights, threshold):
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    return 1 if weighted_sum >= threshold else 0, weights, threshold

# Define specific gate functions
def or_gate(x1, x2):
    return gate([x1, x2], weights=[1, 1], threshold=1)

# Generate a table-like output for the gates
def generate_gate_table(gate_func, gate_name, inputs):
    print(f"\n{gate_name} Gate:")
    print("x1    x2     O/P")
    for x1, x2 in inputs:
        output, weights, threshold = gate_func(x1, x2)
        weights_str = ", ".join(map(str, weights))  # Convert weights to a string
        print(f"{x1}     {x2}       {output}")
    print(f"Selected Weights: {weights_str}\nSelected Threshold: {threshold}")


# Input combinations for testing
input_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Generate tables for AND, OR, and XOR gates
generate_gate_table(or_gate, "OR", input_combinations)