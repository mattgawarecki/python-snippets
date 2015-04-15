import numpy as np
import random
import math
import pdb

def predict(weights, input):
    return sum(w * j for j, w in zip(input, weights))
    
def predict_all(weights, inputs):
    return [predict(weights, i) for i in inputs]
    
def normalize_features(inputs):
    features = list(zip(*inputs))
    feature_norm_params = [(max(f) - min(f), sum(f) / len(f)) for f in features]
    new_inputs = [[(f - n[1]) / n[0] for f, n in zip(i, feature_norm_params)] for i in inputs]
    return new_inputs

def batch_gd(training_data, iteration_count=10, learning_rate=0.01, debug=False):
    inputs = [[1] + i for i in normalize_features([d[:-1] for d in training_data])]
    outputs = [d[-1] for d in training_data]
    weights = [random.random() for _ in inputs[0]]
    
    data_count = len(inputs)
    
    max_errors = []
    for _ in range(iteration_count):
        combined = list(zip(inputs, outputs))
        random.shuffle(combined)
        regression_inputs, regression_outputs = zip(*combined)
        
        predictions = predict_all(weights, regression_inputs)
        errors = [p - o for p, o in zip(predictions, regression_outputs)]
        for w_i in range(len(weights)):
            weights[w_i] -= learning_rate * sum(e * i[w_i] for i, e in zip(regression_inputs, errors)) / data_count
        
        max_error = max(errors) ** 2
        max_errors.append(max_error)
        
    if debug:
        print(max_errors)
        
        import matplotlib.pyplot as plt
        plt.plot(list(range(iteration_count)), max_errors)
        plt.show()
    
    return weights
    
def stochastic_gd(training_data, iteration_count=10, learning_rate=0.01, debug=False):
    inputs = [[1] + i for i in normalize_features([d[:-1] for d in training_data])]
    outputs = [d[-1] for d in training_data]
    weights = [random.random() for _ in inputs[0]]
    
    data_count = len(inputs)
    
    max_errors = []
    for _ in range(iteration_count):
        combined = list(zip(inputs, outputs))
        random.shuffle(combined)
        regression_inputs, regression_outputs = zip(*combined)
        
        max_error = 0
        for i, o in zip(regression_inputs, regression_outputs):
            prediction = predict(weights, i)
            error = prediction - o
            weights = [w - learning_rate * error * j for j, w in zip(i, weights)]
            if abs(error) > abs(max_error):
                max_error = error
        
        max_errors.append(max_error ** 2)
        
    if debug:
        print(max_errors)
    
        import matplotlib.pyplot as plt
        plt.plot(list(range(iteration_count)), max_errors)
        plt.show()
    
    return weights
    
def normal_equations(training_data):
    input_matrix = np.matrix([[1] + d[:-1] for d in training_data])
    output_vector = np.matrix([d[-1] for d in training_data]).getT()
    weights = (input_matrix.getT() * input_matrix).getI() * input_matrix.getT() * output_vector
    return weights
