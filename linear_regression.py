import numpy as np
import random
import math
import pdb

class RegressionModel:
    def __init__(self, *, predict_fn, error_fn, learning_rate=0.1, debug=False):
        self.debug = debug
        self.learning_rate = learning_rate
        self.error_fn = error_fn
        self.predict_fn = predict_fn
        
    def _normalize_features(self, input_data):
        normalized_feature_data = []
        features = list(zip(*input_data))
        for feature_data in features:
            spread = max(feature_data) - min(feature_data)
            mean = sum(feature_data) / len(feature_data)
            normalized_feature_data.append([(example - mean) / spread for example in feature_data])
        return [list(i) for i in zip(*normalized_feature_data)]
    
    def _run_iteration(self, data, *, weights=None):
        inputs = [[1] + d[:-1] for d in data]
        outputs = [d[-1] for d in data]
        weights = weights or [random.random() for _ in inputs[0]]
        
        predictions = []
        for input, output in zip(inputs, outputs):
            prediction = self.predict_fn(weights, input)
            error = self.error_fn(prediction, output)
            weights = [w - self.learning_rate * error * i for i, w in zip(input, weights)]
        
        return weights
    
    def model(self, data, *, iterations=20):
        raw_inputs = [d[:-1] for d in data]
        normalized_inputs = self._normalize_features(raw_inputs)
        outputs = [d[-1] for d in data]
        
        feature_weights = None
        for i in range(iterations):
            if self.debug:
                print('Iteration #{}'.format(i))
            
            combined = [i + [o] for i, o in zip(normalized_inputs, outputs)]
            random.shuffle(combined)
            
            feature_weights = self._run_iteration(combined, weights=feature_weights)
           
        if self.debug:
            pdb.set_trace()
            errors = []
            
            combined = [i + [o] for i, o in zip(normalized_inputs, outputs)]
            for case in combined:
                inputs = [1] + case[:-1]
                output = case[-1]
                prediction = self.predict_fn(feature_weights, inputs)
                error = self.error_fn(prediction, output)
                errors.append(error)
                print('-' * 20)
                print('Predicted result: {}'.format(prediction))
                print('Actual result: {}'.format(output))
                print('Error: {}'.format(error))
            print('=' * 20)
            
            mse = sum(e ** 2 for e in errors) / len(errors)
            print('Mean squared error: {}'.format(mse))

            print('=' * 20)
            
        return feature_weights
    
    
def predict(weights, input):
    return sum(w * j for j, w in zip(input, weights))
    
def error(prediction, output):
    return prediction - output
    

if __name__ == '__main__':
    rm = RegressionModel(predict_fn=predict, error_fn=error, debug=True)
    
# def batch_logistic(training_data, iteration_count=10, learning_rate=0.01, debug=False):
    # inputs = [[1] + i for i in normalize_features([d[:-1] for d in training_data])]
    # outputs = [d[-1] for d in training_data]
    # weights = [random.random() for _ in inputs[0]]
    
    # data_count = len(inputs)
    
    # max_errors = []
    # for _ in range(iteration_count):
        # combined = list(zip(inputs, outputs))
        # random.shuffle(combined)
        # regression_inputs, regression_outputs = zip(*combined)
        
        # predictions = [sigmoid(p) for p in predict_all(weights, regression_inputs)]
        # errors = [p - o for p, o in zip(predictions, regression_outputs)]
        # for w_i in range(len(weights)):
            # weights[w_i] -= learning_rate * sum(e * i[w_i] for i, e in zip(regression_inputs, errors)) / data_count
        
        # max_error = max(errors) ** 2
        # max_errors.append(max_error)
        
    # if debug:
        # print(max_errors)
        
        # import matplotlib.pyplot as plt
        # plt.plot(list(range(iteration_count)), max_errors)
        # plt.show()
    
    # return weights

# def batch_gd(training_data, iteration_count=10, learning_rate=0.01, debug=False):
    # inputs = [[1] + i for i in normalize_features([d[:-1] for d in training_data])]
    # outputs = [d[-1] for d in training_data]
    # weights = [random.random() for _ in inputs[0]]
    
    # data_count = len(inputs)
    
    # max_errors = []
    # for _ in range(iteration_count):
        # combined = list(zip(inputs, outputs))
        # random.shuffle(combined)
        # regression_inputs, regression_outputs = zip(*combined)
        
        # predictions = predict_all(weights, regression_inputs)
        # errors = [p - o for p, o in zip(predictions, regression_outputs)]
        # for w_i in range(len(weights)):
            # weights[w_i] -= learning_rate * sum(e * i[w_i] for i, e in zip(regression_inputs, errors)) / data_count
        
        # max_error = max(errors) ** 2
        # max_errors.append(max_error)
        
    # if debug:
        # print(max_errors)
        
        # import matplotlib.pyplot as plt
        # plt.plot(list(range(iteration_count)), max_errors)
        # plt.show()
    
    # return weights
    
# def stochastic_gd(training_data, iteration_count=10, learning_rate=0.01, debug=False):
    # inputs = [[1] + i for i in normalize_features([d[:-1] for d in training_data])]
    # outputs = [d[-1] for d in training_data]
    # weights = [random.random() for _ in inputs[0]]
    
    # data_count = len(inputs)
    
    # max_errors = []
    # for _ in range(iteration_count):
        # combined = list(zip(inputs, outputs))
        # random.shuffle(combined)
        # regression_inputs, regression_outputs = zip(*combined)
        
        # max_error = 0
        # for i, o in zip(regression_inputs, regression_outputs):
            # prediction = predict(weights, i)
            # error = prediction - o
            # weights = [w - learning_rate * error * j for j, w in zip(i, weights)]
            # if abs(error) > abs(max_error):
                # max_error = error
        
        # max_errors.append(max_error ** 2)
        
    # if debug:
        # print(max_errors)
    
        # import matplotlib.pyplot as plt
        # plt.plot(list(range(iteration_count)), max_errors)
        # plt.show()
    
    # return weights
    
def normal_equations(training_data):
    input_matrix = np.matrix([[1] + d[:-1] for d in training_data])
    output_vector = np.matrix([d[-1] for d in training_data]).getT()
    weights = (input_matrix.getT() * input_matrix).getI() * input_matrix.getT() * output_vector
    return weights
