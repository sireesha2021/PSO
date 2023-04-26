import numpy as np

class MLP():
    def __init__(self, number_of_layers, number_of_neurons, activation_functions, learning_rate, loss_function):
        self.number_of_layers = number_of_layers
        self.number_of_neurons = number_of_neurons
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def fit_and_transform(self, data_original, test_data, outcome, epochs):
        self.data_original = data_original
        self.outcome = outcome
        self.epochs = epochs
        self.weights = [1]*self.number_of_layers
        self.biases = [1]*self.number_of_layers

        # initialize input layer weights and biases with random number [0,1)
        self.weights[0] = np.random.rand(self.number_of_neurons[0], data_original.shape[1])
        self.biases[0] = np.random.rand(self.number_of_neurons[0], 1)
        for i in range(self.number_of_layers):
            if i == 0:
                continue
            else:
                # initialize hidden layer and output layer
                self.weights[i] = np.random.rand(self.number_of_neurons[i], self.number_of_neurons[i - 1])
                self.biases[i] = np.random.rand(self.number_of_neurons[i], 1)

        # initialize z value array size
        z = [1]*self.number_of_layers
        # initialize a value array size
        a = [1]*self.number_of_layers

        # the number of epochs
        for j in range(self.epochs):
            # we use mini-batch learning here, a batch size is 128
            for k in range(0,data_original.shape[0]-128,128):
                # initialize temp variable value with input data
                value = (data_original.iloc[k:k+128,:]).T
                # forward
                for i in range(self.number_of_layers):
                    # get z value by using weights dot product a, and plus bias
                    z[i] = np.dot(self.weights[i], value) + self.biases[i]
                    # apply different activation functions
                    if self.activation_functions[i] == 'sigmoid':
                        value = self.sigmoid(z[i])
                    elif self.activation_functions[i] == 'tanh':
                        value = self.tanh(z[i])
                    elif self.activation_functions[i] == 'ReLU':
                        value = self.ReLU(z[i])
                    elif self.activation_functions[i] == 'Leaky_ReLU':
                        value = self.Leaky_ReLU(z[i])
                    # store a value
                    a[i] = value

                # back-propagation
                # initialize delta of biases and weights array size
                delta_bias = [1] * self.number_of_layers
                delta_weights = [1] * self.number_of_layers
                # calculate output layer first
                if self.loss_function == 'MSE':
                    # different loss function will have different way to calculate delta value of final layer
                    loss = a[-1] - self.outcome[k:k+128]
                    delta_final = np.array(loss) * np.array(self.dsigmoid(z[-1]))
                    delta_bias[-1] = delta_final
                    # if there are more than one layer
                    if len(a) != 1:
                        delta_weights[-1] = np.dot(delta_final, a[-2].T)
                    else:
                        # if only one layer
                        delta_weights[-1] = np.dot(delta_final, self.data_original.iloc[k:k+128,:])
                elif self.loss_function == 'cross_entropy':
                    loss = a[-1] - self.outcome[k:k+128]
                    delta_final = loss
                    delta_bias[-1] = delta_final
                    if len(a) != 1:
                        delta_weights[-1] = np.dot(delta_final, a[-2].T)
                    else:
                        delta_weights[-1] = np.dot(delta_final, self.data_original.iloc[k:k+128,:])

                delta = delta_final
                # calculate other layers' delta value
                for i in range(1, self.number_of_layers):
                    if self.activation_functions[~i] == 'sigmoid':
                        delta = np.array(np.dot(self.weights[~i + 1].T, delta)) * np.array(self.dsigmoid(z[~i]))
                    elif self.activation_functions[~i] == 'tanh':
                        delta = np.array(np.dot(self.weights[~i + 1].T, delta)) * np.array(self.dtanh(z[~i]))
                    elif self.activation_functions[~i] == 'ReLU':
                        delta = np.array(np.dot(self.weights[~i + 1].T, delta)) * np.array(self.dReLU(z[~i]))
                    elif self.activation_functions[~i] == 'Leaky_ReLU':
                        delta = np.array(np.dot(self.weights[~i + 1].T, delta)) * np.array(self.dLeaky_ReLU(z[~i]))
                    # delta of bias is the delta value
                    delta_bias[~i] = delta
                    # delta of weights have to dot product a value
                    if i != self.number_of_layers - 1:
                        delta_weights[~i] = np.dot(delta, a[~i - 1].T)
                    else:
                        delta_weights[~i] = np.dot(delta, self.data_original.iloc[k:k+128,:])
                # finally update weights and biases, remember to divide total delta by 128 which is batch size
                for i in range(self.number_of_layers):
                    self.weights[i] = self.weights[i] - self.learning_rate * delta_weights[i] / 128
                    self.biases[i] = self.biases[i] - self.learning_rate * (np.dot(delta_bias[i],np.matrix(np.ones(128)).T))/128
                # start next epoch

        result = []
        # using the same batch size as train, do forward again
        for k in range(0,test_data.shape[0]-128,128):
            value = test_data.iloc[k:k+128,:].T
            for i in range(self.number_of_layers):

                z[i] = np.dot(self.weights[i], value) + self.biases[i]

                if self.activation_functions[i] == 'sigmoid':
                    value = self.sigmoid(z[i])
                elif self.activation_functions[i] == 'tanh':
                    value = self.tanh(z[i])
                elif self.activation_functions[i] == 'ReLU':
                    value = self.ReLU(z[i])
                elif self.activation_functions[i] == 'Leaky_ReLU':
                    value = self.Leaky_ReLU(z[i])

                a[i] = value

            result = np.vstack((np.array(result).reshape(-1,1),(np.array(a[-1].T))))


        return result

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)

    def ReLU(self, z):
        return np.maximum(z, 0)

    def Leaky_ReLU(self, z):
        alpha = 0.01
        return np.maximum(z, z * alpha)

    def dsigmoid(self, z):
        return np.array(1.0 / (1.0 + np.exp(-z))) * np.array(1 - 1.0 / (1.0 + np.exp(-z)))

    def dtanh(self, z):
        return 1 - np.array(np.tanh(z)) * np.array(np.tanh(z))

    def dReLU(self, z):
        z[z >= 0] = 1
        z[z < 0] = 0
        return z

    def dLeaky_ReLU(self, z):
        z[z >= 0] = 1
        z[z < 0] = 0.01
        return z
