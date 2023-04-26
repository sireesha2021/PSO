import numpy as np


def sigmoid(z):
    # prevent overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def ReLU(z):
    return np.maximum(z, 0)


def Leaky_ReLU(z):
    alpha = 0.01
    return np.maximum(z, z * alpha)


class MLP_PSO():
    def __init__(self, number_of_layers, number_of_neurons, activation_functions, loss_function):
        self.number_of_layers = number_of_layers
        self.number_of_neurons = number_of_neurons
        self.activation_functions = activation_functions
        self.loss_function = loss_function

    def fit_and_transform(self, X_train, X_test, y_train, y_test, epochs, swarm_size, number_of_informants, alpha, beta,
                          gamma, delta, epsilon):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.epochs = epochs
        self.swarm_size = swarm_size
        self.number_of_informants = number_of_informants
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.delta = delta
        self.position = [0] * self.swarm_size
        self.velocity = [0] * self.swarm_size
        self.pbest = [0] * self.swarm_size
        self.ibest = [0] * self.swarm_size
        self.gbest = [0]
        self.informants = [0] * self.swarm_size

        # negative infinite number
        temp_max = -0x3F3F3F3F
        # calculate the length of each particle
        temp = np.zeros(self.swarm_size)
        # do this for each particle
        for j in range(self.swarm_size):
            # calculate dimensions, number of weights and biases of each layer equal to
            # the number of previous layer's neurons times the number of this layer's neurons and plus
            # the number of this layer's neurons
            # calculate input layer first
            swarm_dimension = self.number_of_neurons[0] * self.X_train.shape[1] + self.number_of_neurons[0]
            # calculate rest of the layers
            for i in range(self.number_of_layers):
                if i == 0:
                    continue
                else:
                    swarm_dimension += self.number_of_neurons[i] * self.number_of_neurons[i - 1] + \
                                       self.number_of_neurons[i]

            # initialize position with [0,1)
            self.position[j] = np.random.rand(1, swarm_dimension)
            # we can also use PSO to optimize activation functions, just append them at the end of the position
            for a in self.activation_functions:
                self.position[j] = np.append(self.position[j], a)
            # remember to add the number of activation function to the velocity array too, and then initialize with [0,1)
            self.velocity[j] = np.random.rand(1, swarm_dimension + self.number_of_layers)
            # initialize each particle's best record with initialized position
            self.pbest[j] = self.position[j]
            # selecting each particle's informants by random
            temp_list = list(range(swarm_size))
            # the particle itself will be one of it's informants, so we remove itself from the list first
            temp_list.remove(j)
            # and then shuffle the list
            np.random.shuffle(temp_list)
            # select first few particles as informants
            self.informants[j] = np.array(temp_list[:number_of_informants - 1])
            self.informants[j] = np.append(self.informants[j], j)
            # initialize globe best
            temp[j] = self.update_fitness(X_train, y_train, self.position[j])
            if temp_max < temp[j]:
                temp_max = temp[j]
                self.gbest = self.position[j]
        # initialize informants' best
        for i in range(self.swarm_size):
            temp_max = -0x3F3F3F3F
            for j in range(self.number_of_informants):
                if temp_max < temp[self.informants[i][j]]:
                    temp_max = temp[self.informants[i][j]]
                    self.ibest[i] = self.position[self.informants[i][j]]

        # the number of epochs
        for t in range(self.epochs):
            # this is the previous globe best record
            temp_max = self.update_fitness(X_train, y_train, self.gbest)
            # updating each particle's position
            for i in range(self.swarm_size):
                # the new velocity is composed of old velocity, particle's best, informants' best, and globe best
                self.velocity[i] = self.alpha * self.velocity[i] \
                                   + np.random.rand() * self.beta * (self.pbest[i] - self.position[i]) \
                                   + np.random.rand() * self.gamma * (self.ibest[i] - self.position[i]) \
                                   + np.random.rand() * self.delta * (self.gbest - self.position[i])
                self.position[i] = self.position[i] + self.epsilon * self.velocity[i]
                # boundary of activation functions
                for a in range(swarm_dimension + self.number_of_layers - 1, swarm_dimension - 1, -1):
                    if self.position[i][0, a] > 4.5:
                        self.position[i][0, a] = 4.5
                    elif self.position[i][0, a] < 0.5:
                        self.position[i][0, a] = 0.5
                # the record of fitness of the new position
                temp[i] = self.update_fitness(X_train, y_train, self.position[i])
                # if new record better than previous particle's best, then replace it
                if self.update_fitness(X_train, y_train, self.pbest[i]) < temp[i]:
                    self.pbest[i] = self.position[i]
                # if new record better than globe best, then replace it
                if temp_max < temp[i]:
                    temp_max = temp[i]
                    self.gbest = self.position[i]
            # updating informants' best
            for i in range(self.swarm_size):
                temp_max = -0x3F3F3F3F
                for j in range(self.number_of_informants):
                    if temp_max < temp[self.informants[i][j]]:
                        temp_max = temp[self.informants[i][j]]
                        self.ibest[i] = self.position[self.informants[i][j]]
        # return the globe best fitness on the test dataset as the final result
        return self.update_fitness(X_test, y_test, self.gbest) * 100 / (X_test.shape[0])

    def update_fitness(self, X_data, y_data, position):
        weights = [1] * self.number_of_layers
        biases = [1] * self.number_of_layers
        activation_functions = [1] * self.number_of_layers
        # deserialize weights, biases, and activation functions
        self.deserialization(position, weights, biases, activation_functions)
        # forward
        value = X_data.T
        for i in range(self.number_of_layers):
            z = np.dot(weights[i], value) + biases[i]
            if activation_functions[i] == 'sigmoid':
                value = sigmoid(z)
            elif activation_functions[i] == 'tanh':
                value = tanh(z)
            elif activation_functions[i] == 'ReLU':
                value = ReLU(z)
            elif activation_functions[i] == 'Leaky_ReLU':
                value = Leaky_ReLU(z)

        a = value
        a = a.reshape(-1, 1)
        # threshold is 0.5
        a = np.where(a >= 0.5, 1, 0)
        # compare with the answer
        return np.sum(a == y_data.reshape(-1, 1))

    def deserialization(self, position, weights, biases, activation_functions):
        # pointer indicate where the start position
        index = 0
        position = position.reshape(1, -1)
        # because the weights and biases matrix is grouped by layers, so we deserialize by layers too
        for i in range(self.number_of_layers):
            temp = [0] * self.number_of_neurons[i]
            temp = np.array(temp).reshape(-1, 1)
            # if it is the first layer
            if i == 0:
                # because the number of columns equal to dataset shape[1] which is 4 in this case
                for j in range(self.X_train.shape[1]):
                    # the number of rows equal to hidden layer's neurons, and stack them by columns
                    temp = np.hstack((temp, position[0, index:index + self.number_of_neurons[i]].reshape(-1, 1)))
                    # update pointer
                    index = index + self.number_of_neurons[i]
            else:
                for j in range(self.number_of_neurons[i - 1]):
                    temp = np.hstack((temp, position[0, index:index + self.number_of_neurons[i]].reshape(-1, 1)))
                    index = index + self.number_of_neurons[i]

            # delete the first column, because all of them is 0
            temp = np.delete(temp, 0, axis=1)
            weights[i] = temp
            # the biases is an array that equal with the number of neurons
            biases[i] = position[0, index:index + self.number_of_neurons[i]].reshape(-1, 1)
            index = index + self.number_of_neurons[i]

        # at the end, deserialize activation function by boundary
        for i in range(self.number_of_layers):
            # 0.5<= sigmoid < 1.5
            if 0.5 <= position[0, index] < 1.5:
                activation_functions[i] = 'sigmoid'
            # 1.5<= tanh <2.5
            elif 1.5 <= position[0, index] < 2.5:
                activation_functions[i] = 'tanh'
            # 2.5<= ReLU < 3.5
            elif 2.5 <= position[0, index] < 3.5:
                activation_functions[i] = 'ReLU'
            # 3.5<= Leaky_ReLU <4.5
            elif 3.5 <= position[0, index] <= 4.5:
                activation_functions[i] = 'Leaky_ReLU'
            index = index + 1
