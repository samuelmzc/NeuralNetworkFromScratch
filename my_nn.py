from nn_funcs import *

class L_Layer_NN:
    def __init__(self, X, Y, layers, learning_rate, iterations, regularization, lambd):
        self.X = X
        self.Y = Y
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.lambd = lambd
    
    def train(self, X_dev, Y_dev):
        parameters = initialize_parameters(self.layers)
        costs = []
        dev_costs = []

        for i in range(self.iterations):
            AL, caches = L_layer_forwardprop(self.X, parameters)
            AL_dev, caches_dev = L_layer_forwardprop(X_dev, parameters)
            grads = L_layer_backprop(AL, self.Y, caches)
            parameters = update_params(parameters, grads, self.Y.shape[1], self.learning_rate, self.lambd, self.regularization)

            cost = BinaryCrossEntropy(AL, self.Y, parameters, self.lambd, self.regularization)
            costs.append(cost)

            dev_cost = BinaryCrossEntropy(AL_dev, Y_dev, parameters, self.lambd, self.regularization)
            dev_costs.append(dev_cost)

            if i%10000 == 0:
                print(f"{i}th iteration => Training cost: {cost} / Dev cost = {dev_cost}")
        
        return parameters, costs, dev_costs

        
    def predict(self, parameters, X_dev):
        AL, _ = L_layer_forwardprop(X_dev, parameters)
        AL = (AL > 0.5).astype(int)
        return AL
    

class optimized_L_Layer_NN:
    def __init__(self, X, Y, layers, learning_rate, mini_batch_size, epochs, regularization, lambd, optimizer, lr_decay):
        self.X = X
        self.Y = Y
        self.layers = layers
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.regularization = regularization
        self.lambd = lambd
        self.optimizer = optimizer
        self.lr_decay = lr_decay
    
    def fit(self):
        n, m = self.X.shape
        params = initialize_parameters(self.layers)
        mini_batches = minibatchs(self.X, self.Y, self.mini_batch_size)
        costs = []
        learning_rate = self.learning_rate

        if self.optimizer.lower() == "gd":
            pass
        elif self.optimizer.lower() == "adam":
            v, s = initialize_adam(params)      # Initialize momentums
            t = 0                               # Initialize Adam counter


        for i in range(self.epochs):

            batch_costs = 0
            actual_lr = (1/(1 + self.lr_decay*i))*learning_rate

            for mini_batch in mini_batches:
                (mini_batchX, mini_batchY) = mini_batch
                aL, caches = L_layer_forwardprop(mini_batchX, params)
                grads = L_layer_backprop(aL, mini_batchY, caches)

                if self.optimizer.lower() == "gd":
                    params = update_params(params, grads, m, self.learning_rate, self.lambd, self.regularization)

                elif self.optimizer.lower() == "adam":
                    t += 1
                    params, v, s = update_params_adam(params, grads, v, s, t, actual_lr, beta1 = 0.9,
                                                beta2 = 0.999, epsilon = 1e-8)
                
                batch_costs += BinaryCrossEntropy(aL, mini_batchY, params, self.lambd, self.regularization)
            
            epoch_cost = batch_costs / m

            if i % 10 == 0:
                costs.append(epoch_cost)

            if i % 1000 == 0:
                print(f"Cost for {i}th epoch: {epoch_cost}")
        
        return params, costs

    def predict(self, parameters, X_pred):
        aL, _ = L_layer_forwardprop(X_pred, parameters, )
        aL = (aL > 0.5).astype(int)
        return aL

            








                
        
