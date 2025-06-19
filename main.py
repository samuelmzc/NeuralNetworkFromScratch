import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from my_nn import *
from data_to_XY import *
from plots import *


csv = "angina_dataset.csv"
last_col_str = "Heart Disease"
yes = "Presence"

X, Y, m_train, m_test, m = csv_to_XY_strlabel(csv, last_col_str, yes)

n = X.shape[0] # nº de features


# Split into training and test sets
Y_train = np.zeros((1, m_train))
Y_test = np.zeros((1, m_test))

Y_train[0, :] = Y[0, :m_train]
Y_test[0, :] = Y[0, m_train:]


X_train = X[:, :m_train]
X_test = X[:, m_train:]

layers = np.array([n, 4, 3, 1])
learning_rate = 0.00007
iterations = 100000
mini_batch_size = 2**6
epochs = 5000

regularization = "none"
lambd = 10e7
optimizer = "adam"
lr_decay = 0


optimized_model = optimized_L_Layer_NN(X_train, Y_train, layers, learning_rate, 
                                       mini_batch_size, epochs, regularization, 
                                       lambd, optimizer, lr_decay)

opt_params, costs = optimized_model.fit()
preds = optimized_model.predict(opt_params, X_test)
test_pred = optimized_model.predict(opt_params, X_train)

dev_accuracy_array = (preds == Y_test).astype(int)[0]
dev_accuracy = np.sum(dev_accuracy_array)/len(dev_accuracy_array)*100

train_accuracy_array = (test_pred == Y_train).astype(int)[0]
train_accuracy = np.sum(train_accuracy_array)/len(train_accuracy_array)*100

print(f"Train set accuracy with mini batchs for {epochs} epochs : {train_accuracy}")
print(f"Dev set accuracy with mini batchs for {epochs} epochs : {dev_accuracy}")

epoch_space = np.linspace(1, epochs, int(epochs/10))
plt.plot(epoch_space, np.array(costs))
plt.title(f"Cost | optimizer : {optimizer} | LR decay rate : {lr_decay} | TSA/CVA : {train_accuracy}%/{dev_accuracy}%")
plt.xlabel("Epochs")
plt.ylabel("Cost")

plt.grid()
plt.tight_layout()

plt.savefig(f"{optimizer}_{lr_decay}_{mini_batch_size}.pdf")

plt.show()

'''modelo = L_Layer_NN(X_train, Y_train, layers, learning_rate, iterations, regularization, lambd)
parameters, train_costs, dev_costs = modelo.train(X_test, Y_test)
predictions_dev = modelo.predict(parameters, X_test)
predictions_train = modelo.predict(parameters, X_train)

dev_acc_array = (predictions_dev == Y_test).astype(int)[0]
dev_acc = np.sum(dev_acc_array)/len(dev_acc_array)

train_acc_array = (predictions_train == Y_train).astype(int)[0]
train_acc = np.sum(train_acc_array)/len(train_acc_array)


print(f"train accuracy : {train_acc*100}% \n test accuracy : {dev_acc*100}%")


trainerr_vs_deverr(train_costs, dev_costs)'''


