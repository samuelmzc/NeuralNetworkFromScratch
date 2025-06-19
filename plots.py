import numpy as np
import matplotlib.pyplot as plt

def trainerr_vs_deverr(train_cost, dev_cost):
    iterations = len(train_cost)
    it_space = np.linspace(1, iterations, iterations)/10000
    plt.plot(it_space, train_cost, label = "train set error")
    plt.plot(it_space, dev_cost, label = "dev set error")
    plt.title("Train and Dev Errors")
    plt.xlabel("Iterations (10.000)")
    plt.ylabel("Cost function")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

