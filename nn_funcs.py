import numpy as np


def minibatchs(X, Y, minibatch_size : int):
    '''
    Convierte el batch de X e Y en mini-batches de tamaño minibatch_size

    Argumentos:
    X -- training set, array
    Y -- target labels, array
    minibatch_size -- tamaño de cada mini-batch, int

    Devuelve:
    mini_batchs -- tupla con minibatches de X y de Y
    '''
    n, m = X.shape
    mini_batches = []
    num_complete_batches = int(np.floor(m/minibatch_size))

    # For compleet batches:
    for k in range(num_complete_batches):
        mini_batch_X = X[:, minibatch_size * k : minibatch_size * (k + 1)]

        mini_batch_Y = np.zeros((1, minibatch_size))
        mini_batch_Y[0, :] = Y[0, minibatch_size * k : minibatch_size * (k + 1)]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % minibatch_size != 0:
        mini_batch_X = X[:, minibatch_size * num_complete_batches :]

        mini_batch_Y = np.zeros((1, m % minibatch_size))
        mini_batch_Y[0, :] = Y[0, minibatch_size * num_complete_batches :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



def initialize_parameters(layers):
    """
    Inicializa los pesos y bias de la red neuronal. Para evitar simetrías en la red,
    se le dan valores aleatorios a los pesos, y se multiplican por un número pequeño
    para no obtener valores grandes en el foward prop y optimizar el back prop

    Argumentos:
    layers -- array de numpy que contiene el número de unidades por capa de la red
    (incluyendo la capa del input)

    Devuelve:
    params -- diccionario de python con los pesos y bias
    """

    params = {}
    for l in range(1, len(layers)):
        params["W" + str(l)] = np.random.randn(layers[l], layers[l-1])*0.01
        params["b" + str(l)] = np.zeros((layers[l], 1))
    
    return params

def initialize_adam(parameters):
    """
    Inicia los pesos esperados v y s utilizados para el Adam

    Argumentos:
    parameters -- diccionario con parámetros

    Devuelve:
    v -- diccionario con valores de v
    s -- diccionario con valores de s
    """

    L = int(len(parameters)/2)
    v = {}
    s = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
    
    return v, s

def L2_norm(array):
    """
    Cálculo de la norma L2 de un array

    Argumentos:
    array -- array al que se le quiere calcular la norma

    Devuelve:
    L2_norm -- norma L2 del array
    """

    return (np.sum(array**2))

def sigmoid(z):
    """
    Evalúa la función sigmoide

    Argumentos:
    z -- número, array, etc...
    
    Devuelve:
    σ(z)
    """

    return 1/(1+np.exp(-z))


def relu(z):
    """
    Evalúa ReLU(z)

    Argumentos:
    z -- np.array

    Devuelve:
    ReLU(z)
    """

    return np.maximum(z, 0)


def linear_foward_prop(A, W, b):
    '''
    Obtiene Z = W A + b y almacena los valores A, W, b

    Argumentos:
    A -- vector de activación de la capa anterior (n[l - 1], 1)
    W -- array de pesos de la capa actual (n[l], n[l-1])
    b -- array de bias de capa actual (n[l], 1)

    Devuelve:
    Z -- propagación lineal (sin aplicar la función de activación)
    cache -- tupla de python con A, W y b para facilitar el back prop
    '''

    Z = np.dot(W, A) + b 
    cache = (A, W, b)

    return Z, cache


def activated_propagation(A, W, b, activation, keep_prob = 0.5):
    """
    Aplica la función de activación de la capa a Z

    Argumentos:
    A -- array con la activación de la capa anterior (n[l-1], 1)
    W -- array de pesos de la capa actual (n[l], n[l-1])
    b -- array de bias de la capa actual (n[l], 1)
    activation -- str indicando la activación deseada
    keep_prob -- probabilidad de que la neurona sobreviva (dropout)

    Devuelve:
    A_now -- array con la activación de la capa actual
    """

    Z, linear_cache = linear_foward_prop(A, W, b)

    if activation == "sigmoid":
        A_now = sigmoid(Z)

    if activation == "relu":
        A_now = relu(Z)


    total_cache = (Z, linear_cache)
    
    return A_now, total_cache


def L_layer_forwardprop(X, parameters):
    """
    Foward Propagation de toda la red

    Argumentos:
    X -- array con datos para entrenar la red
    parameters -- diccionario con pesos y bias inicializados de la red

    Devuelve:
    AL -- activación de la última capa (L)
    caches -- lista con los caches calculados
    """

    caches = []
    L = int(len(parameters)/2) # Pues por cada capa tenemos pesos y sesgos (dos parámetros por capa)
    A = X
    for l in range(1, L):
        A_prev = A
        A, cache = activated_propagation(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
        caches.append(cache)
    
    AL, cache = activated_propagation(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    return AL, caches


def BinaryCrossEntropy(AL, Y, parameters, lambd, regularization : str):
    """
    Calcula Binary Cross Entropy cost, con el regularizador correspondiente

    Argumentos:
    AL -- activación de la última capa de la NN, array
    Y -- targets, array
    parameters -- pesos W, b, array (en caso de regularización)
    lambd -- parámetro de regularización, float
    regularization -- regularización especificada, str

    Devuelve:
    J -- Binary cross entropy, float
    """

    m = Y.shape[1]
    L = int(len(parameters)/2)
    J_not_reg = -(1/m)*np.sum(Y*np.log(AL) + (1 - Y)*np.log(1 - AL))

    if regularization.lower() == "none":
        J = J_not_reg

    elif regularization == "L2":
        reg_term = 0
        for l in range(1, L+1):
            reg_term += lambd/(2*m) * L2_norm(parameters["W" + str(l)])
        
        J = J_not_reg + reg_term

    return J 


def dAdWdb(dZ, cache, keep_prob = 0.5):
    """
    Calcula las derivadas dW, db, dA_prev en función de dZ (que depende de la derivada de la activación)

    Argumentos:
    dZ -- array derivada de la función de coste respecto a Z de la capa
    cache -- tupla (Z, A_prev, W, b)
    keep_prob -- probabilidad de que la neurona sobreviva (dropout)

    Devuelve:
    dA_prev -- array derivada de la función de coste respecto a A de la capa anterior
    dW -- array derivada de la """""" respecto a W de la capa actual
    db -- """"""""""""""""""" respecto a b de """"""
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)


    return dA_prev, dW, db

def activated_back_prop(dA, cache, activation = "sigmoid" or "relu"):
    """
    Obtenemos dZ en cierta capa para aplicar luego dWdbdA(dZ, cache)

    Argumentos:
    dA -- array derivada respecto A de la capa
    cache -- tupla (Z, linear_cache)
    activation -- str indicando la activación de la capa

    Devuelve:
    dA_prev -- array con la derivada respecto A de la capa anterior (iteración del backprop)
    dW -- array con la derivada respecto a W de la capa actual
    db -- array con la derivada respecto a b de la capa actual
    """

    assert activation == "sigmoid" or activation == "relu"


    Z, linear_cache = cache

    if activation == "relu":
        if Z > 0:
            dZ = 0
        else:
            dZ = dA
        
        dA_prev, dW, db = dAdWdb(dZ, linear_cache)
    
    if activation == "sigmoid":
        dZ = dA*(sigmoid(Z)*(1-sigmoid(Z)))
        dA_prev, dW, db = dAdWdb(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_layer_backprop(AL, Y, caches):
    """
    Back Propagation de toda la red neuronal.

    Argumentos:
    AL -- activación de la última capa
    Y -- array con target labels
    caches -- array con (Z, A_prev, W, b) de cada capa (1 - L)

    Devuelve:
    grads -- diccionario con el gradiente de A, W y b
    """

    grads = {}
    L = len(caches)
    m = Y.shape[1]
    current_cache = caches[L-1]

    dAL = -(1/m)*((Y)/(AL) - (1 - Y)/(1 - AL))
    dA_prev, dW, db = activated_back_prop(dAL, current_cache, "sigmoid")
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db 

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = activated_back_prop(dAL, current_cache, "sigmoid")
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l+1)] = db
    
    return grads


def update_params(params, grads, m, learning_rate, lambd, regularization : str):
    """
    Una vez obtenidos los gradientes de la función de coste, actualizamos los parámetros mediante descenso
    de gradiente

    Argumentos:
    params -- diccionario con los parámetros W, b de cada capa
    grads -- diccionario con dW, db de cada capa
    learning_rate -- float ritmo de aprendizaje

    Devuelve:
    params -- diccionario con parámetros actualizados
    """

    L = int(len(params)/2)
    if regularization.lower() == "none" or regularization == None:
        for l in range(L):
            params["W" + str(l + 1)] -= learning_rate*grads["dW" + str(l + 1)]
            params["b" + str(l + 1)] -= learning_rate*grads["db" + str(l + 1)]

    elif regularization == "L2":
        for l in range(L):
            params["W" + str(l + 1)] -= learning_rate*(grads["dW" + str(l + 1)] + (lambd/m)*params["W" + str(l+1)])
            params["b" + str(l + 1)] -= learning_rate*grads["db" + str(l + 1)]

    return params

def update_params_adam(params, grads, v, s, t, learning_rate, beta1, beta2, epsilon):
    L = int(len(params)/2)

    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        v_corrected["dW" + str(l)] = v["dW" + str(l)]/(1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)]/(1 - beta1**t)

        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * grads["dW" + str(l)]**2
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * grads["db" + str(l)]**2
        s_corrected["dW" + str(l)] = s["dW" + str(l)]/(1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)]/(1 - beta2**t)

        params["W" + str(l)] -= learning_rate * (v_corrected["dW" + str(l)]/(np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
        params["b" + str(l)] -= learning_rate * (v_corrected["db" + str(l)]/(np.sqrt(s_corrected["db" + str(l)]) + epsilon))
    
    return params, v, s

