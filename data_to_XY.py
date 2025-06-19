import numpy as np
import pandas as pd

def csv_to_XY_strlabel(csv, last_col_str, yes):
    """
    Convierte los datos de un archivo csv a los arrays X, Y para entrenarlos con la red

    Argumentos:
    csv -- str nombre_archivo.csv
    last_col_str -- str con el título de la última columna
    yes -- str que indica y = 1

    Devuelve:
    X -- array con las m muestras (inputs)
    Y -- array target label para cada muestra m
    m_train -- número de muestras para entrenar
    m_test -- número de muestras para testear
    m -- m_train + m_test
    """

    # Read de database (.csv)
    df = pd.read_csv(csv)

    # Define the length of the training and test sets
    m_train = int(np.floor(0.9*len(df)))
    m_test = len(df) - m_train
    m = m_train + m_test

    X = np.zeros((len(df.columns) - 2, m))

    for i in range(1, len(df.columns) - 2):
        X[i, :] = df[f"{df.columns[i]}"].to_numpy()

    # If output = 1 : Disease is Presence / output = 0 : Absence
    Y = np.zeros((1, m))

    for i in range(m):
        if df[last_col_str].to_numpy()[i] == yes:
            Y[0, i] = 1
        else:
            Y[0, i] = 0
    
    return X, Y, m_train, m_test, m


def csv_to_XY(csv, last_col_str):
    """
    Convierte los datos de un archivo csv a los arrays X, Y para entrenarlos con la red

    Argumentos:
    csv -- str nombre_archivo.csv
    last_col_str -- str con el título de la última columna
    yes -- str que indica y = 1

    Devuelve:
    X -- array con las m muestras (inputs)
    Y -- array target label para cada muestra m
    m_train -- número de muestras para entrenar
    m_test -- número de muestras para testear
    m -- m_train + m_test
    """

    # Read de database (.csv)
    df = pd.read_csv(csv)

    # Define the length of the training and test sets
    m_train = int(np.floor(0.7*len(df)))
    m_test = len(df) - m_train
    m = m_train + m_test

    X = np.zeros((len(df.columns) - 2, m))

    for i in range(len(df.columns) - 2):
        X[i, :] = df[f"{df.columns[i + 1]}"].to_numpy()

    # If output = 1 : Disease is Presence / output = 0 : Absence
    Y = np.zeros((1, m))
    Y[0, :] = df[f"{df.columns[-1]}"].to_numpy()
    
    return X, Y, m_train, m_test, m
