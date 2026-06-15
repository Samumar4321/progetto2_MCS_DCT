import math
import numpy as np
    
def dct_1d(segnale):
    N = np.shape(segnale)[0]
    # Vettore per memorizzare i coefficienti DCT
    coeff = np.zeros(N)
    # Ciclo per le frequenze
    for k in range(N):
        somma = 0.0
        # Ciclo dei singoli campioni del segnale
        for j in range(N):
            angolo = k * math.pi * (2 * j + 1) / (2 * N)
            somma += segnale[j] * math.cos(angolo)
        # Normalizzazione del coefficiente
        if k == 0:
            alpha = math.sqrt(1.0 / N)
        else:
            alpha = math.sqrt(2.0 / N)
        coeff[k] = alpha * somma
    return coeff

def dct_2d(matrice):
    num_righe = np.shape(matrice)[0]
    num_colonne = np.shape(matrice)[1]
    matrix_temp = np.zeros((num_righe, num_colonne))
    # Applica dct1 su ogni riga
    for i in range(num_righe):
        matrix_temp[i, :] = dct_1d(np.squeeze(matrice[i, :]))
    matrix_final = np.zeros((num_righe, num_colonne))
    # Applica dct1 su ogni colonna
    for j in range(num_colonne):
        colonna = np.squeeze(matrix_temp[:, j])
        matrix_final[:, j] = dct_1d(colonna)
    return matrix_final