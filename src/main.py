import math
import numpy as np
import time
from scipy.fftpack import dct, dctn

def dct_1d(segnale):  
    N = len(segnale)
    # Vettore per memorizzare i coefficienti DCT
    coeff = np.array([0.0] * N)    
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
    num_righe = len(matrice)
    num_colonne = len(matrice[0])
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

if __name__ == "__main__":
    # segnale di esempio dato dal prof
    # dovrebbe uscire: [401,1 66.0, 109, -112, 65.4, 121, 116, 28.8]
    signal_1d = np.array([231, 32, 233, 161, 24, 71, 140, 245])    
    coef_1d = dct_1d(signal_1d)    
    print("Coefficienti DCT1:")
    for i, coef in enumerate(coef_1d):
        print(f"X_{i} = {coef:.2f} / {coef:.2e}")
    
    # segnale 2d dato dal prof
    # dovrebbe uscire: 1.11e+03 4.40e+01 7.59e+01 -1.38e+02 3.50e+00 1.22e+02 1.95e+02 -1.01e+02
    #                  7.71e+01 1.14e+02 -2.18e+01 4.13e+01 8.77e+00 9.90e+01 1.38e+02 1.09e+01
    #                  4.48e+01 -6.27e+01 1.11e+02 -7.63e+01 1.24e+02 9.55e+01 -3.98e+01 5.85e+01
    #                 -6.99e+01 -4.02e+01 -2.34e+01 -7.67e+01 2.66e+01 -3.68e+01 6.61e+01 1.25e+02
    #                 -1.09e+02 -4.33e+01 -5.55e+01 8.17e+00 3.02e+01 -2.86e+01 2.44e+00 -9.41e+01
    #                 -5.38e+00 5.66e+01 1.73e+02 -3.54e+01 3.23e+01 3.34e+01 -5.81e+01 1.90e+02
    #                  7.88e+01 -6.45e+01 1.18e+02 -1.50e+01 -1.37e+02 -3.06e+01 -1.05e+02 3.98e+01
    #                  1.97e+01 -7.81e+01 9.72e-01 -7.23e+01 -2.15e+01 8.13e+01 6.37e+01 5.90e+00
    signal_2d = np.array([[231, 32, 233, 161, 24, 71, 140, 245],
                        [247, 40, 248, 245, 124, 204, 36, 107],
                        [234, 202, 245, 167, 9, 217, 239, 173],
                        [193, 190, 100, 167, 43, 180, 8, 70],
                        [11, 24, 210, 177, 81, 243, 8, 112],
                        [97, 195, 203, 47, 125, 114, 165, 181],
                        [193, 70, 174, 167, 41, 30, 127, 245],
                        [87, 149, 57, 192, 65, 129, 178, 228]])    
    coef_2d = dct_2d(signal_2d)    
    print("Coefficienti DCT2:")
    for i, row in enumerate(coef_2d):
        for j, coef in enumerate(row):
            print(f"X_{i}_{j} = {coef:.2f} / {coef:.2e}", end=' | ')
        print()
    
    
    # -- LIBRERIA SCIPY --
    print("\n\n--- DCT con libreria scipy ---")
    coeff_1d = dct(signal_1d, type=2, norm='ortho')
    print("Coefficienti DCT1 SCIPY:")
    for i, coef in enumerate(coef_1d):
        print(f"X_{i} = {coef:.2f} / {coef:.2e}")
        
    coeff_2d = dctn(signal_2d, type=2, norm='ortho')
    print("Coefficienti DCT2 SCIPY:")
    for i, row in enumerate(coeff_2d):
        for j, coef in enumerate(row):
            print(f"X_{i}_{j} = {coef:.2f} / {coef:.2e}", end=' | ')
        print()