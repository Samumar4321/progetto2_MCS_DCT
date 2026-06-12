import numpy as np
import time
import matplotlib.pyplot as plt
import tkinter as tk 
from compressioneImmagine import ComprimiImmagine
from InterfacciaImmagini import InterfacciaImmagini
from scipy.fftpack import dctn
from homemade_dct import dct_2d
from PIL import Image

def part1():
    #Definizione dei valori di N
    valori_N = [4, 8, 16, 32, 64, 128, 256]
    tempi_custom = []
    tempi_scipy = []

    # Calcolo dei tempi di esecuzione per DCT personalizzata e DCT di scipy
    for N in valori_N:
        # Generazione di una matrice quadrata casuale NxN
        matrix = np.random.rand(N, N)
        
        # Tempo per DCT personalizzata
        start_time = time.perf_counter()
        dct_2d(matrix)
        end_time = time.perf_counter()
        tempi_custom.append(end_time - start_time)
        
        # Tempo per DCT di scipy
        start_time = time.perf_counter()
        dctn(matrix, type=2, norm='ortho')
        end_time = time.perf_counter()
        tempi_scipy.append(end_time - start_time)

        print(
            f"N = {N:3d} | Fatto in casa: {tempi_custom[-1]:.6f} s | SciPy: {tempi_scipy[-1]:.6f} s"
        )

    # Grafico semilogaritmico dei tempi di esecuzione
    plt.figure(figsize=(10, 6))

    # semilogy imposta la scala logaritmica solo sull'asse Y (ordinate)
    plt.semilogy(
        valori_N,
        tempi_custom,
        marker="o",
        color="indigo",
        linestyle="-",
        label="DCT2 Fatta in casa ($O(N^3)$)",
    )
    plt.semilogy(
        valori_N,
        tempi_scipy,
        marker="s",
        color="crimson",
        linestyle="--",
        label="DCT2 SciPy Fast ($O(N^2 \ log N)$)",
    )

    # Etichette e configurazioni assi
    plt.title("Confronto Tempi di Esecuzione DCT2 al variare di N")
    plt.xlabel("Dimensione della matrice (N x N)")
    plt.ylabel("Tempo di esecuzione [secondi] (Scala Logaritmica)")
    plt.xticks(valori_N)  # Forza la visualizzazione dei nostri punti N su X
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()

    # Mostra e salva il grafico
    plt.show()

def load_img(path):
    try:
        img = Image.open(path).convert('L')
        return np.array(img)
    except Exception as e:
        print(f"Errore nel caricamento dell'immagine: {e}")
        return None

if __name__ == "__main__":
    part1()

    finestra = tk.Tk()
    app = InterfacciaImmagini(finestra)
    finestra.mainloop()    
        

