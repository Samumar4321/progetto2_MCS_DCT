import math
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
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

class CompressioneImmagini:

    def __init__(self, root):
        self.root = root
        self.root.title("Compressione Immagini con DCT")
        self.root.geometry("500x350")
        self.root.resizable(False, False)

        # Variabili per memorizzare percorso immagine
        self.image_path = tk.StringVar()

        # Selezione file .bmp
        lbl_file = tk.Label(root, text="Seleziona un'immagine BMP:", font=("Arial", 10, "bold"))
        lbl_file.pack(pady=(20, 5))

        frame_file = tk.Frame(root)
        frame_file.pack(fill="x", padx=20)

        entry_file = tk.Entry(frame_file, textvariable=self.image_path, width=45)
        entry_file.pack(side="center", padx=(0, 5))

        btn_sfoglia = tk.Button(
            frame_file, text="Sfoglia...", command=self.seleziona_file
        )
        btn_sfoglia.pack(side="center")

        # Inserimento del valore F
        lbl_f = tk.Label(
            root,
            text="Inserisci ampiezza macro-cella F (intero):",
            font=("Arial", 10),
        )
        lbl_f.pack(pady=(20, 5))

        self.entry_f = tk.Entry(root, width=15, justify="center")
        self.entry_f.pack()

        # Inserimento del valore d
        lbl_d = tk.Label(
            root,
            text="Inserisci soglia di taglio d (intero):\n(deve essere compreso tra 0 e 2F - 2)",
            font=("Arial", 10),
        )
        lbl_d.pack(pady=(20, 5))

        self.entry_d = tk.Entry(root, width=15, justify="center")
        self.entry_d.pack()

        # Bottone di conferma
        btn_elabora = tk.Button(
            root,
            text="Avvia Elaborazione",
            font=("Arial", 11, "bold"),
            bg="#2ecc71",
            fg="white",
            command=self.valida_e_invia,
        )
        btn_elabora.pack(pady=30)

    def seleziona_file(self):
        file_selezionato = filedialog.askopenfilename(
            title="Seleziona un file BMP",
            filetypes=[("Immagini BMP", "*.bmp"), ("Tutti i file", "*.*")],
        )
        if file_selezionato:
            self.image_path.set(file_selezionato)

    def valida_e_invia(self):
        percorso = self.image_path.get()
        stringa_f = self.entry_f.get()
        stringa_d = self.entry_d.get()

        # Controllo se il file esiste ed è stato selezionato
        if not percorso or not os.path.exists(percorso):
            messagebox.showerror(
                "Errore", "Seleziona un file .bmp valido dal filesystem."
            )
            return

        # Controllo se F è un intero positivo
        try:
            F = int(stringa_f)
            if F <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Errore", "Il parametro F deve essere un intero positivo."
            )
            return

        # Controllo se d è un intero e rispetta i vincoli legati a F
        soglia_massima = 2 * F - 2
        try:
            d = int(stringa_d)
            if d < 0 or d > soglia_massima:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Errore",
                f"Il parametro d deve essere un intero compreso tra 0 e {soglia_massima} (dato che F={F}).",
            )
            return

        # Se tutti i controlli passano, chiudiamo l'interfaccia e procediamo
        messagebox.showinfo(
            "Configurazione Corretta",
            f"Dati validati con successo!\n\nFile: {os.path.basename(percorso)}\nF = {F}\nd = {d}",
        )
        self.root.destroy()

        # Salviataggio dei dati
        self.risultati = {"percorso": percorso, "F": F, "d": d}

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
        label="DCT2 SciPy Fast ($O(N^2 \log N)$)",
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

    finestra = tk.Tk()
    app = CompressioneImmagini(finestra)
    finestra.mainloop()

    if hasattr(app, "risultati"):
        dati_progetto = app.risultati        
        print("\nDati pronti per l'elaborazione successiva:")
        print(f"Percorso immagine: {dati_progetto['percorso']}")
        print(f"Ampiezza cella (F): {dati_progetto['F']}")
        print(f"Soglia taglio (d): {dati_progetto['d']}")

