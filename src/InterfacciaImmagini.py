import os
from threading import Thread
import tkinter as tk 
from tkinter import filedialog, messagebox
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from compressioneImmagine import ComprimiImmagine

class InterfacciaImmagini:

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
        entry_file.pack(side="left", padx=(0, 5))

        btn_sfoglia = tk.Button(
            frame_file, text="Sfoglia...", command=self.seleziona_file
        )
        btn_sfoglia.pack(side="left")

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

        # Controllo se f è un intero positivo
        try:
            f = int(stringa_f)
            if f <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Errore", "Il parametro f deve essere un intero positivo."
            )
            return

        # Controllo se d è un intero e rispetta i vincoli legati a f
        soglia_massima = 2 * f - 2
        try:
            d = int(stringa_d)
            if d < 0 or d > soglia_massima:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Errore",
                f"Il parametro d deve essere un intero compreso tra 0 e {soglia_massima} (dato che f={f}).",
            )
            return

        # Se tutti i controlli passano, chiudiamo l'interfaccia e procediamo
        messagebox.showinfo(
            "Configurazione Corretta",
            f"Dati validati con successo!\n\nFile: {os.path.basename(percorso)}\nF = {f}\nd = {d}",
        )
        
        Thread(target=self.process_image, args=(percorso, f, d)).start()  
    
    def load_img(self, path):
        try:
            img = Image.open(path).convert('L')
            return np.array(img)
        except Exception as e:
            print(f"Errore nel caricamento dell'immagine: {e}")
            return None
    
    def show_image_comparison(self, img_original, img_compressa, f):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # --- Pannello 1: Immagine Originale ---      
        axes[0].imshow(img_original, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("Immagine Originale")
        axes[0].axis('off')
        
        # --- Pannello 2: Immagine Compressa ---
        axes[1].imshow(img_compressa, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f"Immagine Compressa (Blocchi {f}x{f})")
        axes[1].axis('off')
        
        # Aggiusta gli spazi per non far sovrapporre i titoli e mostra a video
        plt.tight_layout()
        plt.show()
        
    def process_image(self, percorso, f, d):
        img_original = self.load_img(percorso)
        if img_original is None:
            return
        comprimitore = ComprimiImmagine(img_original, f, d)
        img_reconstructed = comprimitore.compress_matrix()
        self.show_image_comparison(img_original, img_reconstructed, f)    
       
