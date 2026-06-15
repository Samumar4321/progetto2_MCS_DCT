import numpy as np
from scipy.fft import dctn, idctn

class ComprimiImmagine:
    
    def __init__(self, matrix, f, d):        
        self.matrix = matrix
        self.f = f
        self.d = d        
    
    def matrix_cut(self):
        H, W = self.matrix.shape
        cut_H = (H // self.f) * self.f
        cut_W = (W // self.f) * self.f
        cropped_matrix = self.matrix[:cut_H, :cut_W]        
        print(f"Dimensioni originali: {H}x{W}")
        print(f"Dimensioni ritagliate: {cut_H}x{cut_W}")
        return cropped_matrix
    
    def split_in_blocks(self, matrix):
        H, W = matrix.shape
        blocks = []
        for i in range(0, H, self.f):
            for j in range(0, W, self.f):
                block = matrix[i:i+self.f, j:j+self.f]
                blocks.append(block)
        return np.array(blocks)
    
    def process_block(self, block):
        c = dctn(block, type=2, norm='ortho')
        for row in range(self.f):
            for col in range(self.f):
                if (row + col) >= self.d:
                    c[row, col] = 0
        ff = idctn(c, type=2, norm='ortho')
        ff_int = np.clip(np.round(ff).astype(int), 0, 255)
        return ff_int
    
    def process_blocks(self, blocks):
        processed_blocks = []
        for block in blocks:
            processed_block = self.process_block(block)
            processed_blocks.append(processed_block)
        return np.array(processed_blocks)
    
    def compress_matrix(self):
        cropped_matrix = self.matrix_cut()
        blocks = self.split_in_blocks(cropped_matrix)
        processed_blocks = self.process_blocks(blocks)
        H, W = cropped_matrix.shape
        reconstructed_matrix = np.zeros((H, W), dtype=int)
        i = 0
        for row in range(0, H, self.f):
            for col in range(0, W, self.f):
                reconstructed_matrix[row:row+self.f, col:col+self.f] = processed_blocks[i]
                i += 1
        return reconstructed_matrix


