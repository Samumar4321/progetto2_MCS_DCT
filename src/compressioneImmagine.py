import numpy as np
from scipy.fft import dctn, idctn

class ComprimiImmagine:
    
    def __init__(self, f, d):        
        self.f = f
        self.d = d        
    
    def matrix_cut(self, matrix):
        H, W = matrix.shape
        cut_H = (H // self.f) * self.f
        cut_W = (W // self.f) * self.f
        cropped_matrix = matrix[:cut_H, :cut_W]        
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
    
    def reconstruct_matrix(self, blocks, original_shape):
        H, W = original_shape
        res = np.zeros((H, W), dtype=int)
        i = 0
        for row in range(0, H, self.f):
            for col in range(0, W, self.f):
                res[row:row+self.f, col:col+self.f] = blocks[i]
                i += 1
        return res
    
    def compress_matrix(self, matrix):
        cropped_matrix = self.matrix_cut(matrix)
        blocks = self.split_in_blocks(cropped_matrix)     
        processed_blocks = self.process_blocks(blocks)
        reconstructed_matrix = self.reconstruct_matrix(processed_blocks, cropped_matrix.shape)
        return reconstructed_matrix


