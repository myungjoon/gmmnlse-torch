import numpy as np
import scipy.io as sio

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 1. MATLAB 파일 읽기
# Read MATLAB files and extract 'S' tensors as numpy arrays
mat_S1 = sio.loadmat('./data/S_tensors_10modes_old.mat')
mat_S2 = sio.loadmat('./data/S_tensors_10modes_new.mat')
S1 = mat_S1['SK']
S2 = mat_S2['SK']

# Compare S1 and S2 using the Frobenius norm of their difference
diff_norm = np.linalg.norm(S1 - S2)
print(f"Frobenius norm of (S1 - S2): {diff_norm}")



