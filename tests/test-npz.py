import numpy as np
import scipy.io as sio

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 1. MATLAB 파일 읽기
mat_S = sio.loadmat("./data/S_tensors_10modes_1550.mat")     # {'__header__', '__version__', '__globals__', 'SK': array(...) }
mat_hrw = sio.loadmat("./data/hrw_8192.mat") # {'hrw': array(...) }
mat_betas = sio.loadmat("./data/betas_1550.mat") # {'betas': array(...) }



# 2. 필요한 키만 추출
S = mat_S["SR"]
hrw = mat_hrw["hrw"]
betas = mat_betas["betas"]

betas = np.transpose(betas, (1, 0))
# betas[:, -1]= 0
# betas = betas[:8, :]

betas[:, 0] *= 1e3
betas[:, 1] *= 1e0
betas[:, 2] *= 1e-3
betas[:, 3] *= 1e-6
betas[:, 4] *= 1e-9
betas[:, 5] *= 1e-12

# 3. npz로 저장
np.savez("./data/predefined_data_1550.npz", S=S, hrw=hrw, betas=betas)