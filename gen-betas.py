import numpy as np


num_modes = 6
betas = np.load(f'./beta_{num_modes}modes.npy')

num_modes_list = [21, 45, 91]

for num_modes in num_modes_list:
    new_betas = np.load(f'./beta_6modes.npy')
    if new_betas.shape[0] < num_modes:
        # Repeat the last row to extend
        num_to_repeat = num_modes - new_betas.shape[0]
        last_row = new_betas[-1:]
        repeat_rows = np.tile(last_row, (num_to_repeat, 1))
        extended_betas = np.concatenate([new_betas, repeat_rows], axis=0)
    else:
        extended_betas = new_betas[:num_modes]
    print(extended_betas.shape)
    np.save(f'./beta_{num_modes}modes.npy', extended_betas)
