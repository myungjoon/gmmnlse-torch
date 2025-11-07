import numpy as np

field_raman = np.load('./data/output_fields_raman.npy')
field_no_raman = np.load('./data/output_fields_no_raman.npy')

intensity_raman = np.abs(field_raman)**2
intensity_no_raman = np.abs(field_no_raman)**2

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(intensity_raman[0], label='Raman Effect', alpha=0.7)
plt.plot(intensity_no_raman[0], 'k--', label='No Raman Effect',  alpha=0.7)
plt.show()