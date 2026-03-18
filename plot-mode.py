import matplotlib.pyplot as plt
import numpy as np


modes_fields = np.load('modes_128x128.npy')

plt.figure(figsize=(15, 12))
plt.subplot(2, 3, 1)
plt.imshow(np.abs(modes_fields[0])**2, aspect='auto', origin='lower', cmap='turbo')
plt.title('Mode 1')
plt.subplot(2, 3, 2)
plt.imshow(np.abs(modes_fields[1])**2, aspect='auto', origin='lower', cmap='turbo')
plt.title('Mode 2')
plt.subplot(2, 3, 3)
plt.imshow(np.abs(modes_fields[2])**2, aspect='auto', origin='lower', cmap='turbo')
plt.title('Mode 3')
plt.subplot(2, 3, 4)
plt.imshow(np.abs(modes_fields[3])**2, aspect='auto', origin='lower', cmap='turbo')
plt.title('Mode 4')
plt.subplot(2, 3, 5)
plt.imshow(np.abs(modes_fields[4])**2, aspect='auto', origin='lower', cmap='turbo')
plt.title('Mode 5')
plt.subplot(2, 3, 6)
plt.imshow(np.abs(modes_fields[5])**2, aspect='auto', origin='lower', cmap='turbo')
plt.title('Mode 6')
plt.show()