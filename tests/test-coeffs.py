import numpy as np
import matplotlib.pyplot as plt

coeffs = [ 0.6780-0.3007j, -0.0009-0.0005j,  0.0008-0.0014j,  0.0028+0.0063j,
        -0.0068-0.0006j,  0.2320+0.0418j,  0.2862-0.0338j,  0.3133-0.1195j]


# plot the absolute value of the coefficients

coeffs = np.array(coeffs)

# normalize the coefficients
coeffs = coeffs / np.sum(np.abs(coeffs))

#print the amplitude of the coefficients
print(np.abs(coeffs))
