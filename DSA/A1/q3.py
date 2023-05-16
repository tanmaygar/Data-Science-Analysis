# Plot Poisson distribution with mean of 5, superposed on top of a Gaussian distribution with mean of 5 and standard deviation of square root of 5.

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

x = np.arange(0, 10, 0.01)

mu_gauss = 5
sigma_gauss = np.sqrt(5)
y_gauss = stats.norm.pdf(x, mu_gauss, sigma_gauss)

mu_poisson = 5
y_poisson = stats.poisson.pmf([0,1,2,3,4,5,6,7,8,9,10], mu_poisson)

# Plot the two distributions
plt.step([0,1,2,3,4,5,6,7,8,9,10], y_poisson, label='Poisson')
plt.plot(x, y_gauss, label='Gaussian')
plt.legend()
plt.show()
