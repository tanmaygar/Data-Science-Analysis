# Plot a Cauchy distribution with μ=0 and γ=1.5 superposed on the top of a Gaussian distribution with μ=0 and σ=1.5.

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mu_cauchy = 0
gamma_cauchy = 1.5
x = np.arange(-10, 10, 0.01)
y_cauchy = stats.cauchy.pdf(x, mu_cauchy, gamma_cauchy)

mu_gauss = 0
sigma_gauss = 1.5
y_gauss = stats.norm.pdf(x, mu_gauss, sigma_gauss)

# Plot the two distributions
plt.plot(x, y_cauchy, label='Cauchy')
plt.plot(x, y_gauss, label='Gaussian')
plt.legend()
plt.show()
