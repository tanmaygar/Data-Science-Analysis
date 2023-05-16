import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import astroML
from astroML.stats import sigmaG

mu_gauss = 1.5
sigma_gauss = 0.5

# generate 1000 random numbers from a Gaussian distribution with mean of 1.5 and standard deviation of 0.5
# generated_samples = np.random.normal(mu_gauss, sigma_gauss, 1000)
normal_distribution = stats.norm(mu_gauss, sigma_gauss)
sampled_points = normal_distribution.rvs(1000)
generated_samples = sampled_points
y_sampled_points = normal_distribution.pdf(sampled_points)
plt.plot(sampled_points, y_sampled_points, 'o')
plt.show()

# sample mean
sample_mean = np.mean(generated_samples)
print('sample mean: {}'.format(sample_mean))

# sample variance
sample_variance = np.var(generated_samples)
print('sample variance: {}'.format(sample_variance))

# sample skewness
sample_skewness = stats.skew(generated_samples)
print('sample skewness: {}'.format(sample_skewness))

# sample kurtosis
sample_kurtosis = stats.kurtosis(generated_samples)
print('sample kurtosis: {}'.format(sample_kurtosis))

# sample standard deviation using MAD
sample_std_mad = 1.4826 * np.median(np.abs(generated_samples - np.median(generated_samples)))
print('sample standard deviation using MAD: {}'.format(sample_std_mad))

# sample standard deviation using sigma G
sample_std_sigma_g = sigmaG(generated_samples)
print('sample standard deviation using sigma G: {}'.format(sample_std_sigma_g))

# Plot pdf of the Gaussian distribution 
x = np.arange(0, 5, 0.01)
y = stats.norm.pdf(x, mu_gauss, sigma_gauss)
plt.plot(x, y)
plt.show()

