# -*- coding: utf-8 -*-
"""DSA_Assignment_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mbXRft2oS3wlihCWRB-B040FqszO7h-R

# Problem 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N = [1, 5, 10]
dof = 3

np.random.seed(1)
x = np.zeros((max(N), int(1e6)))
for i in range(max(N)):
    x[i] = np.random.chisquare(dof, int(1E6))

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=0.05)

plt.title('Problem 1')
### N = 1
ax = fig.add_subplot(3, 1, 1)

x_i = x[:N[0], :].mean(0)

ax.hist(x_i, bins=np.linspace(0,10,1000), histtype='stepfilled', alpha=0.6, density=True, label = 'chi_2')

mu = x_i.mean()
sigma = np.sqrt(2.0*dof/N[0] )
lin_space = np.linspace(0, 10, 1000)
dist = norm(mu, sigma)
ax.plot(lin_space, dist.pdf(lin_space), '-k', label='uniform')
ax.set_xlim(0.0, 10)

plt.legend(loc = 'center right')

ax.text(0.99, 0.95, r"$N = %i$" % N[0], ha='right', va='top', transform=ax.transAxes)

ax.set_xlabel(r'$x$')
ax.set_ylabel('$p(x)$')

### N = 5
ax = fig.add_subplot(3, 1, 2)

x_i = x[:N[1], :].mean(0)

ax.hist(x_i, bins=np.linspace(0,10,1000), histtype='stepfilled', alpha=0.6, density=True, label = 'chi_2')

mu = x_i.mean()
sigma = np.sqrt(2.0*dof/N[1] )
lin_space = np.linspace(0, 10, 1000)
dist = norm(mu, sigma)
ax.plot(lin_space, dist.pdf(lin_space), '-k', label='uniform')
ax.set_xlim(0.0, 10)

plt.legend(loc = 'center right')

ax.text(0.99, 0.95, r"$N = %i$" % N[1], ha='right', va='top', transform=ax.transAxes)

ax.set_xlabel(r'$x$')
ax.set_ylabel('$p(x)$')

### N = 10
ax = fig.add_subplot(3, 1, 3)

x_i = x[:N[2], :].mean(0)

ax.hist(x_i, bins=np.linspace(0,10,1000), histtype='stepfilled', alpha=0.6, density=True, label = 'chi_2')

mu = x_i.mean()
sigma = np.sqrt(2.0*dof/N[2] )
lin_space = np.linspace(0, 10, 1000)
dist = norm(mu, sigma)
ax.plot(lin_space, dist.pdf(lin_space), '-k', label='uniform')
ax.set_xlim(0.0, 10)

plt.legend(loc = 'center right')

ax.text(0.99, 0.95, r"$N = %i$" % N[2], ha='right', va='top', transform=ax.transAxes)

ax.set_xlabel(r'$x$')
ax.set_ylabel('$p(x)$')
plt.tight_layout()
plt.show()

"""# Problem 2"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

data = []
with open('test.dat') as fp:
  lines = fp.readlines()[1:]
  for line in lines:
    data += [line.split()]

col1 = []
for i in range(np.shape(data)[0]):
  col1 += data[i][0]

col1 = []
for d in data:
  col1 += [float(d[0])]
col1 = np.array(col1)

col2 = []
for d in data:
  col2 += [float(d[1])]
col2 = np.array(col2)

log_col1 = np.log(col1)
log_col2 = np.log(col2)

plt.scatter(log_col2, log_col1)
plt.xlabel("Redshit (logscale)")
plt.ylabel("Luminosity (logscale)")
plt.title('Problem 2')
plt.show()

coeff_spear, p_spear = stats.spearmanr(col1, col2)
coeff_pear, p_pear = stats.pearsonr(col1, col2)
coeff_ken, p_ken = stats.kendalltau(col1, col2)

print("Spearman: ", "Correlation coeff = ", coeff_spear, ", p-value = ", p_spear)
print("Pearson: ", "Correlation coeff = ", coeff_pear, ", p-value = ", p_pear)
print("Kendall-tau: ", "Correlation coeff = ", coeff_ken, ", p-value = ", p_ken)

'''
By eye, there is correlation in the data, as we are increasing the redshift the average luminosity is getting higher, excpet for 0 redshift.
'''

"""# Problem 3"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

N = 1000

x = np.linspace(0, 19, 20)
freq = np.array([2.75,7.80,11.64,13.79,14.20,13.15,11.14,8.72,6.34,4.30,2.73,1.62,0.91,0.48,0.24,0.11,0.05,0.02,0.01,0.00])

freq = freq/100

plt.step(x, freq, label='pdf')
dist = stats.weibull_min(2, -1, 6)
x_2 = np.linspace(0, 19, N)
plt.plot(x_2, dist.pdf(x_2), label='best-fit')
plt.legend()
plt.title('Problem 3')
plt.show()

"""# Problem 4"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import t

np.random.seed(0)

N = 1000
dof = N-2

mean = 0
std_dev = 1

X1 = norm.rvs(mean, std_dev, N)
X2 = norm.rvs(mean, std_dev, N)
corr_pear, p_pear = stats.pearsonr(X1, X2)

t_val = -abs(corr_pear) * np.sqrt(dof / (1 -(-abs(corr_pear))*2))

p_t_test = 2 * stats.t.cdf(t_val, dof)

print("Pearson correlation coefficient : ", corr_pear)
print("Pearson p-value : ", p_pear)
print("P Value using student-t-distribution :", p_t_test)