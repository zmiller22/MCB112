#! /usr/bin/env python3

# Student's game.
#
# Usage:
#    ./student-game.py <nX>

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

nX      = int(sys.argv[1])   # number of samples dropped by the tea machine

seed = np.random.randint(0,10000)
np.random.seed(seed)

# Set up the 20x21 dart board grid, and the mu and sigma values
# associated with each grid position.
#
gridrows = np.linspace( 100.0,   5.0, 20)   # rows i of the grid are sigma, std. dev.
gridcols = np.linspace(-100.,  100.0, 21)   # columns j of the grid are mu, mean (location)
nrows    = len(gridrows)
ncols    = len(gridcols)

# Student throws a uniformly distributed dart into the grid, and this
# chooses mu, sigma. These values are unknown to the customers.
#
true_row = np.random.randint(0, nrows)    # Note, randint(0,n) samples 0..n-1
true_col = np.random.randint(0, ncols)

true_sigma = gridrows[true_row]
true_mu    = gridcols[true_col]

# Student's tea distribution machine drops observed samples onto the
# line on the bar: nX of them, X[0..nX-1]
#
X  = np.random.normal(loc=true_mu, scale=true_sigma, size=nX)

sample_mean  = np.mean(X)
sample_stdev = np.std(X, ddof = 1)   ## ddof is "degrees of freedom". 0 = population sd; 1 = sample sd.

# Print text output
# (before showing the graph displays)
#
print ("Welcome to Student's game night...")
print ("")
print ("The RNG seed is {}".format(seed))
print ("")
print ("The hidden grid has {} rows for sigma {}..{},".format(nrows, gridrows[0], gridrows[-1]))
print ("                and {} cols for mu {}..{}.".format(ncols, gridcols[0], gridcols[-1]))
print ("")
print ("Student picked grid col = {}, row = {}".format(true_col, true_row))
print ("and thus mu = {0:.1f}, sigma = {1:.1f}".format(true_mu, true_sigma))
print ("")
print ("Student's tea distribution machine shows the customers {} samples:".format(nX))
for sample in X: print("  {0:8.2f}".format(sample))
print ("")
print ("which give sample mean: {0:8.2f}".format(sample_mean))
print ("   and sample std.dev.: {0:8.2f}".format(sample_stdev))


# The inference rules that Student uses to calculate betting odds
# from: these give him what he thinks are the expected P(mu | x1..xn,
# sigma) distribution.
#
def probdist_beginner(X, sigma, mu_values):
    """ 
    Given an ndarray X_1..X_n, and a known sigma;
    and a list of the mu values in each column;
    return a list of the inferred P(mu | X,sigma) for each column.
    """
    xbar = np.mean(X)
    N    = len(X)
    Pr   = [ stats.norm.pdf(x, loc=xbar, scale= sigma / np.sqrt(N)) for x in mu_values ]  # proportional to std error of the mean
    Z    = sum(Pr)                   # normalization constant
    Pr   = [ p / Z for p in Pr ]     # normalization to a discrete probability distribution
    return Pr

def probdist_advanced(X, mu_values):
    """ 
    Given an ndarray X_1..X_n,
    and a list of the mu values in each column;
    return a list of the inferred P(mu | X) for each column.
    """
    xbar = np.mean(X)
    s    = np.std(X, ddof=1)     # note that numpy.sd() by default calculates a population std dev; to get sample std. dev., set ddof=1
    N    = len(X)
    Pr   = [ stats.norm.pdf(x, loc=xbar, scale= s / np.sqrt(N)) for x in mu_values ]  # proportional to std error of the mean
    Z    = sum(Pr)                   # normalization constant
    Pr   = [ p / Z for p in Pr ]     # normalization to a discrete probability distribution
    return Pr

def tdist_advanced(X, mu_values):
    """ 
    Given an ndarray X_1..X_n,
    and a list of the mu values in each column;
    return a list of the inferred P(mu | X) for each column,
	according to a Student's t distribution with N-1 degrees of freedom.
    """
    N    = len(X)
    t    = [ stats.ttest_1samp(X, mu)[0] for mu in mu_values ]
    Pr   = [ stats.t.pdf(val, N-1) for val in t ]
    Z    = sum(Pr)
    Pr   = [ p / Z for p in Pr ]     # normalization to a discrete probability distribution
    return Pr




# The bar's rules for determining fair odds.
#
PrB = probdist_beginner(X, true_sigma, gridcols)
PrA = probdist_advanced(X, gridcols)


# Set up our graphical display.
#
# We'll show the pub's supposedly "fair odds" probability distribution plot for the
# beginner version and the advanced version, as semilog plots.
#
f, (ax1, ax2) = plt.subplots(2,1, sharey=True)  # figure consists of 2 graphs, 2 rows x 1 col

ax1.semilogy(gridcols, PrB, label="bar's estimate: beginner (sigma known)")
ax1.xaxis.set_ticks(gridcols)
ax1.set(xlabel='$\mu$', ylabel='$P(\mu \mid \sigma)$')
ax1.legend(loc="best")

ax2.semilogy(gridcols, PrA, label="bar's estimate: advanced (sigma unknown)")
ax2.xaxis.set_ticks(gridcols)
ax2.set(xlabel='$\mu$', ylabel='$P(\mu)$')
ax2.legend(loc="best")

plt.show()




