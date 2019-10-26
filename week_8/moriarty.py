#! /usr/bin/env python3

# Moriarty's linear least squares answer for "the cycle of twelve"
#
# Usage:
#   ./moriarty.py w07-data.tbl
#

import numpy as np
import math
import re
import sys

# Ooh look! Parsing for the w07-data.tbl
# This leaves us with
#    N         : number of experiments (columns in the table)
#    G         : number of genes (rows in the table)
#    X[i]      : array of time points, in hrs, for the N experiments
#    S_true[i] : array of sigmas for the experiments
#    Y[i][t]   : GxN: observed tpm for gene i, time point t
#   
datafile = sys.argv[1]
with open(datafile) as f:
    # First header line gives us the time points
    fields = f.readline().split()
    X = []
    for s in fields:
        match = re.search(r'^(\d+)hr', s)
        X.append(int(match.group(1)))
    X = np.array(X)
    N = len(X)

    # Second header line gives us "gene" followed by +=SD's
    fields = f.readline().split()
    S_true = np.zeros(N)
    for i,s in enumerate(fields[1:]):
        match = re.search(r'^\+-(\d+)', s)
        S_true[i] = float(match.group(1))

    # Third header line is just ------ stuff
    f.readline()

    # Remaining lines are data
    genenames = []
    Y = []
    for line in f.readlines():
        fields = line.split()
        genenames.append(fields[0])
        Y.append( np.array( [ float(s) for s in fields[1:]] ))
    G = len(Y)



# Moriarty's method: ordinary least squares on:
#    y_t = b + (a cos p) sin t + (a sin p) cos t
#
b_fit = np.zeros(G)
a_fit = np.zeros(G)
p_fit = np.zeros(G)

b_opt = np.zeros(G)
a_opt = np.zeros(G)
p_opt = np.zeros(G)

for g in range(G):
    # We have to set up a matrix A the way numpy.linalg.lstsq() wants it.
    #
    A = np.zeros((N, 3))  # observations x coefficients
    for i in range(N):
        A[i][0] = 1.
        A[i][1] = np.sin(2. * math.pi * X[i] / 24)  
        A[i][2] = np.cos(2. * math.pi * X[i] / 24)

    try:
        result    = np.linalg.lstsq(A, Y[g])[0]
    except:
        sys.exit("Linear least square fit failed")

    p_fit[g]  = np.arctan(result[2] / result[1])   # in radians at first
    b_fit[g]  = result[0]
    a_fit[g]  = result[1] / np.cos(p_fit[g])

    p_fit[g] = 24 * p_fit[g] / (2 * math.pi)       # now in hours
    if a_fit[g] < 0:                               # there's a symmetry in the solution we have to deal with.
        a_fit[g]  = -a_fit[g]
        p_fit[g] += 12
    while p_fit[g] < 0:  p_fit[g] += 24
    while p_fit[g] > 24: p_fit[g] -= 24



## Output
#
print("{0:12s} {1:>6s} {2:>6s} {3:>6s}".format('genename', 'b', 'a', 'p'))
print("{0:12s} {1:6s} {2:6s} {3:6s}".format('-'*12, '-'*6,'-'*6,'-'*6))
for g in range(G):
    print("{0:12s} {1:6.2f} {2:6.2f} {3:6.2f}".format(genenames[g], b_fit[g], a_fit[g], p_fit[g]))



    
# Output the data set to a file
# Useful for checking if the parser works; compare to original table.
#
def output_data(outfile):
    with open(outfile, 'w') as f:
        print("{0:12s} ".format(""), end='', file=f)
        for i in range(N):
            print("{0:4.0f}hr ".format(X[i]), end='',file=f)
        print('', file=f)

        print("{0:12s} ".format("gene"), end='', file=f)
        for i in range(N):
            label = "+-{0:.0f}".format(S_true[i])
            print("{0:>6s} ".format(label), end='',file=f)
        print('',file=f)

        print("{0:12s} ".format("------------"), end='',file=f)
        for i in range(N):
            print("{0:6s} ".format("------"), end='',file=f)
        print('',file=f)

        for g in range(G):
            print("{0:12s} ".format(genenames[g]), end='',file=f)
            for i in range(N):
                print("{0:6.1f} ".format(Y[g][i]), end='',file=f)
            print('',file=f)
 
                  
            
