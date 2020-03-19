# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
canonical ordering of symplectic group elements
from "How to efficiently select an arbitrary clifford group element"
       by Robert Koenig and John A. Smolin
"""

from numpy import *

def directsum(m1, m2):
    """ direct sum """
    n1 = len(m1[0])
    n2 = len(m2[0])
    output = zeros((n1+n2, n1+n2), dtype=int8)
    for i in range(0, n1):
        for j in range(0, n1):
            output[i, j] = m1[i, j]
    for i in range(0, n2):
        for j in range(0, n2):
            output[i+n1, j+n1] = m2[i, j]
    return output

def inner(v, w):
    """ symplectic inner product """
    t = 0
    for i in range(0, size(v)>>1):
        t += v[2*i]*w[2*i+1]
        t += w[2*i]*v[2*i+1]
    return t%2

def transvection(k, v):
    """ applies transvection Z_k to v """
    return (v+inner(k, v)*k)%2

def int2bits(i, n):
    """ converts integer i to an length n array of bits """
    output = zeros(n, dtype=int8)
    for j in range(0, n):
        output[j] = i&1
        i >>= 1
    return output


def findtransvection(x, y):
    """
    finds h1,h2 such that y = Z_h1 Z_h2 x
    Lemma 2 in the text
    Note that if only one transvection is required output[1] will be
    zero and applying the all-zero transvection does nothing.
    """
    output = zeros((2, size(x)), dtype=int8)
    if array_equal(x, y):
        return output
    if inner(x, y) == 1:
        output[0] = (x+y)%2
        return output
#
    # find a pair where they are both not 00
    z = zeros(size(x))
    for i in range(0, size(x)>>1):
        ii = 2*i
        if ((x[ii]+x[ii+1]) != 0) and ((y[ii]+y[ii+1]) != 0):  # found the pair
            z[ii] = (x[ii]+y[ii])%2
            z[ii+1] = (x[ii+1]+y[ii+1])%2
            if (z[ii]+z[ii+1]) == 0:  # they were the same so they added to 00
                z[ii+1] = 1
                if x[ii] != x[ii+1]:
                    z[ii] = 1
            output[0] = (x+z)%2
            output[1] = (y+z)%2
            return output
    # didn't find a pair
    # so look for two places where x has 00 and y doesn't, and vice versa

    # first y==00 and x doesn't
    for i in range(0, size(x)>>1):
        ii = 2*i
        if ((x[ii]+x[ii+1]) != 0) and ((y[ii]+y[ii+1]) == 0):  # found the pair
            if x[ii] == x[ii+1]:
                z[ii+1] = 1
            else:
                z[ii+1] = x[ii]
                z[ii] = x[ii+1]
            break

    # finally x==00 and y doesn't
    for i in range(0, size(x)>>1):
        ii = 2*i
        if ((x[ii]+x[ii+1]) == 0) and ((y[ii]+y[ii+1]) != 0):  # found the pair
            if y[ii] == y[ii+1]:
                z[ii+1] = 1
            else:
                z[ii+1] = y[ii]
                z[ii] = y[ii+1]
            break
    output[0] = (x+z)%2
    output[1] = (y+z)%2
    return output
###################### end findtransvction

def symplectic(i, n): # output symplectic canonical matrix i of size 2nX2n
    """
    Note, compared to the text the transpose of the symplectic matrix
    is returned.  This is not particularly important since
                 Transpose(g in Sp(2n)) is in Sp(2n)
    but it means the program doesn't quite agree with the algorithm in the
    text. In python, row ordering of matrices is convenient, so it is used
    internally, but for column ordering is used in the text so that matrix
    multiplication of symplectics will correspond to conjugation by
    unitaries as conventionally defined Eq. (2).  We can't just return the
    transpose every time as this would alternate doing the incorrect thing
    as the algorithm recurses.
    """

    nn = 2*n   # this is convenient to have
    # step 1
    s = ((1<<nn)-1)
    k = (i%s)+1
    i //= s
#
    # step 2
    f1 = int2bits(k, nn)
#
    # step 3
    e1 = zeros(nn, dtype=int8) # define first basis vectors
    e1[0] = 1
    T = findtransvection(e1, f1) # use Lemma 2 to compute T
#
    # step 4
    # b[0]=b in the text, b[1]...b[2n-2] are b_3...b_2n in the text
    bits = int2bits(i%(1<<(nn-1)), nn-1)
#
    # step 5
    eprime = copy(e1)
    for j in range(2, nn):
        eprime[j] = bits[j-1]
    h0 = transvection(T[0], eprime)
    h0 = transvection(T[1], h0)
#
    # step 6
    if bits[0] == 1:
        f1 *= 0
     # T' from the text will be Z_f1 Z_h0.  If f1 has been set to zero
     #                                       it doesn't do anything
     # We could now compute f2 as said in the text but step 7 is slightly
     # changed and will recompute f1,f2 for us anyway
#
   # step 7
#  define the 2x2 identity matrix
    id2 = zeros((2, 2), dtype=int8)
    id2[0, 0] = 1
    id2[1, 1] = 1
#
    if n != 1:
        g = directsum(id2, symplectic(i>>(nn-1), n-1))
    else:
        g = id2
#
    for j in range(0, nn):
        g[j] = transvection(T[0], g[j])
        g[j] = transvection(T[1], g[j])
        g[j] = transvection(h0, g[j])
        g[j] = transvection(f1, g[j])
#
    return g
############# end symplectic

def bits2int(b, nn):
    """ converts an nn-bit string b to an integer between 0 and 2^n-1 """
    output = 0
    tmp = 1
    for j in range(0, nn):
        if b[j] == 1:
            output = output+tmp
        tmp = tmp*2
    return output

def numberofcosets(n):
    """ returns  the number of different cosets """
    x = power(2, 2*n-1)*(power(2, 2*n)-1)
    return x

def numberofsymplectic(n):
    """ returns the number of symplectic group elements """
    x = 1
    for j in range(1, n+1):
        x = x*numberofcosets(j)
    return x


def symplecticinverse(n, gn):
    """ produce an index associated with group element gn """

    nn = 2*n   # this is convenient to have

    # step 1
    v = gn[0]
    w = gn[1]

    # step 2
    e1 = zeros(nn, dtype=int8) # define first basis vectors
    e1[0] = 1
    T = findtransvection(v, e1)  # use Lemma 2 to compute T

    # step 3
    tw = copy(w)
    tw = transvection(T[0], tw)
    tw = transvection(T[1], tw)
    b = tw[0]
    h0 = zeros(nn, dtype=int8)
    h0[0] = 1
    h0[1] = 0
    for j in range(2, nn):
        h0[j] = tw[j]


    # step 4
    bb = zeros(nn-1, dtype=int8)
    bb[0] = b
    for j in range(2, nn):
        bb[j-1] = tw[j]
    zv = bits2int(v, nn)-1 # number between 0...2^(2n)-2
                           # indexing non-zero bitstring v of length 2n

    zw = bits2int(bb, nn-1)  # number between 0..2^(2n-1)-1
                             #indexing w (such that v,w is symplectic pair)
    cvw = zw*(power(2, 2*n)-1)+zv
    # cvw is a number indexing the unique coset specified by (v,w)
    # it is between 0...2^(2n-1)*(2^(2n)-1)-1=numberofcosets(n)-1


    #step 5
    if n == 1:
        return cvw

    #step 6
    gprime = copy(gn)
    if b == 0:
        for j in range(0, nn):
            gprime[j] = transvection(T[1], transvection(T[0], gn[j]))
            gprime[j] = transvection(h0, gprime[j])
            gprime[j] = transvection(e1, gprime[j])
    else:
        for j in range(0, nn):
            gprime[j] = transvection(T[1], transvection(T[0], gn[j]))
            gprime[j] = transvection(h0, gprime[j])

    # step 7
    gnew = gprime[2:nn, 2:nn]  # take submatrix
    gnidx = symplecticinverse(n-1, gnew)*numberofcosets(n)+cvw
    return gnidx
####### end symplecticinverse
