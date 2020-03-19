# -*- coding: utf-8 -*-

# Copyright 2017, 2020 BM.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Symplectic matrix group utilities.
"""

import numpy as np


# NOTE: Following code needs to be heavily optimized and should probably
# be moved to a symplectic_utils.py file.

def symplectic(i, n):
    """Output symplectic canonical matrix i of size 2nX2n

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

    nn = 2 * n  # this is convenient to have
    # Step 1
    s = ((1 << nn) - 1)
    k = (i % s) + 1
    i //= s

    # Step 2
    f1 = int2bits(k, nn)

    # Step 3
    e1 = np.zeros(nn, dtype=np.int8)  # define first basis vectors
    e1[0] = 1
    T = findtransvection(e1, f1)  # use Lemma 2 to compute T

    # Step 4
    # b[0]=b in the text, b[1]...b[2n-2] are b_3...b_2n in the text
    bits = int2bits(i % (1 << (nn - 1)), nn - 1)

    # Step 5
    eprime = np.copy(e1)
    for j in range(2, nn):
        eprime[j] = bits[j - 1]
    h0 = transvection(T[0], eprime)
    h0 = transvection(T[1], h0)

    # Step 6
    if bits[0] == 1:
        f1 *= 0
    # T' from the text will be Z_f1 Z_h0.  If f1 has been set to zero
    #                                       it doesn't do anything
    # We could now compute f2 as said in the text but step 7 is slightly
    # changed and will recompute f1,f2 for us anyway

    # Step 7
    # define the 2x2 identity matrix
    id2 = np.zeros((2, 2), dtype=np.int8)
    id2[0, 0] = 1
    id2[1, 1] = 1

    if n != 1:
        g = directsum(id2, symplectic(i >> (nn - 1), n - 1))
    else:
        g = id2

    for j in range(0, nn):
        g[j] = transvection(T[0], g[j])
        g[j] = transvection(T[1], g[j])
        g[j] = transvection(h0, g[j])
        g[j] = transvection(f1, g[j])

    return g


def directsum(m1, m2):
    """ direct sum """
    rows1, cols1 = np.shape(m1)
    rows2, cols2 = np.shape(m2)
    ret = np.zeros((rows1 + rows2, cols1 + cols2), dtype=m1.dtype)
    ret[0:rows1, 0:cols1] = m1.copy()
    ret[rows1:rows1+rows2, cols1:cols1+cols2] = m2.copy()
    return ret


def inner(v, w):
    """ symplectic inner product """
    # TODO: UPDATE
    t = 0
    for i in range(0, np.size(v) >> 1):
        t += v[2 * i] * w[2 * i + 1]
        t += w[2 * i] * v[2 * i + 1]
    return t % 2


def transvection(k, v):
    """ applies transvection Z_k to v """
    return (v + inner(k, v) * k) % 2


def int2bits(i, n):
    """ converts integer i to an length n array of bits """
    # TODO: UPDATE
    output = np.zeros(n, dtype=np.int8)
    for j in range(0, n):
        output[j] = i & 1
        i >>= 1
    return output


def findtransvection(x, y):
    """
    finds h1,h2 such that y = Z_h1 Z_h2 x
    Lemma 2 in the text
    Note that if only one transvection is required output[1] will be
    zero and applying the all-zero transvection does nothing.
    """
    # TODO: UPDATE
    output = np.zeros((2, np.size(x)), dtype=np.int8)
    if np.array_equal(x, y):
        return output
    if inner(x, y) == 1:
        output[0] = (x + y) % 2
        return output

    # find a pair where they are both not 00
    z = np.zeros(np.size(x))
    for i in range(0, np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) != 0) and ((y[ii] + y[ii + 1]) != 0):
            # found the pair
            z[ii] = (x[ii] + y[ii]) % 2
            z[ii + 1] = (x[ii + 1] + y[ii + 1]) % 2
            if (z[ii] + z[ii + 1]) == 0:
                # they were the same so they added to 00
                z[ii + 1] = 1
                if x[ii] != x[ii + 1]:
                    z[ii] = 1
            output[0] = (x + z) % 2
            output[1] = (y + z) % 2
            return output
    # didn't find a pair
    # so look for two places where x has 00 and y doesn't, and vice versa

    # first y==00 and x doesn't
    for i in range(0, np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) != 0) and ((y[ii] +
                                            y[ii + 1]) == 0):  # found the pair
            if x[ii] == x[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = x[ii]
                z[ii] = x[ii + 1]
            break

    # finally x==00 and y doesn't
    for i in range(0, np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] +
             x[ii + 1]) == 0) and ((y[ii] + y[ii + 1]) != 0):  # found the pair
            if y[ii] == y[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = y[ii]
                z[ii] = y[ii + 1]
            break
    output[0] = (x + z) % 2
    output[1] = (y + z) % 2
    return output
