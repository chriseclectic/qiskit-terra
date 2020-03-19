# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

import numpy as np
import random

from qiskit.quantum_info.operators import Clifford
import qiskit.quantum_info.operators.symplectic.symplectic_utils as symp_utils

# ---------------------------------------------------------------------
# Random Clifford Circuit
# From "How to efficiently select an arbitrary clifford group element"
#       by Robert Koenig and John A. Smolin
# ---------------------------------------------------------------------

def rand_symplectic_part(symp, num_qubits):
    """ helper function to generate the symplectic part of the random clifford """
    symp2 = np.zeros([2*num_qubits, 2*num_qubits], dtype=np.uint8)
    symp3 = np.zeros([2*num_qubits, 2*num_qubits], dtype=np.uint8)
    # these interchange rows and columns because the random symplectic code
    # uses a different convention
    for i in range(num_qubits):
        symp2[i] = symp[2*i]
        symp2[i+num_qubits] = symp[2*i+1]
    for i in range(num_qubits):
        symp3[:, i] = symp2[:, 2*i]
        symp3[:, i+num_qubits] = symp2[:, 2*i+1]

    mat = np.zeros([2*num_qubits, 2*num_qubits], dtype=np.uint8)
    mat[0:2*num_qubits, 0:2*num_qubits] = symp3
    return(mat)

def random_clifford(num_qubits, seed=None):
    """pick a random Clifford gate on num_qubits"""
    # compute size of num_qubits-qubit sympletic group
    random.seed(seed)
    size = 1
    for i in range(1, num_qubits+1):
        size = size*(pow(4, i)-1)
    size = size*pow(2, num_qubits*num_qubits)
    rint = random.randrange(size)
    symp = symp_utils.symplectic(rint, num_qubits)
    array = rand_symplectic_part(symp, num_qubits)
    phase = [random.randint(0,1) for _ in range(2*num_qubits)]
    cliff = Clifford(np.eye(2*num_qubits))
    cliff.table.array = array
    cliff.table.phase = phase
    return(cliff)
