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

from random import Random
import numpy as np

from .clifford import Clifford
from .stabilizer_table import StabilizerTable
from .symplectic_utils import symplectic


def random_clifford(num_qubits, seed=None):
    """Return a random N-qubit Clifford operator.

    Args:
        num_qubits (int): the numbe of qubits for the Clifford.
        seed (int): Optional. To set a random seed.

    Returns:
        Clifford: the generated N-qubit clifford operator.

    Reference:
        1. R. Koenig, J.A. Smolin. *How to efficiently select an arbitrary
           Clifford group element*. J. Math. Phys. 55, 122202 (2014).
           `arXiv:1406.2170 [quant-ph] <https://arxiv.org/abs/1406.2170>`_
    """
    # Random number generator
    # We need to use Python random module instead of Numpy.random
    # as we are generating bigints
    rng = Random()
    rng.seed(seed)

    # The algorithm from Ref 1. generates a random Clifford by generating
    # a random symplectic matrix for the Clifford array, and a random
    # symplectic Pauli vector for the Phase.

    # Geneate random phase vector
    phase = np.array(
        [rng.randint(0, 1) for _ in range(2 * num_qubits)], dtype=np.bool)

    # Compute size of N-qubit sympletic group
    # this number will be a bigint if num_qubits > 5
    size = pow(2, num_qubits ** 2)
    for i in range(1, num_qubits+1):
        size *= pow(4, i) - 1

    # Sample a group element by index
    rint = rng.randrange(size)

    # Generate random element of symplectic group
    # TODO: Code needs to be optimized

    symp = symplectic(rint, num_qubits)
    symp2 = np.zeros([2 * num_qubits, 2 * num_qubits], dtype=np.uint8)
    symp3 = np.zeros([2 * num_qubits, 2 * num_qubits], dtype=np.uint8)

    # these interchange rows and columns because the random symplectic code
    #  uses a different convention
    for i in range(num_qubits):
        symp2[i] = symp[2 * i]
        symp2[i + num_qubits] = symp[2 * i + 1]
    for i in range(num_qubits):
        symp3[:, i] = symp2[:, 2 * i]
        symp3[:, i + num_qubits] = symp2[:, 2 * i + 1]

    return Clifford(StabilizerTable(symp3, phase))
