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

"""
CSR Matrix operator class.
"""
# pylint: disable=no-name-in-module

import copy
import re
from numbers import Number
from functools import wraps
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate, HGate, SGate, TGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.mixins import generate_apidocs


# Sparse Matrix
import scipy.sparse as sps

try:
    from qutip.fastsparse import fast_csr_matrix
    from qutip.cy.spmath import zcsr_kron

    _HAS_FAST_CSR = True
except ModuleNotFoundError:
    _HAS_FAST_CSR = False


def to_csr(obj):
    """Convert to CSR matrix using fast_csr_matrix if available."""
    if _HAS_FAST_CSR and isinstance(obj, fast_csr_matrix):
        return obj

    if isinstance(obj, sps.spmatrix):
        obj = obj.tocsr()
    else:
        obj = sps.csr_matrix(obj)

    if obj.dtype != complex:
        obj = obj.astype(complex)
    if _HAS_FAST_CSR:
        obj = fast_csr_matrix((obj.data, obj.indices, obj.indptr), shape=obj.shape)
    return obj


class SparseOp(LinearOp):
    r"""Sparse CSR matrix operator class

    This represents a matrix operator :math:`M` that will
    :meth:`~Statevector.evolve` a :class:`Statevector` :math:`|\psi\rangle`
    by matrix-vector multiplication

    .. math::

        |\psi\rangle \mapsto M|\psi\rangle,

    and will :meth:`~DensityMatrix.evolve` a :class:`DensityMatrix` :math:`\rho`
    by left and right multiplication

    .. math::

        \rho \mapsto M \rho M^\dagger.

    .. note::

        If the `QuTiP <https://github.com/qutip/qutip>`_ package is installed
        this operator will make use of the
        ``qutip.fastsparse.fast_csr_matrix`` class, otherwise it use the SciPy
        ``scipy.sparse.csr_matrix`` class for its CSR matrix.
    """

    _LABEL_MATS = {
        "I": to_csr(IGate().to_matrix()),
        "X": to_csr(XGate().to_matrix()),
        "Y": to_csr(YGate().to_matrix()),
        "Z": to_csr(ZGate().to_matrix()),
        "H": to_csr(HGate().to_matrix()),
        "S": to_csr(SGate().to_matrix()),
        "T": to_csr(TGate().to_matrix()),
        "0": to_csr(np.array([[1, 0], [0, 0]], dtype=complex)),
        "1": to_csr(np.array([[0, 0], [0, 1]], dtype=complex)),
        "+": to_csr(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)),
        "-": to_csr(np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex)),
        "r": to_csr(np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex)),
        "l": to_csr(np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex)),
    }

    _LABEL_DIAGS = {
        "I": np.array([1, 1], dtype=complex),
        "Z": np.array([1, -1], dtype=complex),
        "0": np.array([1, 0], dtype=complex),
        "1": np.array([0, 1], dtype=complex),
    }

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize an operator object.

        Args:
            data (QuantumCircuit or
                  Instruction or
                  Operator or
                  matrix): data to initialize operator.
            input_dims (tuple): the input subsystem dimensions.
                                [Default: None]
            output_dims (tuple): the output subsystem dimensions.
                                 [Default: None]

        Raises:
            QiskitError: if input data cannot be initialized as an operator.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. If the input data is
            a Numpy array of shape (2**N, 2**N) qubit systems will be used. If
            the input operator is not an N-qubit operator, it will assign a
            single subsystem with dimension specified by the shape of the input.
        """
        op_shape = None
        if isinstance(data, (sps.spmatrix, np.ndarray, list)):
            self._data = to_csr(data)
        elif isinstance(data, Pauli):
            self._data = to_csr(data.to_matrix(sparse=True))
        elif isinstance(data, ScalarOp):
            self._data = data.coeff * csr_eye(data.dim)
        elif isinstance(data, (QuantumCircuit, Instruction)):
            # If the input is a Terra QuantumCircuit or Instruction we
            # perform a simulation to construct the unitary operator.
            # This will only work if the circuit or instruction can be
            # defined in terms of unitary gate instructions which have a
            # 'to_matrix' method defined. Any other instructions such as
            # conditional gates, measure, or reset will cause an
            # exception to be raised.
            self._data = self._init_instruction(data).data
        else:
            # Try converting to Operator
            try:
                data = Operator(data, input_dims=input_dims, output_dims=output_dims)
                op_shape = data._op_shape
                self._data = to_csr(data.data)
                if input_dims is None:
                    input_dims = data.input_dims()
                if output_dims is None:
                    output_dims = data.output_dims()
            except QiskitError:
                raise QiskitError("Invalid input data format for SparseOp")

        super().__init__(
            op_shape=op_shape,
            input_dims=input_dims,
            output_dims=output_dims,
            shape=self._data.shape,
        )

    def __array__(self, dtype=None):
        mat = self.data.todense()
        if dtype:
            return np.asarray(mat, dtype=dtype)
        return mat

    def __repr__(self):
        prefix = "SparseOp("
        pad = " " * len(prefix)
        return "{}{},\n{}input_dims={}, output_dims={})".format(
            prefix, repr(self.data), pad, self.input_dims(), self.output_dims()
        )

    def __eq__(self, other):
        """Test if two SparseOps are equal."""
        if not super().__eq__(other):
            return False
        return np.allclose(self.data - other.data, 0, rtol=self.rtol, atol=self.atol)

    @property
    def data(self):
        """Return data."""
        return self._data

    @classmethod
    def from_label(cls, label):
        """Return a tensor product of single-qubit operators.

        Args:
            label (string): single-qubit operator string.

        Returns:
            SparseOp: The N-qubit operator.

        Raises:
            QiskitError: if the label contains invalid characters, or the
                         length of the label is larger than an explicitly
                         specified num_qubits.

        Additional Information:
            The labels correspond to the single-qubit matrices:
            'I': [[1, 0], [0, 1]]
            'X': [[0, 1], [1, 0]]
            'Y': [[0, -1j], [1j, 0]]
            'Z': [[1, 0], [0, -1]]
            'H': [[1, 1], [1, -1]] / sqrt(2)
            'S': [[1, 0], [0 , 1j]]
            'T': [[1, 0], [0, (1+1j) / sqrt(2)]]
            '0': [[1, 0], [0, 0]]
            '1': [[0, 0], [0, 1]]
            '+': [[0.5, 0.5], [0.5 , 0.5]]
            '-': [[0.5, -0.5], [-0.5 , 0.5]]
            'r': [[0.5, -0.5j], [0.5j , 0.5]]
            'l': [[0.5, 0.5j], [-0.5j , 0.5]]
        """
        num_qubits = len(label)

        # Check if Pauli label:
        if re.match(r"^[IXYZ]+$", label):
            return SparseOp(Pauli(label))

        # Check if diagonal op
        if re.match(r"^[IZ01]+$", label):
            dim = 2 ** num_qubits
            indices = np.arange(dim, dtype=np.int32)
            indptr = np.arange(dim + 1, dtype=np.int32)
            diag = np.array(1, dtype=complex)
            for char in label:
                diag = np.kron(SparseOp._LABEL_DIAGS[char], diag)
            if _HAS_FAST_CSR:
                csr = fast_csr_matrix((diag, indices, indptr), shape=(dim, dim))
            else:
                csr = sps.csr_matrix((diag, indices, indptr), shape=(dim, dim))
            return SparseOp(csr)

        # Check general Operator label
        if re.match(r"^[IXYZHST01rl\-+]+$", label) is None:
            raise QiskitError("Label contains invalid characters.")
        # Initialize an identity matrix and apply each gate
        num_qubits = len(label)
        op = SparseOp(csr_eye(2 ** num_qubits))
        for qubit, char in enumerate(reversed(label)):
            if char != "I":
                op = op.compose(SparseOp._LABEL_MATS[char], qargs=[qubit])
        return op

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        din, dout = self.dim
        if din < dout:
            delta = self.compose(self.adjoint()) - csr_eye(din)
        else:
            delta = self.dot(self.adjoint()) - csr_eye(dout)
        return np.allclose(delta._data.data, 0, atol=atol, rtol=rtol)

    def to_instruction(self):
        """Convert to a UnitaryGate instruction."""
        # pylint: disable=cyclic-import
        from qiskit.extensions.unitary import UnitaryGate

        return UnitaryGate(self._data.todense())

    def conjugate(self):
        # Make a shallow copy and update array
        ret = copy.copy(self)
        ret._data = self._data.conj()
        return ret

    def transpose(self):
        # Make a shallow copy and update array
        ret = copy.copy(self)
        ret._data = self._data.T
        ret._op_shape = self._op_shape.transpose()
        return ret

    def compose(self, other, qargs=None, front=False):
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, SparseOp):
            other = SparseOp(other)

        new_shape = self._op_shape.compose(other._op_shape, qargs, front)
        input_dims = new_shape.dims_r()
        output_dims = new_shape.dims_l()

        # Full composition of operators
        csr_a = self._data
        if qargs is None:
            csr_b = other._data
        else:
            if front:
                dims = self.input_dims()
            else:
                dims = self.output_dims()
            csr_b = self._pad_with_identity(dims, other, qargs)

        if front:
            # Composition self * other
            data = csr_a.dot(csr_b)
        else:
            # Composition other * self
            data = csr_b.dot(csr_a)
        ret = SparseOp(data, input_dims, output_dims)
        ret._op_shape = new_shape
        return ret

    def power(self, n):
        """Return the matrix power of the operator.

        Args:
            n (float): the power to raise the matrix to.

        Returns:
            SparseOp: the n-times composed operator.

        Raises:
            QiskitError: if the input and output dimensions of the operator
                         are not equal.
        """
        if not isinstance(n, int):
            raise QiskitError("Can only take integer powers of SparseOp.")
        if self.input_dims() != self.output_dims():
            raise QiskitError("Can only power with input_dims = output_dims.")
        # Override base class power so we can implement more efficiently
        # using Numpy.matrix_power
        ret = copy.copy(self)
        ret._data = self.data ** n
        return ret

    def tensor(self, other):
        if not isinstance(other, SparseOp):
            other = SparseOp(other)
        ret = copy.copy(self)
        ret._data = sps_kron(self._data, other._data)
        ret._op_shape = self._op_shape.tensor(other._op_shape)
        return ret

    def expand(self, other):
        if not isinstance(other, SparseOp):
            other = SparseOp(other)
        ret = copy.copy(self)
        ret._data = sps_kron(other._data, self._data)
        ret._op_shape = other._op_shape.tensor(self._op_shape)
        return ret

    def _add(self, other, qargs=None):
        """Return the operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (SparseOp): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            SparseOp: the operator self + other.

        Raises:
            QiskitError: if other is not an operator, or has incompatible
                         dimensions.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, SparseOp):
            other = SparseOp(other)

        self._op_shape._validate_add(other._op_shape, qargs)
        csr_other = self._pad_with_identity(self.input_dims(), other, qargs)
        ret = copy.copy(self)
        ret._data = self.data + csr_other
        return ret

    def _multiply(self, other):
        """Return the operator self * other.

        Args:
            other (complex): a complex number.

        Returns:
            SparseOp: the operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        ret = copy.copy(self)
        ret._data = other * self._data
        return ret

    @staticmethod
    def _pad_with_identity(dims, other, qargs=None):
        # Trivial case for qargs is None
        if qargs is None:
            return other.data

        # Format inputs
        qargs = tuple(qargs)
        subdims = tuple(dims)

        # Compute total and block dimension
        dim = np.product(subdims)
        dblock = 1
        for i in qargs:
            dblock *= dims[i]

        if other.data.shape != (dblock, dblock):
            raise QiskitError("Invalid qargs for matrix shape.")

        # Can we skip this conversion if mat is already sparse?
        mat = other.data.todense()
        swap_vec = _build_swap_vec(dim, subdims, qargs)
        data, row, col = _build_subsystem_coo_data(mat, swap_vec, dim, dblock)

        # Use custom conversion function to CSR format
        data, indices, indptr = _build_csr_from_coo(data, row, col, dim, dblock)
        if _HAS_FAST_CSR and data.dtype == complex:
            return fast_csr_matrix((data, indices, indptr), (dim, dim))
        return sps.csr_matrix((data, indices, indptr), (dim, dim))

    @classmethod
    def _init_instruction(cls, instruction):
        """Convert a QuantumCircuit or Instruction to a SparseOp."""
        # Initialize an identity operator of the correct size of the circuit
        dimension = 2 ** instruction.num_qubits
        op = SparseOp(csr_eye(dimension))
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        op._append_instruction(instruction)
        return op

    @classmethod
    def _instruction_to_matrix(cls, obj):
        """Return SparseOp for instruction if defined or None otherwise."""
        if not isinstance(obj, Instruction):
            raise QiskitError("Input is not an instruction.")
        mat = None
        if hasattr(obj, "to_matrix"):
            # If instruction is a gate first we see if it has a
            # `to_matrix` definition and if so use that.
            try:
                mat = obj.to_matrix()
            except QiskitError:
                pass
        return mat

    def _append_instruction(self, obj, qargs=None):
        """Update the current SparseOp by apply an instruction."""
        from qiskit.circuit.barrier import Barrier

        mat = self._instruction_to_matrix(obj)
        if mat is not None:
            # Perform the composition and inplace update the current state
            # of the operator
            op = self.compose(mat, qargs=qargs)
            self._data = op.data
        elif isinstance(obj, Barrier):
            return
        else:
            # If the instruction doesn't have a matrix defined we use its
            # circuit decomposition definition if it exists, otherwise we
            # cannot compose this gate and raise an error.
            if obj.definition is None:
                raise QiskitError("Cannot apply Instruction: {}".format(obj.name))
            if not isinstance(obj.definition, QuantumCircuit):
                raise QiskitError(
                    'Instruction "{}" '
                    "definition is {} but expected QuantumCircuit.".format(
                        obj.name, type(obj.definition)
                    )
                )
            if obj.definition.global_phase:
                dimension = 2 ** self.num_qubits
                op = self.compose(
                    ScalarOp(dimension, np.exp(1j * float(obj.definition.global_phase))),
                    qargs=qargs,
                )
                self._data = op.data
            flat_instr = obj.definition.to_instruction()
            for instr, qregs, cregs in flat_instr.definition.data:
                if cregs:
                    raise QiskitError(
                        "Cannot apply instruction with classical registers: {}".format(instr.name)
                    )
                # Get the integer position of the flat register
                if qargs is None:
                    new_qargs = [tup.index for tup in qregs]
                else:
                    new_qargs = [qargs[tup.index] for tup in qregs]
                self._append_instruction(instr, qargs=new_qargs)


# Update docstrings for API docs
generate_apidocs(SparseOp)


# Sparse Utils
@wraps(sps.kron)
def sps_kron(a, b):
    if _HAS_FAST_CSR and a.dtype == complex and b.dtype == complex:
        return zcsr_kron(a, b)
    return sps.kron(a, b)


def csr_eye(dim, dtype=complex):
    """Return CSR identity matrix"""
    data = np.ones(dim, dtype=dtype)
    indices = np.arange(dim, dtype=np.int32)
    indptr = np.arange(dim + 1, dtype=np.int32)
    if _HAS_FAST_CSR and data.dtype == complex:
        return fast_csr_matrix((data, indices, indptr))
    return sps.csr_matrix((data, indices, indptr))


def _get_pattern(num_subsys, qargs):
    """Return permutation pattern for remapping qargs"""
    num_qargs = len(qargs)

    subspace = set(range(num_qargs)).union(qargs)
    trivial = set(i for i in range(num_qargs, num_subsys) if i not in qargs)
    out = np.arange(num_subsys)
    for i in range(num_qargs):
        out[i] = qargs[i]
        subspace.remove(qargs[i])

    for i in range(num_qargs, num_subsys):
        if i not in trivial:
            out[i] = subspace.pop()
    return out


def _build_swap_vec(dim, subdims, qargs):
    num_subsys = len(subdims)
    pattern = _get_pattern(num_subsys, qargs)
    axes = [num_subsys - 1 - i for i in reversed(pattern)]
    ret = np.arange(dim).reshape(list(reversed(subdims)))
    ret = ret.transpose(axes).flatten()
    return ret


def _build_subsystem_coo_data(mat, swap_vec, dim, dblock):
    # TODO: convert to cython function
    if dim > 2 ** 31:
        raise Exception("Combined dimension exceeds max size allowed by 32-bit array indexes.")
    nblock = dim // dblock
    data = np.empty(nblock * mat.size, dtype=mat.dtype)
    row = np.empty(nblock * mat.size, dtype=np.int32)
    col = np.empty(nblock * mat.size, dtype=np.int32)

    idx = 0
    for k in range(nblock):
        for i in range(dblock):
            for j in range(dblock):
                data[idx] = mat[i, j]
                row[idx] = swap_vec[k * dblock + i]
                col[idx] = swap_vec[k * dblock + j]
                idx += 1
    return data, row, col


def _build_csr_from_coo(data, row, col, dim, dblock):
    # TODO: convert to cython function
    indptr = np.arange(0, dblock * dim + 1, dblock, dtype=np.int32)
    indices = np.empty(data.size, dtype=np.int32)
    csrdata = np.empty(data.size, dtype=data.dtype)

    tmp = indptr.copy()
    for k in range(data.size):
        i = row[k]
        j = tmp[i]
        indices[j] = col[k]
        csrdata[j] = data[k]
        tmp[i] += 1

    return csrdata, indices, indptr
