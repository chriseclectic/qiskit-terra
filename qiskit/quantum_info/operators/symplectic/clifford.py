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
Clifford operator class.
"""

import numpy as np

from qiskit import QiskitError
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from .stabilizer_table import StabilizerTable
from .clifford_circuits import append_gate, decompose_clifford


class Clifford(BaseOperator):
    """Clifford table operator class"""

    def __init__(self, data):
        """Initialize an operator object."""

        # Initialize from another Clifford by sharing the underlying
        # StabilizerTable
        if isinstance(data, Clifford):
            self._table = data._table

        # Initialize from ScalarOp as N-qubit identity discarding any global phase
        elif isinstance(data, ScalarOp):
            if not data.is_unitary() or set(data._input_dims) != set([2]):
                raise QiskitError("Can only initalize from N-qubit identity ScalarOp.")
            self._table = StabilizerTable(
                np.eye(2 * len(data._input_dims), dtype=np.bool))

        # Initialize from a QuantumCircuit or Instruction object
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._table = Clifford.from_instruction(data)._table

        # Initialize StabilizerTable directly from the data
        else:
            self._table = StabilizerTable(data)

        # Validate shape of StabilizerTable
        if self._table.size != 2 * self._table.num_qubits:
            raise QiskitError(
                'Invalid Clifford (number of rows {0} != {1}). An {2}-qubit'
                ' Clifford table requires {1} rows.'.format(
                    self._table.size, 2 * self._table.num_qubits, self.num_qubits))

        # TODO: Should we check the input array is a valid Clifford table?
        # This should be done by the `is_unitary` method.

        # Initialize BaseOperator
        dims = self._table.num_qubits * (2,)
        super().__init__(dims, dims)

    def __repr__(self):
        return 'Clifford({})'.format(repr(self.table))

    def __str__(self):
        return 'Clifford: Stabilizer = {}, Destabilizer = {}'.format(
            str(self.stabilizer.to_labels()),
            str(self.destabilizer.to_labels()))

    def __eq__(self, other):
        """Check if two Clifford tables are equal"""
        return super().__eq__(other) and self._table == other._table

    # ---------------------------------------------------------------------
    # Attributes
    # ---------------------------------------------------------------------
    def __getitem__(self, key):
        """Return a stabilizer Pauli row"""
        return self._table.__getitem__(key)

    def __setitem__(self, key, value):
        """Set a stabilizer Pauli row"""
        self._table.__setitem__(key, value)

    @property
    def table(self):
        """Return StabilizerTable"""
        return self._table

    @table.setter
    def table(self, value):
        """Set the stabilizer table"""
        # Note that is setup so it can't change the size of the Clifford
        # It can only replace the contents of the StabilizerTable with
        # another StabilizerTable of the same size.
        if not isinstance(value, StabilizerTable):
            value = StabilizerTable(value)
        self._table._array[:, :] = value._table._array
        self._table._phase[:] = value._table._phase

    @property
    def stabilizer(self):
        """Return the stabilizer block of the StabilizerTable."""
        return StabilizerTable(self._table[self.num_qubits:2*self.num_qubits])

    @stabilizer.setter
    def stabilizer(self, value):
        """Set the value of stabilizer block of the StabilizerTable"""
        inds = slice(self.num_qubits, 2*self.num_qubits)
        self._table.__setitem__(inds, value)

    @property
    def destabilizer(self):
        """Return the destabilizer block of the StabilizerTable."""
        return StabilizerTable(self._table[0:self.num_qubits])

    @destabilizer.setter
    def destabilizer(self, value):
        """Set the value of destabilizer block of the StabilizerTable"""
        inds = slice(0, self.num_qubits)
        self._table.__setitem__(inds, value)

    # ---------------------------------------------------------------------
    # Utility Operator methods
    # ---------------------------------------------------------------------

    def is_unitary(self, atol=None, rtol=None):
        """Return True if the Clifford table is valid."""
        # A valid Clifford is always unitary, so this function is really
        # checking that the underlying Stabilizer table array is a valid
        # Clifford array.

        # Condition is
        # table.T * [[0, 1], [1, 0]] * table = [[0, 1], [1, 0]]
        # where we are block matrix multiplying using symplectic product

        one = np.eye(self.num_qubits, dtype=int)
        zero = np.zeros((self.num_qubits, self.num_qubits), dtype=int)
        seye = np.block([[zero, one], [one, zero]])
        arr = self.table.array.astype(int)

        return np.array_equal(arr.T.dot(seye).dot(arr) % 2, seye)

    # ---------------------------------------------------------------------
    # BaseOperator Abstract Methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        """Return the conjugate of the Clifford."""
        x = self.table.X
        z = self.table.Z
        ret = self.copy()
        ret.table.phase = self.table.phase ^ (np.sum(x & z, axis=1) % 2)
        return ret

    def adjoint(self):
        """Return the conjugate transpose of the Clifford"""
        # TODO: Update to compute directly from table rather than circuit
        # decomposition
        return Clifford(self.to_circuit().inverse())

    def transpose(self):
        """Return the transpose of the Clifford."""
        return self.adjoint().conjugate()

    def compose(self, other, qargs=None, front=False):
        """Return the composed operator.

        Args:
            other (Clifford): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            Clifford: The operator self @ other.

        Raise:
            QiskitError: if operators have incompatible dimensions for
                         composition.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            matrix operators. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, Clifford):
            other = Clifford(other)

        # Validate dimensions. Note we don't need to get updated input or
        # output dimensions from `_get_compose_dims` as the dimensions of the
        # Clifford object can't be changed by composition
        self._get_compose_dims(other, qargs, front)

        if qargs is None or (
                len(qargs) == self.num_qubits and sorted(qargs) == qargs):
            return self._compose_clifford(other, front=front)

        return self._compose_subsystem(other, qargs, front=front)

    def dot(self, other, qargs=None):
        """Return the right multiplied operator self * other.

        Args:
            other (Clifford): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].

        Returns:
            Clifford: The operator self * other.

        Raises:
            QiskitError: if operators have incompatible dimensions for
                         composition.
        """
        return super().dot(other, qargs=qargs)

    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (Clifford): a operator subclass object.

        Returns:
            Clifford: the tensor product operator self ⊗ other.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (Clifford): an operator object.

        Returns:
            Clifford: the tensor product operator other ⊗ self.
        """
        return self._tensor_product(other, reverse=True)

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def to_dict(self):
        """Return dictionary represenation of Clifford object"""
        return {
            "stabilizer": self.stabilizer.to_labels(),
            "destabilizer": self.destabilizer.to_labels()
        }

    @staticmethod
    def from_dict(obj):
        """Load a Clifford from a dictionary"""
        destabilizer = StabilizerTable.from_labels(obj.get('destabilizer'))
        stabilizer = StabilizerTable.from_labels(obj.get('stabilizer'))
        return Clifford(destabilizer + stabilizer)

    def to_matrix(self):
        """Convert operator to Numpy matrix."""
        return self.to_operator().data

    def to_operator(self):
        """Convert to an Operator object."""
        return Operator(self.to_gate())

    def to_circuit(self):
        """Return a QuantumCircuit implementing the Clifford."""
        return decompose_clifford(self)

    def to_gate(self):
        """Return a Gate instruction implementing the Clifford."""
        return self.to_circuit().to_gate()

    @staticmethod
    def from_instruction(instruction):
        """Initialize from a QuantumCircuit or Instruction.

        Args:
            instruction (QuantumCircuit or Instruction): instruction to
                                                         initialize.

        Returns:
            Clifford: the Clifford object for the instruction.

        Raises:
            QiskitError: if the input instruction is non-Clifford or contains
                         classical register instruction.
        """
        if not isinstance(instruction, (QuantumCircuit, Instruction)):
            raise QiskitError("Input must be a QuantumCircuit or Instruction")

        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()

        # Initialize an identity Clifford
        clifford = Clifford(np.eye(2 * instruction.num_qubits))
        append_gate(clifford, instruction)
        return clifford

    # ---------------------------------------------------------------------
    # Internal tensor produce
    # ---------------------------------------------------------------------
    def _tensor_product(self, other, reverse=False):
        """Return the tensor product operator.

        Args:
            other (Clifford): another Clifford operator.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            Clifford: the tensor product operator.

        Raises:
            QiskitError: if other cannot be converted into an Clifford.
        """
        if not isinstance(other, Clifford):
            other = Clifford(other)

        if reverse:
            cliff0 = self
            cliff1 = other
        else:
            cliff0 = other
            cliff1 = self

        # Pad stabilizers and destabilizers
        destab = (cliff0.destabilizer.expand(cliff1.num_qubits * 'I') +
                  cliff1.destabilizer.tensor(cliff0.num_qubits * 'I'))
        stab = (cliff0.stabilizer.expand(cliff1.num_qubits * 'I') +
                cliff1.stabilizer.tensor(cliff0.num_qubits * 'I'))

        # Add the padded table
        table = destab + stab
        return Clifford(table)

    # ---------------------------------------------------------------------
    # Internal composition methods
    # ---------------------------------------------------------------------
    def _compose_subsystem(self, other, qargs, front=False):
        """Return the composition channel."""
        # Create Clifford on full system from subsystem and compose
        nq = self.num_qubits
        no = other.num_qubits
        fullother = self.copy()
        fullother.table.array = np.eye(2 * self.num_qubits, dtype=np.bool)
        for inda, qinda in enumerate(qargs):
            for indb, qindb in enumerate(qargs):
                fullother.table._array[nq - 1 - qinda, nq - 1 - qindb] = other.table._array[
                    no - 1 - inda, no - 1 - indb]
                fullother.table._array[nq - 1 - qinda, 2*nq - 1 - qindb] = other.table._array[
                    no - 1 - inda, 2*no - 1 - indb]
                fullother.table._array[2*nq - 1 - qinda, nq - 1 - qindb] = other.table._array[
                    2*no - 1 - inda, no - 1 - indb]
                fullother.table._array[2*nq - 1 - qinda, 2*nq - 1 - qindb] = other.table._array[
                    2*no - 1 - inda, 2*no - 1 - indb]
                fullother.table._phase[nq - 1 - qinda] = other.table._phase[no - 1 - inda]
                fullother.table._phase[2*nq - 1 - qinda] = other.table._phase[2*no - 1 - inda]
        return self._compose_clifford(fullother, front=front)

    def _compose_clifford(self, other, front=False):
        """Return the composition channel assume other is Clifford of same size as self."""
        if front:
            table1 = self.table
            table2 = other.table
        else:
            table1 = other.table
            table2 = self.table

        num_qubits = self.num_qubits

        array1 = table1.array.astype(int)
        phase1 = table1.phase.astype(int)

        array2 = table2.array.astype(int)
        phase2 = table2.phase.astype(int)

        # Update Pauli table
        pauli = StabilizerTable(array2.dot(array1) % 2)

        # Add phases
        phase = np.mod(array2.dot(phase1) + phase2, 2)

        # Correcting for phase due to Pauli multiplicatio
        ifacts = np.zeros(2 * num_qubits, dtype=np.int)

        for k in range(2 * num_qubits):

            row2 = array2[k]
            x2 = table2.X[k]
            z2 = table2.Z[k]

            # Adding a factor of i for each Y in the image of an operator under the
            # first operation, since Y=iXZ

            ifacts[k] += np.sum(x2 & z2)

            # Adding factors of i due to qubit-wise Pauli multiplication

            for j in range(num_qubits):
                x = 0
                z = 0
                for i in range(2 * num_qubits):
                    if row2[i]:
                        x1 = array1[i, j]
                        z1 = array1[i, j + num_qubits]
                        if (x | z) & (x1 | z1):
                            val = np.mod(np.abs(3 * z1 - x1) - np.abs(3 * z - x) - 1, 3)
                            if val == 0:
                                ifacts[k] += 1
                            elif val == 1:
                                ifacts[k] -= 1
                        x = np.mod(x + x1, 2)
                        z = np.mod(z + z1, 2)

        p = np.mod(ifacts, 4) // 2

        phase = np.mod(phase + p, 2)

        return Clifford(StabilizerTable(pauli, phase))
