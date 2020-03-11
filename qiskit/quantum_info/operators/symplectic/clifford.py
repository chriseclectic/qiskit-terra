# -*- coding: utf-8 -*-

# Copyright 2017, 2020 BM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Clifford operator class.
"""

import numpy as np

from qiskit import QiskitError
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.stabilizer_table import StabilizerTable
from qiskit.quantum_info.operators.symplectic.clifford_append_gate import append_gate


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
        if self._table.size != 2 * self._table.n_qubits:
            raise QiskitError(
                'Invalid Clifford (number of rows {0} != {1}). An {2}-qubit'
                ' Clifford table requires {1} rows.'.format(
                    self._table.size, 2 * self._table.n_qubits, self.n_qubits))

        # TODO: Should we check the input array is a valid Clifford table?
        # This should be done by the `is_unitary` method.

        # Initialize BaseOperator
        dims = self._table.n_qubits * (2,)
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
    def n_qubits(self):
        """The number of qubits for the Clifford."""
        return self._table._n_qubits

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
        return StabilizerTable(self._table[self.n_qubits:2*self.n_qubits])

    @stabilizer.setter
    def stabilizer(self, value):
        """Set the value of stabilizer block of the StabilizerTable"""
        inds = slice(self.n_qubits, 2*self.n_qubits)
        self._table.__setitem__(inds, value)

    @property
    def destabilizer(self):
        """Return the destabilizer block of the StabilizerTable."""
        return StabilizerTable(self._table[0:self.n_qubits])

    @destabilizer.setter
    def destabilizer(self, value):
        """Set the value of destabilizer block of the StabilizerTable"""
        inds = slice(0, self.n_qubits)
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

        one = np.eye(self.n_qubits, dtype=int)
        zero = np.zeros((self.n_qubits, self.n_qubits), dtype=int)
        seye = np.block([[zero, one], [one, zero]])
        arr = self.table.array.astype(int)

        return np.array_equal(arr.T.dot(seye).dot(arr) % 2, seye)

    def to_matrix(self):
        """Convert operator to Numpy matrix."""

        # TODO: IMPLEMENT ME!

        raise NotImplementedError(
            'This method has not been implemented for Clifford operators yet.')

    def to_operator(self):
        """Convert to an Operator object."""

        # TODO: IMPLEMENT ME!

        raise NotImplementedError(
            'This method has not been implemented for Clifford operators yet.')

    # ---------------------------------------------------------------------
    # BaseOperator Abstract Methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        """Return the conjugate of the Clifford."""
        # TODO: Needs testing to see if correct
        x = self.table.X
        z = self.table.Z
        ret = self.copy()
        ret.table.phase = self.table.phase ^ (np.sum(x & z, axis=1) % 2)
        return ret

    def transpose(self):
        """Return the transpose of the Clifford."""

        # TODO: Needs testing to see if correct
        # This is done using block matrix multiplication
        # [[0, 1], [1, 0]] * table.T * [[0, 1], [1, 0]]

        ret = self.copy()
        tmp = ret.destabilizer.X.copy()
        ret.destabilizer.X = ret.stabilizer.Z
        ret.destabilizer.Z = ret.destabilizer.Z.T
        ret.stabilizer.X = ret.stabilizer.X.T
        ret.stabilizer.Z = tmp
        return ret

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

        if qargs is None:
            return self._compose_clifford(other, front)
        else:
            return self._compose_subsystem(other, qargs, front=False)

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
            first = other
            second = self
        else:
            first = self
            second = other
        n_first = first.n_qubits
        n_second = second.n_qubits

        # Pad stabilizers and destabilizers
        destab = (first.destabilizer.tensor(n_second * 'I') +
                  second.destabilizer.expand(n_first * 'I'))
        stab = (first.stabilizer.tensor(n_second * 'I') +
                second.stabilizer.expand(n_first * 'I'))

        # Add the padded table
        table = destab + stab
        return Clifford(table)

    # ---------------------------------------------------------------------
    # Internal composition methods
    # ---------------------------------------------------------------------

    def _compose_subsystem(self, other, qargs, front=False):
        """Return the composition channel."""
        # Create Clifford on full system from subsystem and compose
        nq = self.n_qubits
        no = other.n_qubits
        fullother = self.copy()
        fullother.reset()
        for inda, qubita in enumerate(qargs):
            qinda = qubita.index
            for indb, qubitb in enumerate(qargs):
                qindb = qubitb.index
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
        return self._compose_clifford(fullother, front)

    def _compose_clifford(self, other, front=False):
        """Return the composition channel assume other is Clifford of same size as self."""
        if front:
            clifffront = self
            cliffback = other
        else:
            clifffront = other
            cliffback = self
        output = cliffback.copy()

        farray = clifffront.table._array
        barray = cliffback.table._array
        oarray = output.table._array
        fphase = clifffront.table._phase
        ophase = output.table._phase
        # Zero the array, leave the phases in place
        oarray *= False
        for oind, orow in enumerate(oarray):
            for find, frow in enumerate(farray):
                if barray[oind][find]:
                    # Edit the row in place, return the phase
                    ophase[oind] = self._rowsum_AG(orow, frow, ophase[oind] ^ fphase[find])

        return output

    def _rowsum_AG(self, orow, frow, phase):
        # I've never understood rowsum, here's an easier way
        nq = len(orow)//2
        for ind in range(nq):
            phase = phase ^ self._g_AG(orow[ind], orow[nq+ind], frow[ind], frow[nq+ind])
        np.logical_xor(orow, frow, out=orow)

        return phase

    def _g_AG(self, x1, z1, x2, z2):
        return ((not x1 and z1 and x2 and not z2)
                or (x1 and not z1 and x2 and z2)
                or (x1 and z1 and not x2 and z2))
