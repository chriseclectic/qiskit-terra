# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
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
import re

import numpy as np

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library.standard_gates import (
    HGate,
    IGate,
    SGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.utils.deprecation import deprecate_function

from .clifford_circuits import _append_circuit
from .pauli_list import PauliList
from .stabilizer_table import StabilizerTable


class Clifford(BaseOperator, AdjointMixin):
    """An N-qubit unitary operator from the Clifford group.

    **Representation**

    An *N*-qubit Clifford operator is stored as a length *2N*
    :class:`~qiskit.quantum_info.PauliList` using the convention
    from reference [1].

    * Rows 0 to *N-1* are the *destabilizer* group generators
    * Rows *N* to *2N-1* are the *stabilizer* group generators.

    The internal :class:`~qiskit.quantum_info.PauliList` for the Clifford
    can be accessed using the :attr:`paulis` attribute. The destabilizer or
    stabilizer rows can each be accessed as a length-N Stabilizer Pauli list using
    :meth:`stabilizers` and :meth:`destabiliers` respectively.

    A more easily human readable representation of the Clifford operator can
    be obtained by calling the :meth:`to_dict` method. This representation is
    also used if a Clifford object is printed as in the following example

    .. jupyter-execute::

        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Clifford

        # Bell state generation circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        cliff = Clifford(qc)

        # Print the Clifford
        print(cliff)

        # Print the PauliList in the Clifford
        print(cliff.paulis)


    **Circuit Conversion**

    Clifford operators can be initialized from circuits containing *only* the
    following Clifford gates: :class:`~qiskit.circuit.library.IGate`,
    :class:`~qiskit.circuit.library.XGate`, :class:`~qiskit.circuit.library.YGate`,
    :class:`~qiskit.circuit.library.ZGate`, :class:`~qiskit.circuit.library.HGate`,
    :class:`~qiskit.circuit.library.SGate`, :class:`~qiskit.circuit.library.SdgGate`,
    :class:`~qiskit.circuit.library.CXGate`, :class:`~qiskit.circuit.library.CZGate`,
    :class:`~qiskit.circuit.library.SwapGate`.
    They can be converted back into a :class:`~qiskit.circuit.QuantumCircuit`,
    or :class:`~qiskit.circuit.Gate` object using the :meth:`~Clifford.to_circuit`
    or :meth:`~Clifford.to_instruction` methods respectively. Note that this
    decomposition is not necessarily optimal in terms of number of gates.

    .. note::

        A minimally generating set of gates for Clifford circuits is
        the :class:`~qiskit.circuit.library.HGate` and
        :class:`~qiskit.circuit.library.SGate` gate and *either* the
        :class:`~qiskit.circuit.library.CXGate` or
        :class:`~qiskit.circuit.library.CZGate` two-qubit gate.

    Clifford operators can also be converted to
    :class:`~qiskit.quantum_info.Operator` objects using the
    :meth:`to_operator` method. This is done via decomposing to a circuit, and then
    simulating the circuit as a unitary operator.

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    def __array__(self, dtype=None):
        if dtype:
            return np.asarray(self.to_matrix(), dtype=dtype)
        return self.to_matrix()

    def __init__(self, data, validate=True):
        """Initialize an operator object."""

        # Initialize from another Clifford by sharing the underlying
        # PauliList
        if isinstance(data, Clifford):
            self._paulis = data._paulis

        # Initialize from ScalarOp as N-qubit identity discarding any global phase
        elif isinstance(data, ScalarOp):
            if not data.num_qubits or not data.is_unitary():
                raise QiskitError("Can only initialize from N-qubit identity ScalarOp.")
            array = np.eye(2 * data.num_qubits, dtype=bool)
            x = array[:, 0 : data.num_qubits]
            z = array[:, data.num_qubits : 2 * data.num_qubits]
            self._paulis = PauliList.from_symplectic(z, x)

        # Initialize from a QuantumCircuit or Instruction object
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._paulis = Clifford.from_circuit(data)._paulis

        # Initialize PauliList directly from the data
        else:
            if isinstance(data, (list, np.ndarray)) and np.asarray(data, dtype=bool).ndim == 2:
                data_array = np.asarray(data, dtype=bool)
                num_qubits = data_array.shape[1] // 2
                self._paulis = PauliList.from_symplectic(
                    z=data_array[:, num_qubits : 2 * num_qubits],
                    x=data_array[:, 0:num_qubits],
                )
            else:
                self._paulis = PauliList(data)

            # Validate table is a symplectic matrix
            if validate and not Clifford._is_symplectic(self.paulis.tableau()):
                raise QiskitError(
                    "Invalid Clifford. Input PauliList is not a valid symplectic matrix."
                )

        # Initialize BaseOperator
        super().__init__(num_qubits=self._paulis.num_qubits)

    def __repr__(self):
        return f"Clifford({repr(self.paulis)})"

    def __str__(self):
        labels = self.paulis.to_labels()
        num_qubits = self.num_qubits
        stabilizer = slice(num_qubits, 2 * num_qubits)
        destabilizer = slice(0, num_qubits)
        return f"Clifford: Stabilizer = {labels[stabilizer]}, Destabilizer = {labels[destabilizer]}"

    def __eq__(self, other):
        """Check if two Clifford tables are equal"""
        return super().__eq__(other) and self._paulis == other._paulis

    # ---------------------------------------------------------------------
    # Attributes
    # ---------------------------------------------------------------------
    @deprecate_function(
        "The Clifford.__getitem__ method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the release date. "
        "Use getter of Clifford.paulis property instead.",  # pylint:disable=bad-docstring-quotes
    )
    def __getitem__(self, key):
        """Return a stabilizer Pauli row"""
        return self.table.__getitem__(key)

    @deprecate_function(
        "The Clifford.__getitem__ method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the release date. "
        "Use Clifford.paulis property instead.",  # pylint:disable=bad-docstring-quotes
    )
    def __setitem__(self, key, value):
        """Set a stabilizer Pauli row"""
        self._paulis.__setitem__(key, value)

    @property
    def paulis(self):
        """Return PauliList"""
        return self._paulis

    @property
    @deprecate_function(
        "The Clifford.table method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the release date. "
        "Use Clifford.paulis method instead.",  # pylint:disable=bad-docstring-quotes
    )
    def table(self):
        """Return StabilizerTable"""
        return StabilizerTable(self.paulis.tableau(), phase=self.paulis.phase // 2)

    @table.setter
    @deprecate_function(
        "The Clifford.table method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the release date. "
        "Use Clifford.paulis method instead.",  # pylint:disable=bad-docstring-quotes
    )
    def table(self, value):
        """Set the stabilizer table"""
        # Note this setter cannot change the size of the Clifford
        # It can only replace the contents of the StabilizerTable with
        # another StabilizerTable of the same size.
        if not isinstance(value, StabilizerTable):
            value = StabilizerTable(value)
        self.paulis = value

    @property
    @deprecate_function(
        "The Clifford.stabilizer method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the release date. "
        "Use Clifford.stabilizers method instead.",  # pylint:disable=bad-docstring-quotes
    )
    def stabilizer(self):
        """Return the stabilizer block of the StabilizerTable."""
        return StabilizerTable(self.table[self.num_qubits : 2 * self.num_qubits])

    @stabilizer.setter
    @deprecate_function(
        "The Clifford.stabilizer method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the release date. "
        "Use Clifford.paulis property instead.",  # pylint:disable=bad-docstring-quotes
    )
    def stabilizer(self, value):
        """Set the value of stabilizer block of the StabilizerTable"""
        inds = slice(self.num_qubits, 2 * self.num_qubits)
        self._paulis.__setitem__(inds, value)

    @property
    @deprecate_function(
        "The Clifford.destabilizer method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the release date. "
        "Use Clifford.destabilizers method instead.",  # pylint:disable=bad-docstring-quotes
    )
    def destabilizer(self):
        """Return the destabilizer block of the StabilizerTable."""
        return StabilizerTable(self.table[0 : self.num_qubits])

    @destabilizer.setter
    @deprecate_function(
        "The Clifford.destabilizer method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the release date. "
        "Use Clifford.paulis property instead.",  # pylint:disable=bad-docstring-quotes
    )
    def destabilizer(self, value):
        """Set the value of destabilizer block of the StabilizerTable"""
        inds = slice(0, self.num_qubits)
        self._paulis.__setitem__(inds, value)

    def stabilizers(self):
        """Return the PauliList corresponding to stabilizer

        Returns:
            PauliList: the Pauli list
        """
        return self.paulis[self.num_qubits : 2 * self.num_qubits]

    def destabilizers(self):
        """Return the PauliList corresponding to destabilizer

        Returns:
            PauliList: the Pauli list
        """
        return self.paulis[0 : self.num_qubits]

    # ---------------------------------------------------------------------
    # Utility Operator methods
    # ---------------------------------------------------------------------

    def is_unitary(self):
        """Return True if the Clifford table is valid."""
        # A valid Clifford is always unitary, so this function is really
        # checking that the underlying Stabilizer table array is a valid
        # Clifford array.
        return Clifford._is_symplectic(self.paulis.tableau())

    # ---------------------------------------------------------------------
    # BaseOperator Abstract Methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        return Clifford._conjugate_transpose(self, "C")

    def adjoint(self):
        return Clifford._conjugate_transpose(self, "A")

    def transpose(self):
        return Clifford._conjugate_transpose(self, "T")

    def tensor(self, other):
        if not isinstance(other, Clifford):
            other = Clifford(other)
        return self._tensor(self, other)

    def expand(self, other):
        if not isinstance(other, Clifford):
            other = Clifford(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        # Pad stabilizers and destabilizers
        destab = b.destabilizers().expand(a.num_qubits * "I") + a.destabilizers().tensor(
            b.num_qubits * "I"
        )
        stab = b.stabilizers().expand(a.num_qubits * "I") + a.stabilizers().tensor(
            b.num_qubits * "I"
        )

        # Add the padded table
        return Clifford(destab + stab, validate=False)

    def compose(self, other, qargs=None, front=False):
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        # If other is a QuantumCircuit we can more efficiently compose
        # using the _append_circuit method to update each gate recursively
        # to the current Clifford, rather than converting to a Clifford first
        # and then doing the composition of tables.
        if not front and isinstance(other, (QuantumCircuit, Instruction)):
            ret = self.copy()
            _append_circuit(ret, other, qargs=qargs)
            return ret

        if not isinstance(other, Clifford):
            other = Clifford(other)

        # Validate compose dimensions
        self._op_shape.compose(other._op_shape, qargs, front)

        # Pad other with identities if composing on subsystem
        other = self._pad_with_identity(other, qargs)

        if front:
            clifford1 = self
            clifford2 = other
        else:
            clifford1 = other
            clifford2 = self

        num_qubits = self.num_qubits
        x_indices = slice(0, num_qubits)
        z_indices = slice(num_qubits, 2 * num_qubits)

        array1 = clifford1.paulis.tableau().astype(int)
        phase1 = clifford1.paulis.phase

        array2 = clifford2.paulis.tableau().astype(int)
        phase2 = clifford2.paulis.phase

        # Update Pauli list
        composed_array = array2.dot(array1) % 2
        pauli = PauliList.from_symplectic(
            z=composed_array[:, z_indices],
            x=composed_array[:, x_indices],
            phase=array2.dot(phase1) + phase2,
        )

        # Correcting for phase due to Pauli multiplication
        ifacts = np.zeros(2 * num_qubits, dtype=int)

        for k in range(2 * num_qubits):

            row2 = array2[k]
            x2 = array2[k, x_indices]
            z2 = array2[k, z_indices]

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

        pauli.phase += 2 * p

        return Clifford(pauli, validate=False)

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def to_dict(self):
        """Return dictionary representation of Clifford object."""
        return {
            "stabilizer": self.stabilizers().to_labels(),
            "destabilizer": self.destabilizers().to_labels(),
        }

    @staticmethod
    def from_dict(obj):
        """Load a Clifford from a dictionary"""
        destabilizer = PauliList(obj.get("destabilizer"))
        stabilizer = PauliList(obj.get("stabilizer"))
        return Clifford(destabilizer + stabilizer)

    def to_matrix(self):
        """Convert operator to Numpy matrix."""
        return self.to_operator().data

    def to_operator(self):
        """Convert to an Operator object."""
        return Operator(self.to_instruction())

    def to_circuit(self):
        """Return a QuantumCircuit implementing the Clifford.

        For N <= 3 qubits this is based on optimal CX cost decomposition
        from reference [1]. For N > 3 qubits this is done using the general
        non-optimal compilation routine from reference [2].

        Return:
            QuantumCircuit: a circuit implementation of the Clifford.

        References:
            1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
               structure of the Clifford group*,
               `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_

            2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
               Phys. Rev. A 70, 052328 (2004).
               `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
        """
        # pylint: disable=cyclic-import
        from qiskit.quantum_info.synthesis.clifford_decompose import (
            decompose_clifford,
        )

        return decompose_clifford(self)

    def to_instruction(self):
        """Return a Gate instruction implementing the Clifford."""
        return self.to_circuit().to_gate()

    @staticmethod
    def from_circuit(circuit):
        """Initialize from a QuantumCircuit or Instruction.

        Args:
            circuit (QuantumCircuit or ~qiskit.circuit.Instruction):
                instruction to initialize.

        Returns:
            Clifford: the Clifford object for the instruction.

        Raises:
            QiskitError: if the input instruction is non-Clifford or contains
                         classical register instruction.
        """
        if not isinstance(circuit, (QuantumCircuit, Instruction)):
            raise QiskitError("Input must be a QuantumCircuit or Instruction")

        # Convert circuit to an instruction
        if isinstance(circuit, QuantumCircuit):
            circuit = circuit.to_instruction()

        # Initialize an identity Clifford
        clifford = Clifford(np.eye(2 * circuit.num_qubits), validate=False)
        _append_circuit(clifford, circuit)
        return clifford

    @staticmethod
    def from_label(label):
        """Return a tensor product of single-qubit Clifford gates.

        Args:
            label (string): single-qubit operator string.

        Returns:
            Clifford: The N-qubit Clifford operator.

        Raises:
            QiskitError: if the label contains invalid characters.

        Additional Information:
            The labels correspond to the single-qubit Cliffords are

            * - Label
              - Stabilizer
              - Destabilizer
            * - ``"I"``
              - +Z
              - +X
            * - ``"X"``
              - -Z
              - +X
            * - ``"Y"``
              - -Z
              - -X
            * - ``"Z"``
              - +Z
              - -X
            * - ``"H"``
              - +X
              - +Z
            * - ``"S"``
              - +Z
              - +Y
        """
        # Check label is valid
        label_gates = {
            "I": IGate(),
            "X": XGate(),
            "Y": YGate(),
            "Z": ZGate(),
            "H": HGate(),
            "S": SGate(),
        }
        if re.match(r"^[IXYZHS\-+]+$", label) is None:
            raise QiskitError("Label contains invalid characters.")
        # Initialize an identity matrix and apply each gate
        num_qubits = len(label)
        op = Clifford(np.eye(2 * num_qubits, dtype=bool))
        for qubit, char in enumerate(reversed(label)):
            _append_circuit(op, label_gates[char], qargs=[qubit])
        return op

    # ---------------------------------------------------------------------
    # Internal helper functions
    # ---------------------------------------------------------------------

    @staticmethod
    def _is_symplectic(mat):
        """Return True if input is symplectic matrix."""
        # Condition is
        # table.T * [[0, 1], [1, 0]] * table = [[0, 1], [1, 0]]
        # where we are block matrix multiplying using symplectic product

        dim = len(mat) // 2
        if mat.shape != (2 * dim, 2 * dim):
            return False

        one = np.eye(dim, dtype=int)
        zero = np.zeros((dim, dim), dtype=int)
        seye = np.block([[zero, one], [one, zero]])
        arr = mat.astype(int)
        return np.array_equal(np.mod(arr.T.dot(seye).dot(arr), 2), seye)

    @staticmethod
    def _conjugate_transpose(clifford, method):
        """Return the adjoint, conjugate, or transpose of the Clifford.

        Args:
            clifford (Clifford): a clifford object.
            method (str): what function to apply 'A', 'C', or 'T'.

        Returns:
            Clifford: the modified clifford.
        """
        ret = clifford.copy()
        num_qubits = clifford.num_qubits
        destabilizer = slice(0, num_qubits)
        stabilizer = slice(num_qubits, 2 * num_qubits)
        if method in ["A", "T"]:
            # Apply inverse
            # Update table
            ret.paulis._phase -= ret.paulis._count_y()
            tmp = ret.paulis.x[destabilizer].copy()
            ret.paulis.x[destabilizer] = ret.paulis.z[stabilizer].T
            ret.paulis.z[destabilizer] = ret.paulis.z[destabilizer].T
            ret.paulis.x[stabilizer] = ret.paulis.x[stabilizer].T
            ret.paulis.z[stabilizer] = tmp.T
            ret.paulis._phase -= ret.paulis._count_y()
            # Update phase
            ret.paulis.phase += clifford.dot(ret).paulis.phase
        if method in ["C", "T"]:
            # Apply conjugate
            ret.paulis.phase += 2 * np.sum(ret.paulis.x & ret.paulis.z, axis=1)
        return ret

    def _pad_with_identity(self, clifford, qargs):
        """Pad Clifford with identities on other subsystems."""
        if qargs is None:
            return clifford

        padded = Clifford(np.eye(2 * self.num_qubits, dtype=bool), validate=False)

        inds = list(qargs) + [self.num_qubits + i for i in qargs]

        # Pad Pauli array
        for i, pos in enumerate(qargs):
            padded.paulis.z[inds, pos] = clifford.paulis.z[:, i]
            padded.paulis.x[inds, pos] = clifford.paulis.x[:, i]

        # Pad phase
        padded.paulis._phase[inds] = clifford.paulis._phase

        return padded


# Update docstrings for API docs
generate_apidocs(Clifford)
