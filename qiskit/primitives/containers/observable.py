# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Sparse container class for an Estimator observable.
"""
from __future__ import annotations

from typing import Union, Mapping as MappingType
from collections.abc import Mapping
from collections import defaultdict
from numbers import Real, Complex

from qiskit.quantum_info import Pauli, SparsePauliOp


ObservableKey = "tuple[tuple[int, ...], str]"  # pylint: disable = invalid-name
ObservableKeyLike = Union[Pauli, str, ObservableKey]
ObservableLike = Union[
    SparsePauliOp,
    ObservableKeyLike,
    MappingType[ObservableKeyLike, Real],
]
"""Types that can be natively used to construct an observable."""


class Observable(Mapping[ObservableKey, float]):
    """A sparse container for a observable for an estimator primitive."""

    __slots__ = ("_data", "_num_qubits")

    def __init__(
        self,
        data: Mapping[ObservableKey, float],
        num_qubits: int = None,
        validate: bool = True,
    ):
        """Initialize an observables array.

        Args:
            data: The observable data.
            num_qubits: The number of qubits in the data.
            validate: If ``True``, the input data is validated during initialization.

        Raises:
            ValueError: If ``validate=True`` and the input observable-like is not valid.
        """
        self._data = data
        self._num_qubits = num_qubits
        if validate:
            self.validate()

    def __repr__(self):
        return f"{type(self).__name__}({self._data})"

    def __getitem__(self, key: ObservableKey) -> float:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def num_qubits(self) -> int:
        """The number of qubits in the observable"""
        if self._num_qubits is None:
            num_qubits = 0
            for key in self._data:
                num_qubits = max(num_qubits, 1 + max(key[0]))
            self._num_qubits = num_qubits
        return self._num_qubits

    def validate(self):
        """Validate the consistency in observables array."""
        if not isinstance(self._data, Mapping):
            raise TypeError(f"Observable data type {type(self._data)} is not a Mapping.")

        # If not already set, we compute num_qubits while iterating over data
        # to avoid having to do another iteration later
        data_num_qubits = 0
        for key, value in self._data.items():
            if not isinstance(key, tuple):
                raise TypeError("Invalid Observable key type")
            if len(key) != 2:
                raise ValueError(f"Invalid Observable key value {key}")
            # Check tuple pos types after checking length
            if not isinstance(key[0], tuple) or not isinstance(key[1], str):
                raise TypeError("Invalid Observable key type")
            data_num_qubits = max(data_num_qubits, 1 + max(key[0]))
            if not isinstance(value, Real):
                raise TypeError(f"Value {value} is not a real number")
        if self._num_qubits is None:
            self._num_qubits = data_num_qubits
        elif self._num_qubits < data_num_qubits:
            raise ValueError("Num qubits is less than the maximum qubit in observable keys")

    @classmethod
    def coerce(cls, observable: ObservableLike) -> Observable:
        """Coerce an observable-like object into an :class:`.Observable`.

        Args:
            observable: The observable-like input.

        Returns:
            A coerced observables array.

        Raises:
            TypeError: If the input cannot be formatted because its type is not valid.
            ValueError: If the input observable is invalid.
        """
        return cls._coerce(observable, num_qubits=None)

    @classmethod
    def _coerce(cls, observable, num_qubits=None):
        # Pauli-type conversions
        if isinstance(observable, SparsePauliOp):
            # TODO: Make sparse by removing identity qubits in keys
            data = dict(observable.simplify(atol=0).to_list())
            return cls._coerce(data, num_qubits=observable.num_qubits)

        if isinstance(observable, (Pauli, str, tuple)):
            return cls._coerce({observable: 1}, num_qubits=num_qubits)

        # Mapping conversion (with possible Pauli keys)
        if isinstance(observable, Mapping):
            key_qubits = set()
            unique = defaultdict(float)
            for key, coeff in observable.items():
                if isinstance(key, Pauli):
                    # TODO: Make sparse by removing identity qubits in keys
                    label, phase = key[:].to_label(), key.phase
                    if phase != 0:
                        coeff = coeff * (-1j) ** phase
                    qubits = tuple(range(key.num_qubits))
                    key = (qubits, label)
                elif isinstance(key, str):
                    qubits = tuple(range(len(key)))
                    key = (qubits, key)
                if not isinstance(key, tuple):
                    raise TypeError(f"Invalid key type {type(key)}")
                if len(key) != 2:
                    raise ValueError(f"Invalid key {key}")
                if not isinstance(key[0], tuple) or not isinstance(key[1], str):
                    raise TypeError("Invalid key type")
                # Truncate complex numbers to real
                if isinstance(coeff, Complex):
                    if abs(coeff.imag) > 1e-7:
                        raise TypeError(f"Invalid coeff for key {key}, coeff must be real.")
                    coeff = coeff.real
                unique[key] += coeff
                key_qubits.update(key[0])
            if num_qubits is None:
                num_qubits = 1 + max(key_qubits)
            obs = cls(dict(unique), num_qubits=num_qubits)
            return obs

        raise TypeError(f"Invalid observable type: {type(observable)}")