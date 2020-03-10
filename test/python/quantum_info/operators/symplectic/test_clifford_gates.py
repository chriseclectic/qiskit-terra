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

"""Tests for clifford append gate functions."""

import unittest
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators.symplectic import Clifford, append_gate

class TestCliffordGates(QiskitTestCase):
    """Tests for clifford append gate functions."""

    def test_append_1_qubit_gate(self):
        "Tests for append of 1-qubit gates"

        target_table = {
            "i" : np.array([[[True, False], [False, True]]], dtype=np.bool),
            "id": np.array([[[True, False], [False, True]]], dtype=np.bool),
            "iden": np.array([[[True, False], [False, True]]], dtype=np.bool),
            "x": np.array([[[True, False], [False, True]]], dtype=np.bool),
            "y": np.array([[[True, False], [False, True]]], dtype=np.bool),
            "z": np.array([[[True, False], [False, True]]], dtype=np.bool),
            "h": np.array([[[False, True], [True, False]]], dtype=np.bool),
            "s": np.array([[[True, True], [False, True]]], dtype=np.bool),
            "sdg": np.array([[[True, True], [False, True]]], dtype=np.bool),
            "sinv": np.array([[[True, True], [False, True]]], dtype=np.bool),
            "v": np.array([[[True, True], [True, False]]], dtype=np.bool),
            "w": np.array([[[False, True], [True, True]]], dtype=np.bool),

        }

        target_phase = {
            "i": np.array([[False, False]], dtype=np.bool),
            "id": np.array([[False, False]], dtype=np.bool),
            "iden": np.array([[False, False]], dtype=np.bool),
            "x": np.array([[False, True]], dtype=np.bool),
            "y": np.array([[True, True]], dtype=np.bool),
            "z": np.array([[True, False]], dtype=np.bool),
            "h": np.array([[False, False]], dtype=np.bool),
            "s": np.array([[False, False]], dtype=np.bool),
            "sdg": np.array([[True, False]], dtype=np.bool),
            "sinv": np.array([[True, False]], dtype=np.bool),
            "v": np.array([[False, True]], dtype=np.bool),
            "w": np.array([[True, True]], dtype=np.bool)
        }

        target_stabilizer = {
            "i": "+Z",
            "id": "+Z",
            "iden": "+Z",
            "x": "-Z",
            "y": "-Z",
            "z": "+Z",
            "h": "+X",
            "s": "+Z",
            "sdg": "+Z",
            "sinv": "+Z",
            "v": "-X",
            "w": "-Y",
        }

        target_destabilizer = {
            "i": "+X",
            "id": "+X",
            "iden": "+X",
            "x": "+X",
            "y": "-X",
            "z": "-X",
            "h": "+Z",
            "s": "+Y",
            "sdg": "-Y",
            "sinv": "-Y",
            "v": "+Y",
            "w": "-Z",
        }

        for gate_name in ("i", "id", "iden", "x", "y", "z", "h",
                          "s", "sdg", "v", "w"):
            with self.subTest(msg='append gate %s'%gate_name):
                cliff = Clifford([[1, 0], [0, 1]])
                cliff = append_gate(cliff, gate_name, [0])
                value_table = cliff.table._array
                value_phase = cliff.table._phase
                value_stabilizer = cliff.stabilizer.to_labels()
                value_destabilizer = cliff.destabilizer.to_labels()
                self.assertTrue(np.all(np.array(value_table ==
                                                target_table[gate_name])))
                self.assertTrue(np.all(np.array(value_phase ==
                                                target_phase[gate_name])))
                self.assertTrue(np.all(np.array(value_stabilizer ==
                                                [target_stabilizer[gate_name]])))
                self.assertTrue(np.all(np.array(value_destabilizer ==
                                                [target_destabilizer[gate_name]])))

    def test_1_qubit_identity_relations(self):
        "Tests identity relations for 1-qubit gates"
        for gate_name in ("x", "y", "z", "h"):
            with self.subTest(msg='identity for gate %s'%gate_name):
                cliff = Clifford([[1, 0], [0, 1]])
                cliff1 = cliff.copy()
                cliff = append_gate(cliff, gate_name, [0])
                cliff = append_gate(cliff, gate_name, [0])
                self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity for gate s'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff1 = append_gate(cliff1, 's', [0])
            cliff2 = cliff1.copy()
            cliff1 = append_gate(cliff1, 'sdg', [0])
            cliff2 = append_gate(cliff2, 'sinv', [0])
            self.assertEqual(cliff, cliff1)
            self.assertEqual(cliff, cliff2)

        with self.subTest(msg='identity for gate v'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 'v', [0])
            cliff = append_gate(cliff, 'w', [0])
            print ('vw=i:', cliff, " vs ", cliff1)
            #self.assertEqual(cliff, cliff1)

    def test_1_qubit_commute_relations(self):
        "Tests commutativity relations for 1-qubit gates"
        with self.subTest(msg='identity x*y=z'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 'x', [0])
            cliff = append_gate(cliff, 'y', [0])
            cliff1 = append_gate(cliff1, 'z', [0])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity x*z=y'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 'x', [0])
            cliff = append_gate(cliff, 'z', [0])
            cliff1 = append_gate(cliff1, 'y', [0])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity y*z=x'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 'y', [0])
            cliff = append_gate(cliff, 'z', [0])
            cliff1 = append_gate(cliff1, 'x', [0])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity h*x*h=z'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 'h', [0])
            cliff = append_gate(cliff, 'x', [0])
            cliff = append_gate(cliff, 'h', [0])
            cliff1 = append_gate(cliff1, 'z', [0])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity h*y*h=z'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 'h', [0])
            cliff = append_gate(cliff, 'y', [0])
            cliff = append_gate(cliff, 'h', [0])
            cliff1 = append_gate(cliff1, 'y', [0])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity s*s=z'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 's', [0])
            cliff = append_gate(cliff, 's', [0])
            cliff1 = append_gate(cliff1, 'z', [0])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity sdg*sdg=z'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 'sdg', [0])
            cliff = append_gate(cliff, 'sdg', [0])
            cliff1 = append_gate(cliff1, 'z', [0])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity s*x*sdg=y'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 's', [0])
            cliff = append_gate(cliff, 'x', [0])
            cliff = append_gate(cliff, 'sdg', [0])
            cliff1 = append_gate(cliff1, 'y', [0])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity sdg*h=v'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 'sdg', [0])
            cliff = append_gate(cliff, 'h', [0])
            cliff1 = append_gate(cliff1, 'v', [0])
            print ('sdg*h=v:', cliff, " vs ", cliff1)
            #self.assertEqual(cliff, cliff1)

        with self.subTest(msg='identity h*s=w'):
            cliff = Clifford([[1, 0], [0, 1]])
            cliff1 = cliff.copy()
            cliff = append_gate(cliff, 'h', [0])
            cliff = append_gate(cliff, 's', [0])
            cliff1 = append_gate(cliff1, 'w', [0])
            print ('h*s=w:', cliff, " vs ", cliff1)
            #self.assertEqual(cliff, cliff1)

if __name__ == '__main__':
    unittest.main()
