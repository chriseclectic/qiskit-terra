from numpy import array
from qiskit import QiskitError
try:
    from qiskit import Aer
    from qiskit.providers.aer import noise
except ImportError:
    raise QiskitError("Unable to generate mitigation data without Aer simulator")

# For simulation
import qiskit.utils.mitigation as mit
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.result import Result
from qiskit.test import QiskitTestCase

SEED = 100
SHOTS = 10000


def execute_circs(qc_list, sim, noise_model=None):
    """Run circuits with the readout noise defined in this class"""
    return sim.run(
        qc_list,
        shots=SHOTS,
        seed_simulator=SEED,
        noise_model=noise_model,
        method="density_matrix",
    ).result()

def generate_mitigation_matrices(num_qubits, sim, noise_model, method='tensored'):
    qr = QuantumRegister(num_qubits)
    qubit_list = range(num_qubits)
    meas_calibs, state_labels = mit.complete_meas_cal(
        qubit_list=qubit_list, qr=qr, circlabel="mcal"
    )
    cal_res = sim.run(
        meas_calibs,
        shots=SHOTS,
        seed_simulator=SEED,
        basis_gates=noise_model.basis_gates,
        noise_model=noise_model,
    ).result()

    if method == 'complete':
        meas_fitter = mit.CompleteMeasFitter(
            cal_res, state_labels, qubit_list=qubit_list, circlabel="mcal"
        )
        return meas_fitter.cal_matrix

    elif method == 'tensored':
        mit_pattern = [[qubit] for qubit in qubit_list]
        meas_fitter = mit.TensoredMeasFitter(
            cal_res, mit_pattern=mit_pattern, circlabel="mcal"
        )
        return meas_fitter.cal_matrices

    return None

def hex_to_bin(hex_s, length):
    j = int(hex_s, 16)
    return bin(j)[2:].zfill(length)

def get_counts(result):
    num_qubits = result.header.n_qubits
    counts_dict = result.data.counts
    return {hex_to_bin(key, num_qubits): val for key, val in counts_dict.items()}

def generate_data(num_qubits, circuits, noise_model = None):
    sim = Aer.get_backend("aer_simulator")
    tensor_method_matrices = generate_mitigation_matrices(num_qubits, sim, noise_model, method='tensored')
    complete_method_matrix = generate_mitigation_matrices(num_qubits, sim,
                                                          noise_model,
                                                          method='complete')
    results_noise = execute_circs(circuits, sim, noise_model)
    results_ideal = execute_circs(circuits, sim)
    results_noise_dict = {result.header.name: get_counts(result) for result in results_noise.results}
    results_ideal_dict = {result.header.name: get_counts(result) for result
                          in results_ideal.results}
    result = {}
    result['tensor_method_matrices'] = tensor_method_matrices
    result['complete_method_matrix'] = complete_method_matrix
    result['circuits'] = {}
    for name in results_noise_dict.keys():
        result['circuits'][name] = {'counts_ideal': results_ideal_dict[name], 'counts_noise': results_noise_dict[name]}
    return result

def readout_errors_1(num_qubits):
    # Create readout errors
    readout_errors = []
    for i in range(num_qubits):
        p_error1 = (i + 1) * 0.002
        p_error0 = 2 * p_error1
        ro_error = noise.ReadoutError(
            [[1 - p_error0, p_error0], [p_error1, 1 - p_error1]])
        readout_errors.append(ro_error)

    # Readout Error only
    noise_model = noise.NoiseModel()
    for i in range(num_qubits):
        noise_model.add_readout_error(readout_errors[i], [i])
    return noise_model

def ghz_circuit(num_qubits):
    qc = QuantumCircuit(num_qubits, name = "ghz_{}_qubits".format(num_qubits))
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(i - 1, i)
    qc.measure_all()
    return qc

def first_qubit_h(num_qubits):
    qc = QuantumCircuit(num_qubits, name="first_qubit_h_{}_qubits".format(num_qubits))
    qc.h(0)
    qc.measure_all()
    return qc

def test_1():
    num_qubits = 3
    noise_model = readout_errors_1(num_qubits)
    circuits = [ghz_circuit(num_qubits), first_qubit_h(num_qubits)]
    data = generate_data(num_qubits, circuits, noise_model = noise_model)
    return data

def generate_all_test_data():
    test_data = {}
    test_data['test_1'] = test_1()
    print(test_data)


if __name__ == "__main__":
    generate_all_test_data()

test_data = {'test_1': {'tensor_method_matrices': [array([[0.996525, 0.002   ],
       [0.003475, 0.998   ]]), array([[0.991175, 0.00415 ],
       [0.008825, 0.99585 ]]), array([[0.9886 , 0.00565],
       [0.0114 , 0.99435]])], 'complete_method_matrix': array([[9.771e-01, 1.800e-03, 4.600e-03, 0.000e+00, 5.600e-03, 0.000e+00,
        0.000e+00, 0.000e+00],
       [3.200e-03, 9.799e-01, 0.000e+00, 3.400e-03, 0.000e+00, 5.800e-03,
        0.000e+00, 1.000e-04],
       [8.000e-03, 0.000e+00, 9.791e-01, 2.400e-03, 1.000e-04, 0.000e+00,
        5.700e-03, 0.000e+00],
       [0.000e+00, 8.300e-03, 3.200e-03, 9.834e-01, 0.000e+00, 0.000e+00,
        0.000e+00, 5.300e-03],
       [1.170e-02, 0.000e+00, 0.000e+00, 0.000e+00, 9.810e-01, 2.500e-03,
        5.000e-03, 0.000e+00],
       [0.000e+00, 9.900e-03, 0.000e+00, 0.000e+00, 3.900e-03, 9.823e-01,
        0.000e+00, 3.500e-03],
       [0.000e+00, 0.000e+00, 1.310e-02, 0.000e+00, 9.400e-03, 1.000e-04,
        9.857e-01, 1.200e-03],
       [0.000e+00, 1.000e-04, 0.000e+00, 1.080e-02, 0.000e+00, 9.300e-03,
        3.600e-03, 9.899e-01]]), 'circuits': {'ghz_3_qubits': {'counts_ideal': {'111': 5015, '000': 4985}, 'counts_noise': {'111': 4955, '000': 4886, '001': 16, '100': 46, '010': 36, '101': 23, '011': 29, '110': 9}}, 'first_qubit_h_3_qubits': {'counts_ideal': {'000': 4932, '001': 5068}, 'counts_noise': {'000': 4844, '001': 4962, '100': 56, '101': 65, '011': 37, '010': 35, '110': 1}}}}}

