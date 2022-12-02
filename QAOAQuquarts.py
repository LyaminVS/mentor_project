import cirq
import numpy as np
import networkx as nx
import tqdm.auto
from cirq.contrib.svg import SVGCircuit
import sympy
import scipy
import custom_gates
import pandas as pd
import copy


class MAXCUTSolver:
    def __init__(self, qudit_dimension=4, layers=1, graph=None, weights=None, number_of_restarts=1, noise=False):
        self.number_of_restarts = number_of_restarts
        self.noise = noise
        self.layers = layers
        self.G = copy.deepcopy(graph)
        self.weights = weights
        self.circuit = cirq.Circuit()
        self.qudits = None
        self.measurements = None
        self.results = []
        self.alpha = sympy.Symbol("alpha")
        self.beta = sympy.Symbol("beta")
        self.data_for_hist = None
        self.best_params = []
        self.is_odd = False
        self.is_noisy = noise
        self.qudit_dimension = qudit_dimension
        if self.G is None:
            self.node_number = np.random.randint(9) + 3
            self.edges_number = np.random.randint(self.node_number * (self.node_number - 1) / 2 + 1)
            self.G = nx.gnm_random_graph(self.node_number, self.edges_number)
        else:
            self.node_number = self.G.number_of_nodes()
            self.edges_number = self.G.number_of_edges()
        self.pos = nx.spring_layout(self.G)
        if self.node_number % 2 == 1 and self.qudit_dimension == 4:
            self.G.add_node(self.node_number + 1)
            self.node_number += 1
            self.is_odd = True

        if weights is None:
            self.weights = np.random.rand(self.edges_number + 1) * 10
        nx.set_edge_attributes(
            self.G,
            {e: {"weight": self.weights[i]} for i, e in enumerate(self.G.edges())}
        )
        self.sim = cirq.DensityMatrixSimulator()

    def draw_graph(self):
        G_for_draw = copy.copy(self.G)
        if self.is_odd:
            G_for_draw = self.G.subgraph(list(self.G.nodes())[:-1])

        nx.draw_networkx(G_for_draw, self.pos, with_labels=True, alpha=0.5, node_size=500, width=self.weights)
        edge_labels = dict([((n1, n2), np.round(d['weight'], 2))
                            for n1, n2, d in self.G.edges(data=True)])

        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels,
                                     font_color='red')

    def create_circuit_4(self, layers, is_best_params=False, p_one_qubit=0.001, p_two_qubit=0.01):
        self.circuit = cirq.Circuit()
        qudits = cirq.LineQid.range(self.node_number // 2, dimension=4)
        self.circuit.append(custom_gates.QuquartH().on_each(qudits))
        if self.is_noisy:
            self.circuit.append(custom_gates.QuquartDepolarizingChannel(p_one_qubit).on_each(qudits))
        for i in range(layers):
            mixing_ham = []
            exp1 = self.alpha
            exp2 = self.beta
            if i != layers - 1 or is_best_params:
                exp1 = self.best_params[i][0]
                exp2 = self.best_params[i][1]
            for (u, v, w) in self.G.edges(data=True):
                if min(u, v) % 2 == 0 and min(u, v) + 1 == max(u, v):
                    mixing_ham.append(custom_gates.InnerQuquartZZ(exp1 * w["weight"]).on(qudits[min(u, v) // 2]))
                    if self.is_noisy:
                        mixing_ham.append(custom_gates.QuquartDepolarizingChannel(p_one_qubit).on(qudits[min(u, v) // 2]))
                elif u % 2 == 0 and v % 2 == 0:
                    mixing_ham.append(
                        custom_gates.OuterQuquartZZ(exp1 * w["weight"], 0, 2).on(qudits[u // 2], qudits[v // 2]))
                    if self.is_noisy:
                        mixing_ham.append(
                            custom_gates.DoubleQuquartDepolarizingChannel(p_two_qubit).on(qudits[u // 2], qudits[v // 2]))
                elif u % 2 == 1 and v % 2 == 0:
                    mixing_ham.append(
                        custom_gates.OuterQuquartZZ(exp1 * w["weight"], 1, 2).on(qudits[(u - 1) // 2], qudits[v // 2]))
                    if self.is_noisy:
                        mixing_ham.append(
                            custom_gates.DoubleQuquartDepolarizingChannel(p_two_qubit).on(qudits[(u - 1) // 2], qudits[v // 2]))
                elif u % 2 == 0 and v % 2 == 1:
                    mixing_ham.append(
                        custom_gates.OuterQuquartZZ(exp1 * w["weight"], 0, 3).on(qudits[u // 2], qudits[(v - 1) // 2]))
                    if self.is_noisy:
                        mixing_ham.append(
                            custom_gates.DoubleQuquartDepolarizingChannel(p_two_qubit).on(qudits[u // 2], qudits[(v - 1) // 2]))
                elif u % 2 == 1 and v % 2 == 1:
                    mixing_ham.append(custom_gates.OuterQuquartZZ(exp1 * w["weight"], 1, 3).on(qudits[(u - 1) // 2],
                                                                                               qudits[(v - 1) // 2]))
                    if self.is_noisy:
                        mixing_ham.append(
                            custom_gates.DoubleQuquartDepolarizingChannel(p_two_qubit).on(qudits[(u - 1) // 2],
                                                                                qudits[(v - 1) // 2]))

            problem_ham = [cirq.Moment(custom_gates.QuquartX(exp2).on_each(qudits))]
            self.circuit.append(mixing_ham)
            self.circuit.append(problem_ham)
            if self.is_noisy:
                self.circuit.append(custom_gates.QuquartDepolarizingChannel(p_one_qubit).on_each(qudits))

        self.circuit.append((cirq.measure(qudit) for qudit in qudits))

    def create_circuit_2(self, layers, is_best_params=False, p_one_qubit=0.001, p_two_qubit=0.01):
        self.circuit = cirq.Circuit()
        qudits = cirq.LineQid.range(self.node_number, dimension=2)
        self.circuit.append(cirq.H.on_each(qudits))
        if self.is_noisy:
            self.circuit.append(cirq.Moment(cirq.bit_flip(p=p_one_qubit).on_each(qudits)))
        for i in range(layers):
            exp1 = self.alpha
            exp2 = self.beta
            if i != layers - 1 or is_best_params:
                exp1 = self.best_params[i][0]
                exp2 = self.best_params[i][1]
            self.circuit.append([
                (cirq.ZZPowGate(exponent=exp1 * w["weight"]).on(qudits[u], qudits[v]),
                 cirq.depolarize(p=p_two_qubit, n_qubits=2).on(qudits[u], qudits[v])) if self.is_noisy else cirq.ZZPowGate(exponent=exp1 * w["weight"]).on(qudits[u], qudits[v])
                for (u, v, w) in self.G.edges(data=True)
            ])

            self.circuit.append(cirq.Moment(cirq.XPowGate(exponent=exp2).on_each(qudits)))
            if self.is_noisy:
                self.circuit.append(cirq.Moment(cirq.bit_flip(p=p_one_qubit).on_each(qudits)))

        self.circuit.append((cirq.measure(qudit) for qudit in qudits))

    def create_circuit(self, layers, is_best_params=False, p_one_qubit=0.001, p_two_qubit=0.01):
        if self.qudit_dimension == 2:
            self.create_circuit_2(layers, is_best_params, p_one_qubit, p_two_qubit)
        elif self.qudit_dimension == 4:
            self.create_circuit_4(layers, is_best_params, p_one_qubit, p_two_qubit)

    def draw_circuit(self):
        self.create_circuit(1)
        return SVGCircuit(self.circuit)

    def parse_from_4(self, measurements):
        parsed_measurements = pd.DataFrame()
        for i in range(self.node_number // 2):
            parsed_measurements[f"q({2 * i}) (d=4)"] = measurements[f"q({i}) (d=4)"] // 2
            parsed_measurements[f"q({2 * i + 1}) (d=4)"] = measurements[f"q({i}) (d=4)"] % 2
        if self.is_odd:
            parsed_measurements = parsed_measurements.iloc[:, :-1]
        return parsed_measurements

    def estimate_cost(self, measurements):
        cost_value = 0.0

        if self.qudit_dimension == 4:
            measurements = self.parse_from_4(measurements)

        for u, v, w in self.G.edges(data=True):
            u_samples = measurements[f"q({u}) (d={self.qudit_dimension})"]
            v_samples = measurements[f"q({v}) (d={self.qudit_dimension})"]

            u_signs = (-1) ** u_samples
            v_signs = (-1) ** v_samples

            term_signs = u_signs * v_signs
            term_val = np.mean(term_signs) * w["weight"]
            cost_value += term_val
        return cost_value

    def make_step(self, params):
        sample_results = self.sim.sample(
            self.circuit, params={self.alpha: params[0], self.beta: params[1]}, repetitions=20000
        )
        self.results.append(self.estimate_cost(sample_results))
        if self.qudit_dimension == 4:
            sample_results = self.parse_from_4(sample_results)
        return sample_results

    def solve(self):
        for layer_num in tqdm.auto.tqdm(range(self.layers), leave=False):
            self.create_circuit(layer_num + 1)
            self.train_one_layer()

        sample_results = self.make_step(self.best_params[-1])
        head = sample_results.columns.to_list()
        if "alpha" in head:
            head.remove("alpha")
        if "beta" in head:
            head.remove("beta")
        sample_results['answer'] = sample_results[head].astype(str).values.sum(axis=1)
        self.data_for_hist = sample_results['answer'].value_counts(sort=True)

    def train_one_layer(self):
        alpha = np.linspace(0, 2, num=self.number_of_restarts)
        beta = np.linspace(0, 2, num=self.number_of_restarts)
        fun_results = []
        params = []
        for a in tqdm.auto.tqdm(alpha, leave=False):
            for b in tqdm.auto.tqdm(beta, leave=False):
                results = scipy.optimize.minimize(self.solve_for_parameters, np.array((a, b)), method='COBYLA')
                fun_results.append(results.fun)
                params.append(results.x)
        self.best_params.append(params[fun_results.index(min(fun_results))])
        return min(fun_results)

    def get_data_for_hist(self):
        return self.data_for_hist

    def get_best_params(self):
        return self.best_params

    def get_hist(self, accuracy=0):
        return self.data_for_hist[self.data_for_hist >= self.data_for_hist.max() * accuracy].plot(kind='bar')

    def draw_colored_graph(self):
        color_code = list(list(self.data_for_hist.keys())[0])
        all_colors = ["limegreen", "gold"]
        colors = []
        for i, color in enumerate(color_code):
            colors.append(all_colors[int(color)])
        G_for_draw = copy.copy(self.G)
        if self.is_odd:
            G_for_draw = self.G.subgraph(list(self.G.nodes())[:-1])

        nx.draw_networkx(G_for_draw, self.pos, with_labels=True, alpha=0.5, node_size=500, width=self.weights,
                         node_color=colors)
        edge_labels = dict([((n1, n2), np.round(d['weight'], 2))
                            for n1, n2, d in self.G.edges(data=True)])

        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels,
                                     font_color='red')

    def solve_for_parameters(self, params):
        sample_results = self.sim.sample(
            self.circuit, params={self.alpha: params[0], self.beta: params[1]}, repetitions=20000
        )
        return self.estimate_cost(sample_results)

    def get_fidelities(self):
        self.create_circuit(self.layers, is_best_params=True, p_one_qubit=0, p_two_qubit=0)

        zero_state = np.zeros((2**self.node_number, 2**self.node_number))
        zero_state[0][0] = 1

        densities_with_noise = [zero_state]
        for step_result in self.sim.simulate_moment_steps(self.circuit):
            densities_with_noise.append(step_result.density_matrix())

        self.create_circuit(self.layers, is_best_params=True, p_one_qubit=0.01, p_two_qubit=0.1)

        densities_without_noise = [zero_state]
        for step_result in self.sim.simulate_moment_steps(self.circuit):
            densities_without_noise.append(step_result.density_matrix())

        fidelities = []
        for i in range(len(densities_with_noise)):
            fidelities.append(cirq.fidelity(cirq.density_matrix(densities_with_noise[i], atol=0.001),
                                            cirq.density_matrix(densities_without_noise[i], atol=0.001), atol=0.001))
        return fidelities

    def classical_solve(self):
        count = self.node_number
        if self.is_odd:
            count -= 1
        all_states = ["0" * (count - len(bin(i)[2:])) + bin(i)[2:] for i in range(2 ** count)]
        result = []
        for i, state in enumerate(all_states):
            result.append(0)
            for (u, v, w) in self.G.edges(data=True):
                if state[u] != state[v]:
                    result[i] += w["weight"]

        all_best_params = []

        for j in range(result.count(max(result))):
            all_best_params.append(all_states[result.index(max(result))])
            result[result.index(max(result))] = -1

        return all_best_params
