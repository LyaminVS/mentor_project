import cirq
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from cirq.contrib.svg import SVGCircuit
import sympy
import scipy
import custom_gates


class MAXCUTSolver:
    def __init__(self, qudit_dimension=4, layers=1, graph=None, weights=None):
        self.layers = layers
        self.G = graph
        self.weights = weights
        self.circuit = None
        self.qudits = None
        self.measurements = None
        self.results = None
        self.results = []
        self.alpha = sympy.Symbol("alpha")
        self.beta = sympy.Symbol("beta")
        self.data_for_hist = None
        self.best_params = None
        if graph is None:
            self.node_number = np.random.randint(9) + 3
            self.edges_number = np.random.randint(self.node_number * (self.node_number - 1) / 2 + 1)
            self.G = nx.gnm_random_graph(self.node_number, self.edges_number)
        else:
            self.node_number = graph.number_of_nodes()
            self.edges_number = graph.number_of_edges()

        if weights is None:
            self.weights = np.random.rand(self.edges_number + 1) * 10
        self.qudit_dimension = qudit_dimension
        nx.set_edge_attributes(
            self.G,
            {e: {"weight": self.weights[i]} for i, e in enumerate(self.G.edges())}
        )
        self.create_circuit()
        self.sim = cirq.Simulator()

    def draw_graph(self):
        nx.draw(self.G, with_labels=True, alpha=0.5, node_size=500, width=self.weights)

    def create_circuit(self):
        qudits = cirq.LineQid.range(self.node_number, dimension=self.qudit_dimension)
        self.circuit = cirq.Circuit()
        self.circuit.append(custom_gates.QuquartH().on_each(qudits))
        mixing_ham = [
                custom_gates.QuquartZZ(self.alpha * w["weight"], 0, 1).on(qudits[u], qudits[v])
                for (u, v, w) in self.G.edges(data=True)
            ]
        problem_ham = cirq.Moment(custom_gates.QuquartX(self.beta, 0, 1).on_each(qudits))
        for _ in range(self.layers):
            self.circuit.append(mixing_ham)
            self.circuit.append(problem_ham)
        self.circuit.append((cirq.measure(qudit) for qudit in qudits))

    def draw_circuit(self):
        return SVGCircuit(self.circuit)

    def estimate_cost(self, measurements):
        cost_value = 0.0

        for u, v, w in self.G.edges(data=True):
            u_samples = measurements[f"q({u}) (d=4)"]
            v_samples = measurements[f"q({v}) (d=4)"]

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
        return sample_results

    def solve(self):
        best_params = scipy.optimize.minimize(self.solve_for_parameters, np.array((1, 1)), method='COBYLA').x
        sample_results = self.make_step(best_params)
        head = sample_results.columns.to_list()
        head.remove("alpha")
        head.remove("beta")
        sample_results['answer'] = sample_results[head].astype(str).values.sum(axis=1)
        self.data_for_hist = sample_results['answer'].value_counts(sort=True)
        self.best_params = best_params

    def get_data_for_hist(self):
        return self.data_for_hist

    def get_best_params(self):
        return self.best_params

    def get_hist(self, accuracy=0):
        return self.data_for_hist[self.data_for_hist >= self.data_for_hist.max() * accuracy].plot(kind='bar')

    def get_colored_graph(self):
        color_code = list(list(self.data_for_hist.keys())[0])
        all_colors = ["limegreen", "gold"]
        colors = []
        for i, color in enumerate(color_code):
            colors.append(all_colors[int(color)])
        nx.draw(self.G, with_labels=True, alpha=0.5, node_size=500, width=self.weights, node_color=colors)

    def solve_for_parameters(self, params):
        sample_results = self.sim.sample(
            self.circuit, params={self.alpha: params[0], self.beta: params[1]}, repetitions=20000
        )
        return self.estimate_cost(sample_results)

    def classical_solve(self):
        all_states = ["0" * (self.node_number - len(bin(i)[2:])) + bin(i)[2:] for i in range(2 ** self.node_number)]
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
