import time

import numpy as np

import QAOAQuquarts
import networkx as nx
import matplotlib.pyplot as plt
import tqdm.auto


class MAXCUTTester:
    def __init__(self):
        pass

    def cost_plot(self, min_layers=1, max_layers=1, dimension=2, noise=False):
        graph = nx.Graph()
        n = 5
        graph.add_nodes_from([i for i in range(0, n)])
        graph.add_edges_from([(i, (i + 1) % n) for i in range(0, n)])
        solver = QAOAQuquarts.MAXCUTSolver(qudit_dimension=dimension, layers=max_layers, graph=graph, noise=noise,
                                           p_one_qubit=0.01, p_two_qubit=0.1)
        cost = []
        layers = []
        for num in tqdm.auto.tqdm(range(1, max_layers + 1), leave=True):
            solver.create_circuit(num)
            cost.append(solver.train_one_layer())
            layers.append(num)
        fig, ax = plt.subplots(1, 1)

        solver_1 = QAOAQuquarts.MAXCUTSolver(qudit_dimension=dimension, layers=max_layers, graph=graph, noise=noise,
                                           p_one_qubit=0.01, p_two_qubit=0.1)
        cost_1 = []
        layers_1 = []
        for num_1 in tqdm.auto.tqdm(range(1, max_layers + 1), leave=True):
            solver_1.create_circuit(num_1)
            cost_1.append(solver_1.train_one_layer())
            layers_1.append(num_1)

        ax.set_xlabel("layers", fontsize=15)
        ax.set_ylabel("cost function", fontsize=15)



        ax.plot(layers[min_layers - 1:], cost[min_layers - 1:], label="Кубиты")

        ax.plot(layers_1[min_layers - 1:], cost_1[min_layers - 1:], label="Кудиты")

        plt.legend()

    def classical_solve_time(self, graph):
        solver = QAOAQuquarts.MAXCUTSolver(qudit_dimension=2, layers=1, graph=graph)
        beg = time.time()
        solver.classical_solve()
        end = time.time()
        return end - beg

    def simulator_solve_time(self, graph, dimension=2, layers=1):
        solver = QAOAQuquarts.MAXCUTSolver(qudit_dimension=dimension, layers=layers, graph=graph)
        beg = time.time()
        solver.solve()
        end = time.time()
        return end - beg


    def classical_solve_time_plot(self, min_node_number=1, max_node_number=1):
        times = []
        node_nums = []
        for node_num in tqdm.auto.tqdm(range(min_node_number, max_node_number + 1)):
            graph = nx.Graph()
            graph.add_nodes_from([i for i in range(0, node_num)])
            graph.add_edges_from([(i, (i + 1) % node_num) for i in range(0, node_num)])
            node_nums.append(node_num)
            times.append(self.classical_solve_time(graph))

        fig, ax = plt.subplots(1, 1)

        ax.set_xlabel("node number")
        ax.set_ylabel("time")

        ax.plot(node_nums, times)

    def simulator_solve_time_plot(self, min_node_number=1, max_node_number=1, dimension=2):
        times = []
        node_nums = []
        for node_num in tqdm.auto.tqdm(range(min_node_number, max_node_number + 1)):
            graph = nx.Graph()
            graph.add_nodes_from([i for i in range(0, node_num)])
            graph.add_edges_from([(i, (i + 1) % node_num) for i in range(0, node_num)])
            node_nums.append(node_num)
            times.append(self.simulator_solve_time(graph, dimension))

        fig, ax = plt.subplots(1, 1)

        ax.set_xlabel("node number")
        ax.set_ylabel("time")

        ax.plot(node_nums, times)

    def get_fidelity_plot(self, p_one_qubit=0.01, p_two_qubit=0.1, layers=1, dimension=2, node_number=6):
        graph = nx.Graph()
        graph.add_nodes_from([i for i in range(0, node_number)])
        graph.add_edges_from([(i, (i + 1) % node_number) for i in range(0, node_number)])
        solver = QAOAQuquarts.MAXCUTSolver(qudit_dimension=dimension, layers=layers, graph=graph, noise=True, p_one_qubit=p_one_qubit, p_two_qubit=p_two_qubit)
        solver.solve()
        fid = solver.get_fidelities()
        fig, ax = plt.subplots(1, 1)

        ax.set_xlabel("gate number")
        ax.set_ylabel("fidelity")

        ax.plot(np.arange(len(fid[:-1:2])), fid[:-1:2])

