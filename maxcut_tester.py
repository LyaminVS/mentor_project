import time
import QAOAQuquarts
import networkx as nx
import matplotlib.pyplot as plt
import tqdm.auto


class MAXCUTTester:
    def __init__(self):
        pass

    def cost_plot(self, min_layers=1, max_layers=1, dimension=2):
        graph = nx.Graph()
        n = 5
        graph.add_nodes_from([i for i in range(0, n)])
        graph.add_edges_from([(i, (i + 1) % n) for i in range(0, n)])
        solver = QAOAQuquarts.MAXCUTSolver(qudit_dimension=dimension, layers=max_layers, graph=graph)
        cost = []
        layers = []
        for num in tqdm.auto.tqdm(range(1, max_layers + 1), leave=True):
            solver.create_circuit(num)
            cost.append(solver.train_one_layer())
            layers.append(num)
        fig, ax = plt.subplots(1, 1)

        ax.set_xlabel("layers")
        ax.set_ylabel("cost function")

        ax.plot(layers[min_layers - 1:], cost[min_layers - 1:])

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
