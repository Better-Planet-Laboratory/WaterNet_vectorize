import networkx as nx
import numpy as np
from water.basic_functions import ppaths, tt, time_elapsed, my_pool
from wwvec.basin_vectorization.basin_class import BasinData
from collections import defaultdict
from functools import cached_property


class Connector:
    """
    The `Connector` class is used to connect the components of a basin based on certain conditions.

    Args:
        basin_data (BasinData):
         The basin data object containing various grids and information about the basin.
        min_probability (float, optional):
         The minimum probability required for a cell to be considered as a node in the graph. Defaults to 0.1.

    Attributes:
        max_elevation_diff (int): The maximum allowable elevation difference that two adjacent cells can have to have
                                    an edge in the graph. Defaults to 25.
        num_rows (int): The number of rows in the basin grid.
        num_cols (int): The number of columns in the basin grid.
        nodes (set): A set of nodes forming the graph.
        elevation_grid (ndarray): The elevation grid of the basin.
        weight_grid (ndarray): The weight grid of the basin.
        component_grid (ndarray): The component grid of the basin.
        component (int): The main component of the basin.
        graph (nx.DiGraph): The graph representation of the basin.

    Methods:
        get_component_min_elevation_points(): Returns the minimum elevation points for each component in the basin.
        get_paths(cut_offs): Finds the shortest paths from disconnected components to the main component.
        get_weight(node1, node2): Computes the weight of an edge between two nodes.
        add_edges_to_graph(nodes_list): Adds edges to the graph based on the nodes list.

    """
    def __init__(self, basin_data: BasinData, min_probability=None, max_elevation_diff: int=20,):
        if min_probability is None:
            min_probability = basin_data.min_val
        self.max_elevation_diff = max_elevation_diff
        self.num_rows, self.num_cols = basin_data.component_grid.shape
        self.nodes = {
            (row, col) for row in range(self.num_rows) for col in range(self.num_cols)
            if basin_data.probability_grid[row, col] > min_probability or basin_data.component_grid[row, col] > 0
        }
        self.elevation_grid = basin_data.elevation_grid.astype(np.float32)
        self.weight_grid = basin_data.weight_grid
        self.component_grid = basin_data.component_grid
        self.main_component = basin_data.main_component
        self.graph = nx.DiGraph()
        self.add_edges_to_graph()

    @cached_property
    def component_information(self):
        rows, cols = np.where((self.component_grid > 0))
        component_information = defaultdict(
            lambda: {'nodes': [], 'min_elevation': np.inf, 'min_elevation_node': (-1, -1)}
        )
        for (row, col) in zip(rows, cols):
            component = self.component_grid[row, col]
            component_information[component]['nodes'].append((row, col))
            elevation = self.elevation_grid[row-1: row+2, col-1: col+2].mean()
            if elevation < component_information[component]['min_elevation']:
                component_information[component]['min_elevation'] = elevation
                component_information[component]['min_elevation_node'] = (row, col)
        return component_information

    def get_paths(self, cut_offs: list = (2, 8, 100)):
        """
        Parameters
        ----------
        cut_offs : list, optional
            List of cutoff values used for multi-source Dijkstra algorithm.
            Defaults to [2, 8, 100].

        Returns
        -------
        tuple
            A tuple containing:
              - paths_to_return : dict
                A dictionary where keys are target points and values are their corresponding paths.
                Each path is represented as a list of (row, col) coordinates.
                The value also includes an 'i' key representing the cutoff index plus 3.
              - init_targets : list
                A list of initial target points before any paths are found.
              - components_seen : set
                A set containing all unique component values encountered during the process.

        """
        sources = self.component_information[self.main_component]['nodes']
        targets = [
            component_info['min_elevation_node'] for (component, component_info) in self.component_information.items()
            if component != self.main_component
        ]
        init_targets = targets.copy()
        components_seen = {self.main_component}
        paths_to_return = {}
        for i, cutoff in enumerate(cut_offs):
            # num_added = []
            paths = nx.multi_source_dijkstra_path(
                G=self.graph, sources=sources, cutoff=cutoff
            )
            target_paths = [(target, paths.get(target, [])) for target in targets]
            target_paths.sort(key=lambda x: len(x[1]))
            for target, path in target_paths:
                if len(path) > 0:
                    path = np.array(path)
                    new_component = self.component_grid[target]
                    path_to_save = []
                    for row, col in path[::-1]:
                        current_component = self.component_grid[row, col]
                        if current_component not in components_seen:
                            path_to_save.append((row, col))
                            if current_component != new_component:
                                self.component_information[new_component]['nodes'].append((row, col))
                            self.weight_grid[row, col] = 0
                            self.component_grid[row, col] = new_component
                            self.update_weight((row, col))
                        else:
                            break
                    paths_to_return[target] = {'path': path_to_save, 'i': i+3}
                    sources += self.component_information[new_component]['nodes']
                    components_seen.add(new_component)
                    targets.remove(target)
            #         num_added.append(target)
            # print(len(num_added))
            if len(targets) == 0:
                break
        return paths_to_return, init_targets, components_seen

    def update_weight(self, node):
        for other_node in self.graph.adj[node]:
            new_weight_1 = self.get_weight(node, other_node)
            self.graph.adj[node][other_node]['weight'] = new_weight_1
            new_weight_2 = self.get_weight(other_node, node)
            self.graph.adj[other_node][node]['weight'] = new_weight_2

    def get_weight(
            self, node1, node2,
            *args, **kwargs
    ):
        """
        Parameters
        ----------
        node1: Tuple[int, int]
            The coordinates of the first node.

        node2: Tuple[int, int]
            The coordinates of the second node.

        *args
            Variable length positional arguments.

        **kwargs
            Variable length keyword arguments.

        Returns
        -------
        float
            The weight of the edge between the two nodes.

        """
        row1, col1 = node1
        row2, col2 = node2
        elevation1 = self.elevation_grid[row1, col1]
        elevation2 = self.elevation_grid[row2, col2]
        elevation_diff = max(0, elevation1 - elevation2)
        weight = self.weight_grid[row1, col1]
        # If the elevation difference is too large, then we don't want to use the edge at all,
        # if the elevation is 0, then we will defer to how well the model did there,
        # and in the final case, we scale the elevation_diff by weight if that increases the weight.
        if elevation_diff > self.max_elevation_diff:
            # The Idea is that our DEM isn't terrible, so water likely shouldn't gain too much elevation,
            # we set that at 20 meters.
            elevation_diff = 1e30
        if elevation_diff == 0:
            # The idea is that if we have a bunch of cells all with a zero elevation difference,
            # then we should use whichever cells the model was most certain about
            return max(weight, 0)
        else:
            # Similarly, we scale the elevation difference up where the model is less certain, but
            # we never scale the elevation difference down. We don't scale the elevation difference down
            # to avoid the graph from searching upstream along cells where the scaled model outputs are 1
            # for a nearby connection
            return max(weight * elevation_diff, elevation_diff)

    def add_edges_to_graph(self, nodes_list=None):
        """
        Parameters
        ----------
        nodes_list : list, optional
            List of nodes to be added as edges to the graph. If not provided, it will use the nodes list of the graph.

        """
        if nodes_list is None:
            nodes_list = self.nodes
        indices = [-1, 0, 1]
        edges = [[(row, col), (row+i, col+j), self.get_weight((row, col), (row+i, col+j))]
                 for (row, col) in nodes_list for i in indices for j in indices
                 if (row+i, col+j) in self.nodes and (i != 0 or j != 0)]
        self.graph.add_weighted_edges_from(edges)
        # edges = [[(row, col), (row+i, col+j)]
        #          for (row, col) in nodes_list for i in indices for j in indices
        #          if (row+i, col+j) in self.nodes and (i != 0 or j != 0)]
        # self.graph.add_edges_from(edges)