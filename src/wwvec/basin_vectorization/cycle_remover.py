import numpy as np
import shapely
import networkx as nx
import geopandas as gpd
import xarray as xr


class CycleRemover:
    """

    A class for removing cycles from waterway vector data.

    Attributes:
    - waterways (list): List of waterway geometries.
    - intersection_points (list): List of intersection points.
    - elevation_data (DataArray): DataArray containing elevation data.
    - graph (DiGraph): Graph representing the waterway paths.

    Methods:
    - make_graph(): Creates a directed graph representing the waterway paths.
    - add_edges_to_graph(): Adds edges to the graph based on the waterway paths and elevation data.
    - get_coordinate_elevations(coordinates): Retrieves elevation values for given coordinates.
    - find_paths(): Finds paths without cycles in the graph.

    """
    def __init__(
            self, waterways: gpd.GeoDataFrame, intersection_points: list[(float, float)], elevation_data: xr.DataArray
    ):
        self.waterways = waterways[~waterways.from_tdx].geometry.to_list()
        self.intersection_points = intersection_points
        self.elevation_data = elevation_data
        self.graph = self.make_graph()

    def make_graph(self):
        self.graph = nx.DiGraph()
        self.add_edges_to_graph()
        return self.graph

    def add_edges_to_graph(self):
        edges = []
        weight_lambda = lambda x: 0 if x < 0 else x

        for waterway in self.waterways:
            coordinates = np.array([list(coords) for coords in waterway.coords])
            elevations = self.get_coordinate_elevations(coordinates)
            for i, coords in enumerate(coordinates[:-1]):
                next_coords = coordinates[i+1]
                elevation = elevations[i]
                next_elevation = elevations[i+1]
                diff = elevation - next_elevation
                edges.append((tuple(coords), tuple(next_coords), weight_lambda(diff)))
                edges.append((tuple(next_coords), tuple(coords), weight_lambda(-diff)))
        self.graph.add_weighted_edges_from(edges)

    def get_coordinate_elevations(self, coordinates):
        x = xr.DataArray(coordinates[:, 0], dims='points')
        y = xr.DataArray(coordinates[:, 1], dims='points')
        elevations = self.elevation_data[0].sel(x=x, y=y, method='nearest').values.astype(float)
        return elevations

    def find_paths(self):
        new_paths = []
        paths = nx.multi_source_dijkstra_path(self.graph, sources=self.intersection_points)
        seen_nodes = set()
        for path in paths.values():
            new_path = []
            for coords in path[::-1]:
                if coords not in seen_nodes:
                    new_path.append(coords)
                    seen_nodes.add(coords)
                else:
                    new_path.append(coords)
                    break
            if len(new_path) > 1:
                new_paths.append(shapely.LineString(new_path))
        return new_paths
