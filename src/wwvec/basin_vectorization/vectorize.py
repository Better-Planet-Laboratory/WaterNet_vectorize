import shapely
import numpy as np
import sys
import geopandas as gpd
import pandas as pd
from functools import cached_property
from wwvec.basin_vectorization.basin_class import BasinData
from collections import defaultdict

# The investigate_row_col method is called recursively, and the lines can get pretty long. In the long run, it might
# be better to rewrite that function as a loop.
sys.setrecursionlimit(100000000)


# We generally want to make all waterway segments up to intersections (so a waterway only intersects another waterway at
# its head or tail). We then connect those waterways to the reference waterways. This is useful if one of our waterways
# runs parallel to a reference waterway, we won't connect each cell, only the head or tail (or one central point if
# neither the head or tail boarders the reference waterway).

class Vectorizer:
    """
    Vectorizer class

    This class is responsible for vectorizing the thinned waterways data in a given basin.

    Attributes:
        bounds (tuple): The bounds of the basin in the form (minx, miny, maxx, maxy).
        x_res (float): The x resolution of the basin data.
        y_res (float): The y resolution of the basin data.
        reference_waterway_data (list): A list of shapely LineString objects representing the reference waterway data.
        thin_grid (np.ndarray): The thin grid representation of the waterways data.
        connecting_points_coordinates (set): The set of coordinates for all cells that border the tdx waterways.
        list_of_cell_lists (list): A list of cell lists.
        line_strings (list): A list of shapely LineString objects representing the waterways.
        intersection_points (list): A list of intersection points between the waterways and the tdx waterways.
        connections_seen (defaultdict): A defaultdict that keeps track of the connections seen between waterways.
        clean_embed (np.ndarray): The copy of the thin grid with all cells labeled as 2 assigned a value of 0.
        count_grid (np.ndarray): A grid whose cells are the number of neighboring cells that are waterways.
        init_count_copy (np.ndarray): A copy of the count grid.
        init_count_grid (np.ndarray): A copy of the count grid.
        new_linestrings (list): A list of new line strings representing the waterways.

    Methods:
        reference_waterway_points() -> list: Returns a list of shapely Point objects representing the reference waterway points.
        make_connecting_points_coordinates() -> set: Make the set of coordinates for all cells that border the tdx waterways.
        connecting_lines() -> list: Makes a list of linestrings connecting the model's linestrings to the tdx linestrings.
        make_the_line_string_gdf() -> None: Makes a GeoDataFrame with two columns, one indicating if the waterway came from
            tdx-hydro, the other is the geometry column.
        connect_to_base_waterways() -> None: Determines the points for the model's waterways which will be connected to the
            tdx-waterways.
        embed_in_larger(grid: np.ndarray, side_increase: int) -> np.ndarray: Embeds the grid in a larger grid for convenience.
        make_count_8_grid(grid: np.ndarray) -> np.ndarray: Makes a grid whose cells are the number of neighboring cells
            that are waterways.
        make_all_cell_lists() -> None: Updates list_of_cell_lists which will be used to make line strings.
        add_remaining_cell_lists() -> None: Check and add all cells that haven't been fully investigated yet.
        make_all_cell_lists_starting_at_1(investigate_all: bool=False) -> None: Investigate all cells that border only one
            other cell (these should be sources or targets of waterways).
        make_all_cell_lists_starting_at_2(investigate_all: bool=False) -> None: Investigate all cells that boarder two
            other cells (so somewhere in the middle of a waterway)
        add_to_connections_seen(node1, node2): A dicitonary that keeps track of the connections seen.
        investigate_row_col(row, col, cell_list, investigate_all: bool=False) -> None: Investigates a cell,
            then investigates any adjacent cells under appropriate conditions
        row_col_array_to_midpoint_coordinates(row_col_array)-> None: Coverts the cell (row, col) to its midpoint coordinates.
        make_shapely_line_strings() -> None: Makes a shapely LineString object representing the waterways.
    """
    def __init__(
            self, thin_grid: np.ndarray, reference_waterway_data: list[shapely.LineString], basin_data: BasinData
    ):
        self.bounds = basin_data.basin_probability.rio.bounds()
        self.x_res, self.y_res = np.abs(basin_data.basin_probability.rio.resolution())
        self.reference_waterway_data = reference_waterway_data
        self.thin_grid = thin_grid.copy()
        self.connecting_points_coordinates = self.make_connecting_points_coordinates()
        self.list_of_cell_lists = []
        self.line_strings = []
        self.intersection_points = []
        self.connections_seen = defaultdict(set)
        self.thin_grid[thin_grid == 2] = 0
        self.clean_embed = self.embed_in_larger(grid=self.thin_grid, side_increase=1)
        self.count_grid = self.make_count_8_grid(self.clean_embed)
        self.init_count_copy = self.count_grid.copy()
        self.init_count_grid = self.count_grid.copy()
        self.make_all_cell_lists()
        self.make_shapely_line_strings()
        self.connect_to_base_waterways()
        self.make_the_line_string_gdf()

    @cached_property
    def reference_waterway_points(self) -> list[shapely.Point]:
        """
        Returns a list of shapely.Point objects representing the reference waterway points.
        """
        points = []
        for waterway in self.reference_waterway_data:
            if hasattr(waterway, 'geoms'):
                for geom in waterway.geoms:
                    points.extend(shapely.points(geom.coords))
            else:
                points.extend(shapely.points(waterway.coords))
        reference_waterway_points = np.array(points)
        return reference_waterway_points


    def make_connecting_points_coordinates(self) -> set:
        """
        Make the set of coordinates for all cells that boarder the tdx waterways.
        """
        rows, cols = np.where(self.thin_grid == 1)
        connecting_row_cols = []
        connecting_points = set()
        rows_cols = zip(rows, cols)
        for (row, col) in rows_cols:
            if np.any(self.thin_grid[row-1:row+2, col-1:col+2] == 2):
                connecting_row_cols.append((row, col))
        connecting_row_cols = np.array(connecting_row_cols)
        if len(connecting_row_cols) > 0:
            coords = self.row_col_array_to_midpoint_coordinates(connecting_row_cols)
            connecting_points.update([(x, y) for (x, y) in coords])
        return connecting_points

    @cached_property
    def connecting_lines(self) -> list[shapely.LineString]:
        """
        Makes a list of linestrings connecting the model's linestrings to the tdx linestrings.
        """
        if len(self.intersection_points) > 0:
            connecting_points = shapely.points(self.intersection_points)
            tree = shapely.STRtree(self.reference_waterway_points)
            nearest_points = tree.query_nearest(geometry=connecting_points)
            connecting_lines = shapely.linestrings(
                [
                    (connecting_points[i].coords[0], self.reference_waterway_points[j].coords[0])
                    for i, j in zip(*nearest_points)
                ]
            )
            self.intersection_points = [self.reference_waterway_points[j].coords[0] for i, j in zip(*nearest_points)]
            return list(connecting_lines)
        return []


    def make_the_line_string_gdf(self) -> None:
        """
        Makes a geodataframe with two columns, one indicating if the waterway came from tdx-hydro, the other is the
        geometry column.
        """
        num_new = len(self.new_linestrings)
        num_old = len(self.reference_waterway_data)
        self.line_strings = gpd.GeoDataFrame(
            {'from_tdx': [False]*num_new, 'geometry': [geometry for geometry in self.new_linestrings]},
            crs=4326
        )
        old_data = gpd.GeoDataFrame(
            {'from_tdx': [True]*num_old, 'geometry': [geometry for geometry in self.reference_waterway_data]},
            crs=4326)
        self.line_strings = pd.concat([old_data, self.line_strings], ignore_index=True)
        self.line_strings['from_tdx'] = self.line_strings['from_tdx'].astype(bool)


    def connect_to_base_waterways(self) -> None:
        """
        Determines the points for the models waterways which will be connected to the tdx-waterways.
        """
        new_linestrings = []
        connecting_points_coordinates_list = []
        for line_string in self.line_strings:
            coords_list = line_string.coords
            head_coords = tuple(coords_list[0])
            tail_coords = tuple(coords_list[-1])
            if head_coords in self.connecting_points_coordinates and tail_coords in self.connecting_points_coordinates:
                # If the waterway forms a closed loop with the tdx waterways, we ignore it.
                continue
            else:
                new_linestrings.append(line_string)
                if head_coords in self.connecting_points_coordinates:
                    self.intersection_points.append(head_coords)
                elif tail_coords in self.connecting_points_coordinates:
                    self.intersection_points.append(tail_coords)
                else:
                    # If neither the head or tail boarders a tdx waterway, check if any of other points do.
                    for coords in coords_list:
                        if coords in self.connecting_points_coordinates:
                            self.intersection_points.append(coords)
                            break
        new_linestrings += self.connecting_lines
        self.new_linestrings = new_linestrings



    def embed_in_larger(self, grid: np.ndarray, side_increase: int) -> np.ndarray:
        """
        embeds the grid in a larger grid for convience. We will look at subgrids grid[row-1:row+2, col-1:col+2],
         and in the embeded grid we will always have 1<=row<=old_num_rows, 1<=col<=old_num_col so we never go out of
          bounds in the embedded grid.
        """
        num_rows, num_cols = grid.shape
        num_rows += 2*side_increase
        num_cols += 2*side_increase
        copy = np.zeros((num_rows, num_cols), dtype=grid.dtype)
        copy[side_increase:-side_increase, side_increase:-side_increase] = grid
        return copy

    def make_count_8_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Makes a grid whose cells are the number of neighboring cells that are waterways. 'count_8' for 8 connectivity.
        """
        grid = grid.copy()
        grid[grid > 0] = 1
        count_grid = np.zeros(grid.shape, dtype=np.int16)
        rows, cols = np.where(grid == 1)
        for row, col in zip(rows, cols):
            count_grid[row, col] = grid[row - 1:row + 2, col - 1:col + 2].sum() - 1
        count_grid = count_grid
        return count_grid

    def make_all_cell_lists(self) -> None:
        """Updates list_of_cell_lists which will be used to make line strings"""
        while np.any(self.count_grid == 1):
            self.make_all_cell_lists_starting_at_1()
        self.step_1 = self.count_grid.copy()
        self.make_all_cell_lists_starting_at_2()
        while np.any(self.count_grid == 1):
            self.make_all_cell_lists_starting_at_1()
        self.add_remaining_cell_lists()

    def add_remaining_cell_lists(self) -> None:
        """Check and add all cells that haven't been fully investigated yet."""
        while np.any(self.count_grid>0):
            rows, cols = np.where(self.count_grid >= 1)
            for row, col in zip(rows, cols):
                cell_list = [(row, col)]
                self.list_of_cell_lists.append(cell_list)
                self.investigate_row_col(row, col, cell_list, True)

    def make_all_cell_lists_starting_at_1(self, investigate_all: bool=False) -> None:
        """Investigate all cells that boarder only one other cell (these should be sources or targets of waterways)"""
        rows, cols = np.where(self.count_grid == 1)
        for row, col in zip(rows, cols):
            if self.count_grid[row, col] == 1:
                cell_list = [(row, col)]
                self.list_of_cell_lists.append(cell_list)
                self.investigate_row_col(row, col, cell_list, investigate_all)

    def make_all_cell_lists_starting_at_2(self, ignore_init: bool=False) -> None:
        """Investigate all cells that boarder two other cells (so somewhere in the middle of a waterway)"""
        if ignore_init:
            rows, cols = np.where((self.count_grid == 2))
        else:
            rows, cols = np.where((self.count_grid == 2) & (self.init_count_grid == 2))
        for row, col in zip(rows, cols):
            cell_list = [(row, col)]
            if self.count_grid[row, col] > 0:
                self.investigate_row_col(row, col, cell_list)
            if self.count_grid[row, col] > 0:
                cell_list.reverse()
                self.investigate_row_col(row, col, cell_list)
            if len(cell_list) > 1:
                self.list_of_cell_lists.append(cell_list)

    def add_to_connections_seen(self, node1: (int, int), node2: (int, int)) -> None:
        """
        adds nodes to a dictionary of sets of connections seen. Used so that we don't turn around while investigating
        a cell.
        """
        self.connections_seen[node1].add(node2)
        self.connections_seen[node2].add(node1)

    def investigate_row_col(
            self, row: int, col: int, cell_list: list[shapely.LineString], investigate_all: bool=False
    ) -> None:
        """Investigates a cell, then investigates any adjacent cells under appropriate conditions"""
        self.count_grid[row, col] -= 1
        row2, col2 = None, None
        for i, j in [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            row1, col1 = row + i, col + j
            if self.count_grid[row1, col1] > 0:
                if (row1, col1) not in self.connections_seen[(row, col)]:
                    self.add_to_connections_seen((row, col), (row1, col1))
                    self.count_grid[row1, col1] -= 1
                    if self.init_count_grid[row1, col1] == 1 and not investigate_all:
                        cell_list.append((row1, col1))
                    elif self.init_count_grid[row1, col1] == 2 and not investigate_all:
                        cell_list.append((row1, col1))
                        self.investigate_row_col(row1, col1, cell_list, investigate_all)
                    else:
                        cell_list.append((row1, col1))
                    break
        else:
            # In this case there is only one cell in the list, so we can't make a line string from it. This can occur
            # if the cell boarders the reference waterways, but no other model waterways cells.
            cell_list.pop()

    def row_col_array_to_midpoint_coordinates(self, row_col_array) -> np.ndarray:
        x_resolution = self.x_res
        y_resolution = np.abs(self.y_res)
        x_min, _, _, y_max = self.bounds
        x_y_array = np.zeros(row_col_array.shape)
        x_y_array[:, 0] = x_min + x_resolution*(row_col_array[:, 1] + .5)
        x_y_array[:, 1] = y_max - y_resolution*(row_col_array[:, 0] + .5)
        return x_y_array

    def make_shapely_line_strings(self) -> None:
        for row_col_list in self.list_of_cell_lists:
            # We have to decrease by 1 because row,col are from the embedded grid.
            row_col_array = np.array(row_col_list) - 1
            if len(row_col_array) > 1:
                midpoint_coordinates = self.row_col_array_to_midpoint_coordinates(row_col_array)
                self.line_strings.append(shapely.LineString(midpoint_coordinates))
        self.line_strings = shapely.line_merge(shapely.MultiLineString(self.line_strings))
        self.line_strings = shapely.node(self.line_strings)
        if hasattr(self.line_strings, 'geoms'):
            self.line_strings = list(self.line_strings.geoms)
        else:
            self.line_strings = [self.line_strings]

