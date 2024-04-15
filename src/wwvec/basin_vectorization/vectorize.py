import shapely
import numpy as np
from water.basic_functions import ppaths, tt, time_elapsed
import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

class Vectorizer:

    def __init__(self, thin_grid, waterways_data, bbox, x_res, y_res, plot_data=False):
        self.bounds = bbox
        self.x_res, self.y_res = x_res, y_res
        self.waterways_data = waterways_data
        self.thin_grid = thin_grid.copy()
        self.base_waterway_points = self.make_base_waterway_points()
        self.connecting_dict, self.connecting_rows_cols = self.get_connecting_lines()
        sys.setrecursionlimit(100000000)
        self.line_strings = []
        self.seen = {}
        self.square_shaped = [
            [(1, 0), (0, 1), (1, 1)],
            [(1, 0), (0, -1), (1, -1)],
            [(-1, 0), (0, 1), (-1, 1)],
            [(-1, 0), (0, -1), (-1, -1)]
        ]

        self.thin_grid[thin_grid == 2] = 0
        self.clean_embed = self.embed_in_larger(grid=self.thin_grid, side_increase=1)
        # thin_grid[*rows_cols_remove] = 1
        self.count_grid = self.make_count_8_grid(self.clean_embed)
        self.connecting_rows_cols += 1
        self.init_count_copy = self.count_grid.copy()
        # print(self.connecting_rows_cols)
        if len(self.connecting_rows_cols) > 0:
            # self.count_grid[self.connecting_rows_cols[:, 0], self.connecting_rows_cols[:, 1]] += 1
            self.connecting_rows_cols = set([(r, c) for (r, c) in self.connecting_rows_cols])
        # self.count_grid[*rows_cols_remove] = 0
        # thin_grid[*rows_cols_remove] = 0
        if plot_data:
            pass
            # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            # ax[0].imshow(self.init_count_copy)
            # ax[1].imshow(self.count_grid)
        self.init_count_grid = self.count_grid.copy()
        self.make_all_simple_linestrings(plot_data)
        self.make_shapely_line_strings()

        self.line_strings = shapely.line_merge(shapely.MultiLineString(self.line_strings))
        self.line_strings = shapely.node(self.line_strings)
        if hasattr(self.line_strings, 'geoms'):
            self.line_strings = list(self.line_strings.geoms)
        else:
            self.line_strings = [self.line_strings]
        self.connect_to_base_waterways(waterways_data)


    def make_base_waterway_points(self):
        points = []
        for waterway in self.waterways_data:
            if hasattr(waterway, 'geoms'):
                for geom in waterway.geoms:
                    points.extend(shapely.points(geom.coords))
            else:
                points.extend(shapely.points(waterway.coords))
        self.base_waterway_points = np.array(points)
        return self.base_waterway_points

    def get_connecting_lines(self):
        rows, cols = np.where(self.thin_grid == 1)
        connecting_row_cols = []
        connecting_dict = {}
        rows_cols = list(zip(rows, cols))
        for (row, col) in rows_cols:
            if np.any(self.thin_grid[row-1:row+2, col-1:col+2] == 2):
                connecting_row_cols.append((row, col))
        connecting_row_cols = np.array(connecting_row_cols)
        if len(connecting_row_cols) > 0:
            coords = self.row_col_array_to_midpoint_coordinates(connecting_row_cols)
            connecting_points = shapely.points(coords)
            tree = shapely.STRtree(self.base_waterway_points)
            nearest_points = tree.query_nearest(geometry=connecting_points)
            for i, j in zip(*nearest_points):
                connecting_dict[tuple(coords[i])] = {
                    'line': shapely.LineString([connecting_points[i], self.base_waterway_points[j]]),
                    'point': self.base_waterway_points[j].coords[0]
                }
        return connecting_dict, connecting_row_cols


    def connect_to_base_waterways(self, waterways_data):
        new_linestrings = []
        intersection_points = []
        for line_string in self.line_strings:
            to_add = [line_string]
            coords_list = line_string.coords
            head_coords = coords_list[0]
            tail_coords = coords_list[-1]
            if head_coords in self.connecting_dict and tail_coords in self.connecting_dict:
                to_add = []
            elif head_coords in self.connecting_dict:
                to_add.append(self.connecting_dict[head_coords]['line'])
                intersection_points.append(self.connecting_dict[head_coords]['point'])
            elif tail_coords in self.connecting_dict:
                to_add.append(self.connecting_dict[tail_coords]['line'])
                intersection_points.append(self.connecting_dict[tail_coords]['point'])
            else:
                for coords in coords_list:
                    if coords in self.connecting_dict:
                        to_add.append(self.connecting_dict[coords]['line'])
                        intersection_points.append(self.connecting_dict[coords]['point'])
                        break
            if len(to_add) > 0:
                to_add_geom = shapely.unary_union(to_add)
                if hasattr(to_add_geom, 'geoms'):
                    new_linestrings.extend(to_add_geom.geoms)
                else:
                    new_linestrings.append(to_add_geom)
        # self.line_strings = new_linestrings
        num_new = len(new_linestrings)
        num_old = len(waterways_data)
        self.line_strings = gpd.GeoDataFrame(
            {'from_tdx': [False]*num_new, 'geometry': [geometry for geometry in new_linestrings]},
            crs=4326
        )
        old_data = gpd.GeoDataFrame(
            {'from_tdx': [True]*num_old, 'geometry': [geometry for geometry in waterways_data]},
            crs=4326)
        self.line_strings = pd.concat([old_data, self.line_strings], ignore_index=True)
        self.line_strings['from_tdx'] = self.line_strings['from_tdx'].astype(bool)
        self.intersection_points = intersection_points


    def embed_in_larger(self, grid, side_increase):
        num_rows, num_cols = grid.shape
        num_rows += 2*side_increase
        num_cols += 2*side_increase
        copy = np.zeros((num_rows, num_cols), dtype=grid.dtype)
        copy[side_increase:-side_increase, side_increase:-side_increase] = grid
        return copy

    def make_count_8_grid(self, grid):
        grid = grid.copy()
        grid[grid > 0] = 1
        count_grid = np.zeros(grid.shape, dtype=np.int16)
        rows, cols = np.where(grid == 1)
        for row, col in zip(rows, cols):
            count_grid[row, col] = grid[row - 1:row + 2, col - 1:col + 2].sum() - 1
        count_grid = count_grid
        return count_grid

    def make_count_4_grid(self, grid):
        count_grid = np.zeros(grid.shape, dtype=np.int16)
        rows, cols = np.where(grid == 1)
        for row, col in zip(rows, cols):
            count_grid[row, col] = grid[row - 1, col] + grid[row + 1, col] + grid[row, col - 1] + grid[row, col + 1]
        count_grid = count_grid
        return count_grid

    def make_all_simple_linestrings(self, plot_data=False):
        # cg0 = self.count_grid.copy()
        while np.any(self.count_grid == 1):
            self.make_all_linestrings_starting_at_1()
        # cg1 = self.count_grid.copy()
        self.step_1 = self.count_grid.copy()
        self.make_all_linestrings_starting_at_2()
        while np.any(self.count_grid == 1):
            self.make_all_linestrings_starting_at_1()
        # cg2 = self.count_grid.copy()
        self.add_odd_shaped()
        # cg3 = self.count_grid.copy()
        self.make_all_linestrings_starting_at_1(True)
        self.make_all_linestrings_starting_at_2(True)
        # cg4 = self.count_grid.copy()
        if plot_data:
            pass
            # fig, ax = plt.subplots(1, 5, sharex=True, sharey=True)
            # ax[0].imshow(cg0)
            # ax[1].imshow(cg1)
            # ax[2].imshow(cg2)
            # ax[3].imshow(cg3)
            # ax[4].imshow(cg4)

    def add_odd_shaped(self):
        rows, cols = np.where(self.count_grid >= 2)
        for row, col in zip(rows, cols):
            others_seen = set()
            for square_shape in self.square_shaped:
                orow1, ocol1 = row + square_shape[0][0], col + square_shape[0][1]
                orow2, ocol2 = row + square_shape[1][0], col + square_shape[1][1]
                orow3, ocol3 = row + square_shape[2][0], col + square_shape[2][1]
                row_corner, col_corner = row + square_shape[2][0]/2, col + square_shape[2][1]/2
                if (self.clean_embed[row, col] == self.clean_embed[orow1, ocol1]
                        == self.clean_embed[orow2, ocol2] == self.clean_embed[orow3, ocol3]):
                    self.line_strings.append([(row, col), (row_corner, col_corner)])
                    self.line_strings.append([(orow1, ocol1), (row_corner, col_corner)])
                    self.line_strings.append([(orow2, ocol2), (row_corner, col_corner)])
                    self.line_strings.append([(orow3, ocol3), (row_corner, col_corner)])
                    self.count_grid[row, col] = max(self.count_grid[row, col] - 3, 0)
                    self.count_grid[orow1, ocol1] = max(self.count_grid[orow1, ocol1] - 3, 0)
                    self.count_grid[orow2, ocol2] = max(self.count_grid[orow2, ocol2] - 3, 0)
                    self.count_grid[orow3, ocol3] = max(self.count_grid[orow3, ocol3] - 3, 0)
                elif self.count_grid[row, col] > 0 and self.count_grid[orow1, ocol1] > 0 and self.count_grid[orow2, ocol2] > 0:
                    others = [(orow1, ocol1), (orow2, ocol2)]
                    self.line_strings.append([(orow1, ocol1), (row, col), (orow2, ocol2)])
                    not_in_seen = [others for others in others if others not in others_seen]
                    other_new_val = lambda r, c: self.count_grid[r, c] - 2 if (r, c) not in others_seen\
                        else self.count_grid[r, c] - 1
                    self.count_grid[row, col] = max(self.count_grid[row, col] - len(not_in_seen), 0)
                    self.count_grid[orow1, ocol1] = max(other_new_val(orow1, ocol1), 0)
                    self.count_grid[orow2, ocol2] = max(other_new_val(orow2, ocol2), 0)
                    others_seen.update(others)

    def make_all_linestrings_starting_at_1(self, investigate_all=False):
        rows, cols = np.where(self.count_grid == 1)
        for row, col in zip(rows, cols):
            if self.count_grid[row, col] == 1:
                line_string_list = [(row, col)]
                self.line_strings.append(line_string_list)
                self.investigate_row_col(row, col, line_string_list, investigate_all)

    def make_all_linestrings_starting_at_2(self, ignore_init=False):
        if ignore_init:
            rows, cols = np.where((self.count_grid == 2))
        else:
            rows, cols = np.where((self.count_grid == 2) & (self.init_count_grid == 2))
        for row, col in zip(rows, cols):
            line_string_list = [(row, col)]
            if self.count_grid[row, col] > 0:
                self.investigate_row_col(row, col, line_string_list)
            if self.count_grid[row, col] > 0:
                line_string_list.reverse()
                self.investigate_row_col(row, col, line_string_list)
            if len(line_string_list) > 1:
                self.line_strings.append(line_string_list)

    def investigate_row_col(self, row, col, line_string_list, investigate_all=False):
        self.count_grid[row, col] -= 1
        row2, col2 = None, None
        # 114,81
        for i, j in [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            row1, col1 = row + i, col + j
            if len(line_string_list) > 1:
                row2, col2 = line_string_list[-2]
            if row1 != row2 or col1 != col2:
                if self.count_grid[row1, col1] > 0:
                    self.count_grid[row1, col1] -= 1
                    if self.init_count_grid[row1, col1] == 1:
                        line_string_list.append((row1, col1))
                    elif self.init_count_grid[row1, col1] == 2 or (investigate_all and self.count_grid[row1, col1]>0):
                        line_string_list.append((row1, col1))
                        self.investigate_row_col(row1, col1, line_string_list, investigate_all)
                    else:
                        line_string_list.append((row1, col1))
                    return

    def row_col_array_to_midpoint_coordinates(self, row_col_array):
        x_resolution = self.x_res
        y_resolution = self.y_res
        x_min, _, _, y_max = self.bounds
        x_y_array = np.zeros(row_col_array.shape)
        if y_resolution < 0:
            y_resolution = -y_resolution
        x_y_array[:, 0] = x_min + x_resolution*(row_col_array[:, 1] + .5)
        x_y_array[:, 1] = y_max - y_resolution*(row_col_array[:, 0] + .5)
        return x_y_array

    def make_shapely_line_strings(self):
        row_col_lists = self.line_strings
        self.line_strings = []
        for row_col_list in row_col_lists:
            row_col_array = np.array(row_col_list) - 1
            if len(row_col_array) > 1:
                self.line_strings.append(shapely.LineString(self.row_col_array_to_midpoint_coordinates(row_col_array)))

