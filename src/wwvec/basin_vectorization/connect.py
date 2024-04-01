import networkx as nx
import numpy as np
from water.basic_functions import ppaths, tt, time_elapsed, my_pool
from wwvec.raster_to_vector.color_grid import color_raster
from wwvec.basin_vectorization.basin_class import BasinData


class Connector:
    def __init__(self, basin_data: BasinData, min_probability=.1):
        self.num_rows, self.num_cols = basin_data.colored_grid.shape
        self.nodes = {
            (row, col) for row in range(self.num_rows) for col in range(self.num_cols)
            if basin_data.probability_grid[row, col] > min_probability or basin_data.colored_grid[row, col] > 0
        }
        # print(len(self.nodes), len(self.nodes)**2)
        self.elevation_grid = basin_data.elevation_grid.astype(np.float32)
        self.probability_grid = basin_data.probability_grid
        self.weight_grid = basin_data.probability_grid.copy()
        self.weight_grid[self.weight_grid > 0] = -np.log2(self.weight_grid[self.weight_grid > 0])
        # self.weight_grid[self.weight_grid < .1] = .1
        self.colored_grid = basin_data.colored_grid
        self.color = basin_data.main_color
        # self.waterways_grid = waterways_grid
        self.graph = nx.DiGraph()
        self.add_edges_to_graph()

    def get_color_min_elevation_points(self):
        rows, cols = np.where(self.colored_grid > 0)
        min_elevation_points = {}
        for (row, col) in zip(rows, cols):
            color = self.colored_grid[row, col]
            elevation = self.elevation_grid[row-1:row+2, col-1:col+2].mean()
            color_info = min_elevation_points.setdefault(color, {'min_elevation': elevation, 'node': (row, col)})
            if elevation < color_info['min_elevation']:
                color_info['min_elevation'] = elevation
                color_info['node'] = (row, col)
        return {color_info['node'] for color_info in min_elevation_points.values()}

    def get_paths(self):
        sources = [(row, col) for (row, col) in zip(*np.where(self.colored_grid == self.color))]
        targets = self.get_color_min_elevation_points()
        init_targets = targets.copy()
        colors_seen = {self.color}
        paths_to_return = {}
        # cutoff_lambda = lambda x: 2.0*4**x
        cut_offs = [2, 8, 100]
        # for i in range(2):
        for i, cutoff in enumerate(cut_offs):
            paths = nx.multi_source_dijkstra_path(
                G=self.graph, sources=sources, cutoff=cutoff, weight=self.get_weight
            )
            # paths = nx.multi_source_dijkstra_path(
            #     G=self.graph, sources=sources, cutoff=cutoff_lambda(i), weight=self.get_weight
            # )
            target_paths = [(target, paths.get(target, [])) for target in targets]
            target_paths.sort(key=lambda x: len(x[1]))
            for target, path in target_paths:
                if len(path) > 0:
                    path = np.array(path)
                    new_color = self.colored_grid[target]
                    path_to_save = []
                    for row, col in path[::-1]:
                        current_color = self.colored_grid[row, col]
                        if current_color not in colors_seen:
                            path_to_save.append((row, col))
                            self.weight_grid[row, col] = 0
                            self.colored_grid[row, col] = new_color
                        else:
                            break
                        # self.colored_grid[path[:, 0], path[:, 1]] = new_color
                    paths_to_return[target] = {'path': path_to_save, 'i': i+3}
                    sources += [(row, col) for (row, col) in zip(*np.where(self.colored_grid == new_color))]
                    colors_seen.add(new_color)
                    # self.probability_grid[path[:, 0], path[:, 1]] = 1
                    targets.remove(target)
            # print(i+3, cutoff_lambda(i), len(paths_to_return), len(targets))
            if len(targets) == 0:
                break
        return paths_to_return, init_targets, colors_seen

    def get_weight(self, node1, node2, *args, **kwargs):
        row1, col1 = node1
        row2, col2 = node2
        elevation1 = self.elevation_grid[row1, col1]
        elevation2 = self.elevation_grid[row2, col2]
        weight1 = self.weight_grid[row1, col1]
        elevation_diff = max(0, elevation1 - elevation2)
        if elevation_diff > 20:
            elevation_diff = 1000
        if elevation_diff == 0:
            return weight1
        else:
            return max(weight1 * elevation_diff, elevation_diff)
            # return ((1 - avg_probability) * elevation_diff/1000)
            # return min((1 - avg_probability), elevation_diff/1000)
            # return ((1 - avg_probability) + elevation_diff/1000)/2

    def add_edges_to_graph(self, nodes_list=None):
        if nodes_list is None:
            nodes_list = self.nodes
        indices = [-1, 0, 1]
        edges = [[(row, col), (row+i, col+j), self.get_weight((row, col), (row+i, col+j))]
                 for (row, col) in nodes_list for i in indices for j in indices
                 if (row+i, col+j) in self.nodes]
        self.graph.add_weighted_edges_from(edges)

#
# if __name__ == "__main__":
#     import rasterio as rio
#     from rasterio import features
#     import geopandas as gpd
#     from water.basic_functions import ppaths, tt, time_elapsed
#     import matplotlib.pyplot as plt
#     rwanda_path = ppaths.country_data/'africa/rwanda'
#     geoglows_path = rwanda_path/'geoglows_ww.parquet'
#     output_path = rwanda_path/'output_data_merged/rwanda_merged_data.tif'
#     elevation_path = rwanda_path/'elevation_merged/rwanda_merged_data.tif'
#     # print('opening data')
#     s = tt()
#     geoglows = gpd.read_parquet(geoglows_path)
#     with rio.open(output_path) as rio_f:
#         grid = rio_f.read()[0]
#         shape = rio_f.shape
#         # print(shape)
#         transform = rio_f.transform
#         ww_grid = features.rasterize(shapes=geoglows.geometry, transform=transform, out_shape=shape)
#         rounded_grid = grid.copy()
#         rounded_grid[grid >= .5] = 1
#         rounded_grid[grid < .5] = 0
#         rounded_grid = rounded_grid.astype(np.int8)
#     with rio.open(elevation_path) as el_f:
#         elevation_data = el_f.read()[0]
#         elevation_data = elevation_data.astype(np.int16)
#     # fig, ax = plt.subplots()
#     # fig, ax = plt.subplots()
#     # new_grid = rounded_grid.copy()
#     # new_grid[new_grid > 0] = 2
#     # new_grid[ww_grid == 1] = 1
#     # ax.imshow(new_grid)
#     num_rows, num_cols = grid.shape
#     rounded_grid[ww_grid == 1] = 1
#     grid[ww_grid == 1] = 1
#     # grid = (grid - .15)/.85
#     grid[grid < 0] = 0
#     grid[grid > 1] = 1
#     # temp_dir = rwanda_path/'temp_slices'
#     # grid_size = 1000
#     # inputs_list = [dict(
#     #     grid=grid, rounded_grid=rounded_grid, elevation_data=elevation_data,
#     #     waterway_grid=ww_grid, i=i, j=j, temp_dir=temp_dir, grid_size=grid_size
#     # ) for i in range(num_rows//grid_size + 1) for j in range(num_cols//grid_size + 1)]
#     #
#     # my_pool(num_proc=30, func=make_and_save_temp_slice, input_list=inputs_list, use_kwargs=True)
#     # new_grid = np.zeros(grid.shape)
#     # for i in range(num_rows//grid_size + 1):
#     #     for j in range(num_cols//grid_size + 1):
#     #         file = temp_dir/f'{i}_{j}.npy'
#     #         if file.exists():
#     #             output = np.load(temp_dir/f'{i}_{j}.npy')
#     #             row_del, col_del = output.shape
#     #             if row_del != grid_size or col_del != grid_size:
#     #                 print(file, output.shape)
#     #             new_grid[i*grid_size: i*grid_size + row_del, j*grid_size: j*grid_size + col_del] = output
#     # new_grid[new_grid>3] = 3
#     # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
#     # ax[0].imshow(grid)
#     # ax[1].imshow(new_grid)
#     print(num_rows, num_cols)
#     # input_data = [{'grid'}]
#     st = tt()
#     grid = grid[1600:2100, 4600:5100]
#     rounded_grid = rounded_grid[1600:2100, 4600:5100]
#     elevation_data = elevation_data[1600:2100, 4600:5100]
#     ww_grid = ww_grid[1600:2100, 4600:5100]
#     rounded_grid[ww_grid == 1] = 1
#     grid = (grid - .1)/.9
#     grid[grid < 0] = 0
#     grid[grid > 1] = 1
#     time_elapsed(s, 2)
#     s = tt()
#     print('coloring grid')
#     colored_grid, *_ = color_raster(rounded_grid, elevation_data)
#     color = colored_grid[ww_grid == 1][0]
#     all_colors = np.unique(colored_grid[ww_grid == 1])
#     colored_grid[np.isin(colored_grid, all_colors)] = color
#     print(len(all_colors))
#     time_elapsed(s, 2)
#     s=tt()
#     print('making connector')
#     conn = Connector(colored_grid, elevation_data, grid, color)
#     time_elapsed(s, 2)
#     s=tt()
#     print('finding paths')
#     pths, tar, comp = conn.get_paths()
#     time_elapsed(st, 2)
#     # time_elapsed(s, 2)
#     # s=tt()
#     new_grid = rounded_grid.copy()
#     new_grid[new_grid > 0] = 1
#     new_grid[ww_grid == 1] = 2
#     for node in tar:
#         path, i = pths.get(node, {'path': [], 'i': 0}).values()
#         for (row, col) in path:
#             if new_grid[row, col] == 0:
#                 new_grid[row, col] = i
#
#     fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
#     ax[0].imshow(new_grid)
#     ax[1].imshow(conn.component_grid)