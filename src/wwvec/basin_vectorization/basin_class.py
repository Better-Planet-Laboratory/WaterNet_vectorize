from wwvec.paths import BasinPaths
import shapely
from wwvec.basin_vectorization.cut_bbox_raster import make_bbox_raster
import numpy as np
from rasterio import features
from wwvec.raster_to_vector.thin_grid import thinner
from wwvec.raster_to_vector.color_grid import color_raster
# import matplotlib.pyplot as plt

class BasinData:
    def __init__(
            self, basin_geometry: shapely.Polygon,
            stream_geometry: shapely.LineString,
            paths: BasinPaths, bbox_buffer: float=.005,
            **kwargs
    ):
        self.bbox_buffer = bbox_buffer
        self.paths = paths
        self.basin_geometry = basin_geometry
        self.stream_geometry = stream_geometry
        bbox = tuple(self.basin_geometry.bounds)
        self.grid_bbox = (bbox[0] - bbox_buffer, bbox[1] - bbox_buffer, bbox[2] + bbox_buffer, bbox[3] + bbox_buffer)
        self.basin_probability, self.basin_elevation, self.basin_grid, self.waterway_grid\
            = self.cut_basin_data(**kwargs)
        self.elevation_grid = self.basin_elevation[0].to_numpy().astype(np.int16)
        self.probability_grid = self.make_probability_grid()
        self.rounded_grid = self.make_rounded_grid(**kwargs)
        self.colored_grid, *_ = color_raster(self.rounded_grid, self.elevation_grid)
        self.main_color = self.make_main_color()
        self.remove_waterways_out_of_basin()
        self.colored_grid, *_ = color_raster(self.rounded_grid, self.elevation_grid)
        self.main_color = self.make_main_color()
        self.scale_probability_grid(**kwargs)

    def cut_basin_data(self, stream_buffer=.0001, **kwargs):
        bbox = self.grid_bbox
        basin_probability = make_bbox_raster(bbox, base_dir=self.paths.waterways_grid)
        basin_probability = basin_probability.astype(np.float32)/255
        basin_elevation = make_bbox_raster(bbox, base_dir=self.paths.elevation_path)
        basin_elevation = basin_elevation.rio.reproject_match(basin_probability)
        shape = basin_probability[0].shape
        transform = basin_probability.rio.transform()
        basin_geometry_grid = features.rasterize(
            shapes=[self.basin_geometry], out_shape=shape, transform=transform
        )
        waterway_grid = features.rasterize(
            shapes=[self.stream_geometry.buffer(stream_buffer)], out_shape=shape, transform=transform, all_touched=True
        )
        waterway_grid[basin_geometry_grid == 0] = 0
        return basin_probability, basin_elevation, basin_geometry_grid, waterway_grid

    def make_rounded_grid(self, round_value=.5, **kwargs):
        self.rounded_grid = self.probability_grid.copy()
        self.rounded_grid[self.rounded_grid >= round_value] = 1
        self.rounded_grid[self.rounded_grid < round_value] = 0
        self.rounded_grid = self.rounded_grid.astype(np.int8)
        return self.rounded_grid

    def make_probability_grid(self):
        self.probability_grid = self.basin_probability.to_numpy()[0]
        self.probability_grid[self.waterway_grid == 1] = 1
        return self.probability_grid

    def make_main_color(self):
        main_colors = np.unique(self.colored_grid[np.where(self.waterway_grid == 1)])
        if len(main_colors) > 0:
            main_color = main_colors[0]
            self.colored_grid[np.isin(self.colored_grid, main_colors)] = main_color
            return main_color
        return -1

    def get_color_min_elevation_points(self):
        rows, cols = np.where(self.colored_grid > 0)
        num_rows, num_cols = self.colored_grid.shape
        min_elevation_points = {}
        for (row, col) in zip(rows, cols):
            color = self.colored_grid[row, col]
            elevation = self.elevation_grid[
                        max(row - 1, 0): min(row + 2, num_rows), max(col - 1, 0): min(col + 2, num_cols)
                        ].mean()
            color_info = min_elevation_points.setdefault(color, {'min_elevation': elevation, 'node': (row, col)})
            if elevation < color_info['min_elevation']:
                color_info['min_elevation'] = elevation
                color_info['node'] = (row, col)
        return {color_info['node'] for color_info in min_elevation_points.values()}

    def remove_waterways_out_of_basin(self):
        colors_to_remove = []
        min_elevation_points = self.get_color_min_elevation_points()
        main_colors = {self.main_color} if self.main_color > 0 else set()
        for color in np.unique(self.colored_grid):
            if color > 0:
                rows, cols = np.where(self.colored_grid == color)
                if self.basin_grid[rows, cols].sum()/len(rows) > .5:
                    main_colors.add(color)
        for row, col in min_elevation_points:
            if self.basin_grid[row, col] == 0:
                color = self.colored_grid[row, col]
                if color not in main_colors:
                    colors_to_remove.append(color)
        to_change = np.where(np.isin(self.colored_grid, colors_to_remove) | (self.basin_grid == 0))
        self.rounded_grid[to_change] = 0
        self.colored_grid[to_change] = 0
        self.probability_grid[to_change] = 0
        self.colored_grid[self.colored_grid < 0] = 0


    def scale_probability_grid(self, min_val = .1, max_val = .5, **kwargs):
        self.probability_grid = (self.probability_grid - min_val) / (max_val - min_val)
        self.probability_grid[self.probability_grid > 1] = 1
        self.probability_grid[self.probability_grid < 0] = 0


def remove_small_land(
        grid, small_land_count: int, color_count_neg: list,
        color_grid: np.ndarray, neg_elevation_difference: list,
        small_land_count_elevation: int=250, elevation_difference_max: int=3
):
    to_change = []
    num_changed = 0
    for color, count in enumerate(color_count_neg):
        color = -color
        if color < 0:
            elevation_difference = neg_elevation_difference[-color]
            if ((count < small_land_count) or
                    (count < small_land_count_elevation and elevation_difference < elevation_difference_max)):
                num_changed += 1
                to_change.append(color)
    if len(to_change) > 0:
        to_check = np.where(np.isin(color_grid, to_change))
        grid[to_check] = 1
    return grid


def remove_small_waterways(
        grid, small_waterways_count: int, color_count_pos: list, color_grid: np.ndarray, keep_colors: set
):
    to_change = []
    for color, count in enumerate(color_count_pos):
        if color > 0 and color not in keep_colors:
            if count < small_waterways_count:
                to_change.append(color)
    if len(to_change) > 0:
        grid[np.where(np.isin(color_grid, to_change))] = 0
    return grid


def post_connections_clean(new_grid, elevation_grid, waterway_grid):
    colored, _, neg_color_count, _, neg_elevation_difference = color_raster(new_grid, elevation_grid)
    new_grid = remove_small_land(
        grid=new_grid, small_land_count=4, color_grid=colored,
        neg_elevation_difference=neg_elevation_difference, color_count_neg=neg_color_count,
        small_land_count_elevation=10
    )
    rows, cols = np.where(waterway_grid == 1)
    new_grid[rows, cols] = 2
    if np.any(new_grid == 1):
        new_grid = thinner(new_grid, elevation_grid)
    new_grid[rows, cols] = 2
    new_copy = new_grid.copy()
    new_copy[rows, cols] = 0
    colored, pos_color_count, _, pos_elevation_difference, _ = color_raster(
        new_copy, elevation_grid
    )
    cleaned_grid = remove_small_waterways(
        grid=new_grid, small_waterways_count=4,
        color_count_pos=pos_color_count, color_grid=colored, keep_colors=set()
    )
    return cleaned_grid