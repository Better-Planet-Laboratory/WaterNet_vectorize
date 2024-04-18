from functools import cached_property
import shapely
import numpy as np
from rasterio import features
from wwvec.paths import BasinPaths
from wwvec.basin_vectorization.cut_bbox_raster import make_bbox_raster
from wwvec.basin_vectorization.thin_grid import thinner
from wwvec.basin_vectorization.components import find_raster_components
import xarray as xr


class BasinData:
    """
    BasinData is a class that represents data related to a basin.
     It initializes the instance variables by cutting and merging the model
      waterway probability and elevation data for the basin.
       It also performs various operations on the data to generate grids and components related to the basin.

    Methods:
    - __init__(self, basin_geometry: shapely.Polygon, stream_geometry: shapely.LineString,
     paths: BasinPaths, bbox_buffer: float=0.005, **kwargs):
        Initializes the BasinData instance with the given parameters and performs the necessary data operations.
    - cut_basin_data(self, stream_buffer=.0001, **kwargs) -> (xr.DataArray, xr.DataArray, np.ndarray, np.ndarray):
        Cuts and merges the model waterway probability and elevation data for the basin,
         and returns the resulting arrays.
    - make_rounded_grid(self, round_value=.5, **kwargs) -> np.ndarray:
        Creates a rounded version of the probability grid based on the given round_value threshold.
    - probability_grid(self) -> np.ndarray:
        Treated as an instance variable. Creates a numpy array copy of the basin_probability xarray.DataArray and sets
         the tdx-waterways to have a probability of 1 based on the burned waterway raster.
    - connected_grid(self) -> np.ndarray:
        Treated as an instance variable. Creates a copy of the rounded_grid,
         which will be the mutable grid used to connect the disconnected waterways.
    - find_main_component(self) -> int:
        Finds the main component of the tdx-waterways and sets all tdx-waterways to have the same component.
    - find_component_min_elevation_points(self) -> Set[(int, int)]:
        Finds the minimum elevation points for each component and returns them as a set of (row, col) tuples.
    - remove_waterways_out_of_basin(self):
        Removes waterways that are outside of the basin based on the component grid and basin grid.
    - make_weight_grid(self, min_val: float = .1, max_val: float = .5, **kwargs):
        rescales the probability grid using the min_val and max_val inputs,
         then applies -np.log2 to the positive values.

    Instance Variables:
    - bbox_buffer: Buffer distance for the bounding box of the basin.
    - paths: Object that contains the paths for the basin data.
    - basin_geometry: shapely.Polygon representing the geometry of the basin.
    - stream_geometry: shapely.LineString representing the geometry of the stream.
    - grid_bbox: Bounding box of the basin geometry with buffer.
    - basin_probability: xarray.DataArray representing the basin probability.
    - basin_elevation: xarray.DataArray representing the basin elevation.
    - basin_grid: numpy.ndarray representing the basin geometry grid.
    - waterway_grid: numpy.ndarray representing the waterway grid.
    - elevation_grid: numpy.ndarray representing the elevation grid.
    - probability_grid: numpy.ndarray representing the probability grid.
    - rounded_grid: numpy.ndarray representing the rounded probability grid.
    - component_grid: numpy.ndarray representing the component grid.
    - main_component: Integer value representing the main component of the tdx-waterways.
    - weight_grid: The cells are given a weight based on how strongly the model thinks they are water.
    - min_value: The minimum acceptable waterway value for the cell to be considered as a node in the connection process
    """
    def __init__(
            self, basin_geometry: shapely.Polygon,
            stream_geometry: shapely.LineString,
            paths: BasinPaths, bbox_buffer: float=0.005,
            **kwargs
    ):
        # Buffer the bounding box a tiny amount so no points land exactly on the boundary
        self.bbox_buffer = bbox_buffer
        """Buffer distance for the bounding box of the basin."""
        self.paths = paths
        self.basin_geometry = basin_geometry
        self.stream_geometry = stream_geometry
        bbox = tuple(self.basin_geometry.bounds)
        self.grid_bbox = (bbox[0] - bbox_buffer, bbox[1] - bbox_buffer, bbox[2] + bbox_buffer, bbox[3] + bbox_buffer)
        self.basin_probability, self.basin_elevation, self.basin_grid, self.waterway_grid\
            = self.cut_basin_data(**kwargs)
        self.elevation_grid = self.basin_elevation[0].to_numpy().astype(np.int16)
        self.rounded_grid = self.make_rounded_grid(**kwargs)
        self.component_grid, *_ = find_raster_components(self.rounded_grid, self.elevation_grid)
        self.main_component = self.find_main_component()
        self.remove_waterways_out_of_basin()
        self.component_grid, *_ = find_raster_components(self.rounded_grid, self.elevation_grid)
        self.main_component = self.find_main_component()
        self.weight_grid = self.make_weight_grid(**kwargs)

    def cut_basin_data(
            self, stream_buffer=.0001, **kwargs
    ) -> (xr.DataArray, xr.DataArray, ):
        """
        Cuts and merges the model waterway probability and elevation data for the basin, and burns the tdx basin and
        waterway data to a raster (using the same resolution and bounding box for the cut waterway data).
        The basin bounding box has already been buffered slightly in the __init__ method, this extra space will be used
        to remove waterways that should be considered as a waterway in an adjacent basin. The waterways are buffered
        slightly, so they take up a bit more area in the burned raster.

        Parameters
        ----------
        stream_buffer : float, optional
            Buffer distance around stream geometry. Default is 0.0001.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        basin_probability : xarray.DataArray
            Array representing the basin probability.
        basin_elevation : xarray.DataArray
            Array representing the basin elevation.
        basin_geometry_grid : numpy.ndarray
            Array representing the basin geometry grid.
        waterway_grid : numpy.ndarray
            Array representing the waterway grid.

        """
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

    def make_rounded_grid(self, round_value=.5, **kwargs) -> np.ndarray:
        """
        Makes a rounded version of the probability grid, rounded at round_value.

        Parameters
        ----------
        round_value : float, optional
            The threshold value for rounding. Grid values greater than or equal to this value will be rounded up to 1,
             while grid values less than this value will be rounded down to 0. Default is 0.5.
        **kwargs : optional
            Additional keyword arguments.

        Returns
        -------
        rounded_grid : ndarray
            A copy of the probability grid, where values have been rounded to 0 or 1
             based on the given round_value threshold.
        """
        self.rounded_grid = self.probability_grid.copy()
        self.rounded_grid[self.rounded_grid >= round_value] = 1
        self.rounded_grid[self.rounded_grid < round_value] = 0
        self.rounded_grid = self.rounded_grid.astype(np.int8)
        return self.rounded_grid

    @cached_property
    def probability_grid(self) -> np.ndarray:
        """
        Makes a numpy array copy of the basin_probability xarray.DataArray, and then sets the tdx-waterways to have
        a probability of 1, where the locations of the tdx-waterways are taken from the burned waterway raster.

        Returns:
            numpy.ndarray: The generated probability grid.

        """
        probability_grid = self.basin_probability.to_numpy()[0]
        probability_grid[self.waterway_grid == 1] = 1
        return probability_grid

    @cached_property
    def connected_grid(self) -> np.ndarray:
        """
        Creates a copy of the rounded_grid, which will be the mutable grid used to connect the disconnected waterways.

        Returns:
            np.ndarray: the connected grid.
        """
        new_grid = self.rounded_grid.copy()
        new_grid[new_grid > 0] = 1
        new_grid = new_grid.astype(np.int8)
        return new_grid

    def find_main_component(self) -> int:
        """
        Finds the integer value of the tdx-waterways component (there could potentially be more than 1 depending on
        the rasterization process), then sets all of the tdx-waterways to have the same component,
         and returns the main component.

        Returns
        -------
        main_component: int, the integer value corresponding the tdx-waterways, or -1 if there is no tdx-waterway.
        """
        main_components = np.unique(self.component_grid[np.where(self.waterway_grid == 1)])
        if len(main_components) > 0:
            main_component = main_components[0]
            self.component_grid[np.isin(self.component_grid, main_components)] = main_component
            return main_component
        return -1

    def find_component_min_elevation_points(self) -> set:
        """
        Finds the minimum elevation points for each component in the given component grid.

        Returns:
            set: A set of tuples representing the coordinates of the minimum elevation points for each component.

        """
        rows, cols = np.where(self.component_grid > 0)
        num_rows, num_cols = self.component_grid.shape
        min_elevation_points = {}
        for (row, col) in zip(rows, cols):
            component = self.component_grid[row, col]
            elevation = self.elevation_grid[
                        max(row - 1, 0): min(row + 2, num_rows), max(col - 1, 0): min(col + 2, num_cols)
                        ].mean()
            component_info = min_elevation_points.setdefault(
                component, {'min_elevation': elevation, 'node': (row, col)}
            )
            if elevation < component_info['min_elevation']:
                component_info['min_elevation'] = elevation
                component_info['node'] = (row, col)
        return {component_info['node'] for component_info in min_elevation_points.values()}

    def remove_waterways_out_of_basin(self):
        """
        Remove waterways out of the basin.

        Removes waterways that flow out of the basin by setting their corresponding values in the relevant grids to 0.

        Parameters:
        - self: Reference to the current object.

        Returns:
        None
        """
        components_to_remove = []
        min_elevation_points = self.find_component_min_elevation_points()
        main_components = {self.main_component} if self.main_component > 0 else set()
        for component in np.unique(self.component_grid):
            if component > 0:
                rows, cols = np.where(self.component_grid == component)
                if self.basin_grid[rows, cols].sum()/len(rows) > .5:
                    main_components.add(component)
        for row, col in min_elevation_points:
            if self.basin_grid[row, col] == 0:
                component = self.component_grid[row, col]
                if component not in main_components:
                    components_to_remove.append(component)
        to_change = np.where(np.isin(self.component_grid, components_to_remove) | (self.basin_grid == 0))
        self.rounded_grid[to_change] = 0
        self.component_grid[to_change] = 0
        self.probability_grid[to_change] = 0
        self.component_grid[self.component_grid < 0] = 0

    def make_weight_grid(self, min_val: float=.1, max_val: float=.5, **kwargs):
        """
        The weight grid will be used to make graph weights.
        We want anything the model mostly thinks as water to have a value of 1 (which will have a small weight),
        and any cells the model thinks isn't water to have a value of 0. Any cell with a weight of 0 will be excluded
        from the graph.

        Parameters
        ----------
        min_val : float, optional
            The minimum value to scale the probability grid. Default is 0.1.

        max_val : float, optional
            The maximum value to scale the probability grid. Default is 0.5.

        **kwargs
            Additional keyword arguments if necessary.
        """
        self.min_val = min_val
        self.weight_grid = self.probability_grid.copy()
        self.weight_grid = (self.weight_grid - min_val) / (max_val - min_val)
        self.weight_grid[self.weight_grid > 1] = 1
        self.weight_grid[self.weight_grid < 0] = 0
        self.weight_grid[self.weight_grid > 0] = -np.log2(self.weight_grid[self.weight_grid > 0])
        return self.weight_grid

def remove_small_land(
        grid: np.ndarray, small_land_count: int, negative_component_counts: list,
        component_grid: np.ndarray, negative_elevation_difference: list,
        small_land_count_elevation: int=250, elevation_difference_max: int=3
):
    """
    Remove small land components from a grid.

    Parameters
    ----------
    grid : np.ndarray
        The input grid.
    small_land_count : int
        The minimum count of negative components to consider as small land.
    negative_component_counts : list
        The counts of negative components.
    component_grid : np.ndarray
        The grid representing components.
    negative_elevation_difference : list
        The elevation differences of negative components.
    small_land_count_elevation : int, optional
        The count of negative components to consider as small land when elevation difference is small. Default is 250.
    elevation_difference_max : int, optional
        The maximum elevation difference to consider as small land when count is small. Default is 3.

    Returns
    -------
    np.ndarray
        The grid with small land components removed.
    """
    to_change = []
    num_changed = 0
    for component, count in enumerate(negative_component_counts):
        component = -component
        if component < 0:
            elevation_difference = negative_elevation_difference[-component]
            if ((count < small_land_count) or
                    (count < small_land_count_elevation and elevation_difference < elevation_difference_max)):
                num_changed += 1
                to_change.append(component)
    if len(to_change) > 0:
        to_check = np.where(np.isin(component_grid, to_change))
        grid[to_check] = 1
    return grid


def remove_small_waterways(
        grid: np.ndarray, small_waterways_count: int, positive_component_counts: list,
        component_grid: np.ndarray, components_to_keep: set
):
    """
    Parameters
    ----------
    grid : np.ndarray
        The grid representing the waterways.
    small_waterways_count : int
        The minimum count of waterways for it to be considered a small waterway.
    positive_component_counts : list
        A list containing the count of waterways for each component in the grid.
    component_grid : np.ndarray
        A grid representing the components of the waterways.
    components_to_keep : set
        A set containing the components that should not be removed.

    Returns
    -------
    np.ndarray
        The updated grid with small waterways removed.

    """
    to_change = []
    for component, count in enumerate(positive_component_counts):
        if component > 0 and component not in components_to_keep:
            if count < small_waterways_count:
                to_change.append(component)
    if len(to_change) > 0:
        grid[np.where(np.isin(component_grid, to_change))] = 0
    return grid


def post_connections_clean(new_grid: np.ndarray, elevation_grid: np.ndarray, waterway_grid: np.ndarray):
    """
    Parameters
    ----------
    new_grid : numpy.ndarray
        The grid representing the new connections.

    elevation_grid : numpy.ndarray
        The grid representing the elevation of the land.

    waterway_grid : numpy.ndarray
        The grid representing the waterways.

    Returns
    -------
    cleaned_grid : numpy.ndarray
        The grid with cleaned connections.
    """
    component_grid, _, negative_component_count, _, negative_elevation_difference = find_raster_components(
        new_grid, elevation_grid
    )
    new_grid = remove_small_land(
        grid=new_grid, small_land_count=4, component_grid=component_grid,
        negative_elevation_difference=negative_elevation_difference, negative_component_counts=negative_component_count,
        small_land_count_elevation=10
    )
    rows, cols = np.where(waterway_grid == 1)
    new_grid[rows, cols] = 2
    if np.any(new_grid == 1):
        new_grid = thinner(new_grid, elevation_grid)
    new_grid[rows, cols] = 2
    new_copy = new_grid.copy()
    new_copy[rows, cols] = 0
    component_grid, positive_component_counts, _, pos_elevation_difference, _ = find_raster_components(
        new_copy, elevation_grid
    )
    cleaned_grid = remove_small_waterways(
        grid=new_grid, small_waterways_count=4, positive_component_counts=positive_component_counts,
        component_grid=component_grid, components_to_keep=set()
    )
    return cleaned_grid