import shapely
import numpy as np
from rasterio import features
from wwvec.paths import PolygonizedPaths
from wwvec.basin_vectorization.cut_bbox_raster import make_bbox_raster
import xarray as xr
import json
import geopandas as gpd
import pandas as pd


class BasinPolygonizer:
    def __init__(
            self, basin_geometry: shapely.Polygon,
            paths: PolygonizedPaths, bbox_buffer: float = 0.005,
            thresholds: list[float] = None,
            **kwargs
    ):
        # Buffer the bounding box a tiny amount so no points land exactly on the boundary
        self.bbox_buffer = bbox_buffer
        """Buffer distance for the bounding box of the basin."""
        self.stream_id = paths.stream_id
        self.min_val = .1
        self.paths = paths
        # self.basin_geometry = basin_geometry.buffer(0.002)
        self.basin_geometry = basin_geometry
        bbox = tuple(self.basin_geometry.bounds)
        self.grid_bbox = (bbox[0] - bbox_buffer, bbox[1] - bbox_buffer, bbox[2] + bbox_buffer, bbox[3] + bbox_buffer)
        self.basin_probability, self.basin_grid = self.cut_basin_data(**kwargs)
        self.thresholds = thresholds if thresholds is not None else [0.2, 0.35, 0.5]

    def polygonize(self):
        gdfs = []
        probability = self.basin_probability.to_numpy()[0]
        crs = self.basin_probability.rio.crs
        transform = self.basin_probability.rio.transform()
        for threshold in self.thresholds:
            binary_array = (probability > threshold * 255).astype(np.uint8)
            shapes = list(features.shapes(
                binary_array, self.basin_grid*binary_array, connectivity=8, transform=transform
            ))
            json_dict = [json.dumps(shape[0]) for shape in shapes]
            geometries = [shapely.from_geojson(shape_json) for shape_json in json_dict]
            gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
            gdf['threshold'] = threshold
            gdfs.append(gdf)
        gdf = pd.concat(gdfs, ignore_index=True)
        gdf['tdx_stream_id'] = self.stream_id
        return gdf

    def cut_basin_data(
            self, stream_buffer=.0001, **kwargs
    ) -> (xr.DataArray, np.ndarray):
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
        basin_geometry_grid : numpy.ndarray
            Array representing the basin geometry grid.

        """
        bbox = self.grid_bbox
        basin_probability = make_bbox_raster(bbox, base_dir=self.paths.waterway_grids, min_pixels=5)
        basin_probability = basin_probability
        shape = basin_probability[0].shape
        transform = basin_probability.rio.transform()
        basin_geometry_grid = features.rasterize(
            shapes=[self.basin_geometry], out_shape=shape, transform=transform, all_touched=True
        )
        basin_geometry_grid[2:-2] += basin_geometry_grid[:-4] + basin_geometry_grid[4:]
        basin_geometry_grid[:, 2:-2] += basin_geometry_grid[:, :-4] + basin_geometry_grid[:, 4:]
        basin_geometry_grid[basin_geometry_grid > 0] = 1
        return basin_probability, basin_geometry_grid


def basin_polygonizer(
        basin_geometry: shapely.Polygon,
        paths: PolygonizedPaths, bbox_buffer: float = 0.005,
        thresholds: list[float] = None, overwrite: bool=False,
        **kwargs
) -> gpd.GeoDataFrame:
    if not paths.save_path.exists() or overwrite:
        gdf = BasinPolygonizer(basin_geometry, paths, bbox_buffer, thresholds, **kwargs).polygonize()
        gdf.to_parquet(paths.save_path)
    else:
        gdf = gpd.read_parquet(paths.save_path)
    return gdf

if __name__ == '__main__':
    path = PolygonizedPaths(hydro2_id=5020054880, stream_id=3483)
    basins_gdf = gpd.read_file(path.tdx_basins/'basin_5020054880.gpkg')
    basin_geom = basins_gdf[basins_gdf['streamID'] == 3483].reset_index(drop=True).geometry[0]
    gdf = basin_polygonizer(basin_geom, path, overwrite=True)