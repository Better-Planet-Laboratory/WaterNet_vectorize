import rioxarray as rxr
from rioxarray.merge import merge_arrays
import xarray as xr
from pathlib import Path
import shapely
import numpy as np
import rasterio as rio
from pyproj import Transformer
from pyproj.crs import CRS
import geopandas as gpd
import os


def name_to_box(file_name: str, buffer: float = 0.) -> shapely.box:
    """
    Parameters
    ----------
    file_name : str
        The name of the file from which to create the box.
    buffer : float, optional
        The buffer size to be added to the box dimensions. Default is 0.

    Returns
    -------
    shapely.box
        The bounding box created from the file name.

    """
    bbox = file_name.split('.tif')[0].split('_')[1:]
    bbox = [float(val) - buffer*(-1)**(ind//2) for ind, val in enumerate(bbox)]
    return shapely.box(*bbox)


def make_directory_gdf(dir_path: Path, use_name: bool = True) -> gpd.GeoDataFrame:
    """
    Makes a GeoDataFrame whose entries correspond to the tif files in dir_path,
    and whose geometry is the bounding box of the raster file in 4326. use_name assumes that the file has a name
    like bbox_{w}_{s}_{e}_{n}.tif where w,s,e,n are the bounding box values.
    """
    dir_name = dir_path.name
    parquet_path = dir_path/f'{dir_name}.parquet'
    if not parquet_path.exists():
        file_paths = list(dir_path.glob('*.tif'))
        dir_gdf = {'file_name': [], 'geometry': []}
        for file in file_paths:
            dir_gdf['file_name'].append(file.name)
            if use_name:
                dir_gdf['geometry'].append(name_to_box(file.name, 0))
            else:
                with rio.open(file) as rio_f:
                    crs = rio_f.crs
                    crs_4326 = CRS.from_epsg(4326)
                    bbox = tuple(rio_f.bounds)
                    if crs != crs_4326:
                        transformer = Transformer.from_crs(crs_from=crs, crs_to=crs_4326, always_xy=True)
                        bbox = transformer.transform_bounds(*bbox)
                    box = shapely.box(*bbox)
                dir_gdf['geometry'].append(box)
        dir_gdf = gpd.GeoDataFrame(dir_gdf, crs=4326)
        dir_gdf.to_parquet(parquet_path)
    else:
        dir_gdf = gpd.read_parquet(parquet_path)
        if len(dir_gdf) != len(list(dir_path.glob('*.tif'))):
            os.remove(parquet_path)
            dir_gdf = make_directory_gdf(dir_path)
    return dir_gdf


def get_file_paths_intersecting_bbox(
        bbox: tuple or list, base_dir: Path
) -> list[Path]:
    """
    Finds raster files in the base directory (base_dir) that intersects the entered bounding box (bbox).

    Parameters
    ----------
    bbox : tuple or list
        The bounding box coordinates in the form [x_min, y_min, x_max, y_max].
         The bounding box defines the region of interest for retrieving file paths.

    base_dir : Path
        The base directory where the files are located.
         The method will search for files in this directory that intersect with the given bounding box.

    Returns
    -------
    file_paths : list
        The list of file paths that intersect with the bounding box.

    """
    box = shapely.box(*bbox)
    try:
        directory_gdf = make_directory_gdf(base_dir, use_name=True)
    except:
        directory_gdf = make_directory_gdf(base_dir, use_name=False)
    intersects_bbox = directory_gdf[directory_gdf.intersects(box.buffer(0.01))]
    file_paths = [base_dir/file_name for file_name in intersects_bbox.file_name]
    # print(file_paths)
    return file_paths


def cut_and_merge_files(
        file_paths: list[Path], bbox: list or tuple
) -> xr.DataArray:
    """
    Cuts the raster files in file_paths to the entered bounding box, then merges that data

    Parameters
    ----------
    file_paths : list of Path
        List of file paths to be processed.

    bbox : tuple
        Bounding box coordinates in the order of west, south, east, and north.

    Returns
    -------
    xarray.DataArray
        Merged array resulting from cutting and merging the input files within the specified bounding box.
    """
    w, s, e, n = bbox
    temp_arrays = [rxr.open_rasterio(file_path) for file_path in file_paths]
    arrays = []
    for array in temp_arrays:
        if array.dtype in [np.float32, np.float64, np.float16]:
            array = array.rio.set_nodata(np.nan)
        else:
            array = array.rio.set_nodata(0)
        subarray = array[:, (s <= array.y) & (array.y <= n), (w <= array.x) & (array.x <= e)]
        bands, num_rows, num_cols = subarray.shape
        # Need to investigate further. When reprojecting some arrays with a small overlap (1 grid cell),
        # rioxarray/ rasterio removes that cell, making the array have 0 in its shape. so we will force num_rows>1
        # and num_cols>1.
        if num_rows > 1 and num_cols > 1:
            subarray = subarray.where(subarray < 1e30, other=array.rio.nodata)
            arrays.append(subarray)
    return merge_arrays(arrays, bounds=(w, s, e, n))


def make_bbox_raster(
        bbox: list or tuple, base_dir: Path
) -> xr.DataArray:
    """
    Finds files in base_dir that intersects bbox (using get_file_paths_intersecting_bbox), then cuts and merges that
    data using cut_and_merge_files

    Parameters
    ----------
    bbox : list
        A list containing the coordinates of the bounding box in the following format: [x_min, y_min, x_max, y_ max].

    base_dir : Path
        The base directory where the raster files are located.

    Returns
    -------
    raster : xarray.DataArray
        The raster image obtained by cutting and merging the raster files that intersect with the given bounding box.
    """
    file_paths = get_file_paths_intersecting_bbox(bbox, base_dir=base_dir)
    raster = cut_and_merge_files(file_paths, bbox)
    return raster


if __name__ == '__main__':
    # x, y = -162.34942503417992, 68.59222928209014
    # bbox = (x - .001, y - .001, x + .001, y + .001)
    bbox = (-157.39161215142718, 70.67360831541608, -157.30027493866345, 70.74238630903194)
    from wwvec.paths import ppaths
    based = ppaths.data/'model_outputs_zoom_level_6'
    raster = make_bbox_raster(bbox=bbox, base_dir=based)