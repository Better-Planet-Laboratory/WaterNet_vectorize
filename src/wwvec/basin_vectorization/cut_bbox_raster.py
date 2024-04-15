import rioxarray as rxr
from rioxarray.merge import merge_arrays
import xarray as xr
from water.basic_functions import ppaths, Path
from water.make_country_waterways.cut_data import make_directory_gdf
import shapely
from rasterio.warp import Resampling
import matplotlib.pyplot as plt
import numpy as np


def get_file_paths_intersecting_bbox(
        bbox: tuple or list, base_dir: Path
) -> list[Path]:
    """
    Finds raster files in the base directory (base_dir) that intersects the entered bounding box (bbox).

    Parameters
    ----------
    bbox : tuple or list
        The bounding box coordinates in the form [xmin, ymin, xmax, ymax].
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
        if 0 not in subarray.shape:
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
        A list containing the coordinates of the bounding box in the following format: [xmin, ymin, xmax, ymax].

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
