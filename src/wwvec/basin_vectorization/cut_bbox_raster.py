import rioxarray as rxr
from rioxarray.merge import merge_arrays
from water.basic_functions import ppaths, Path
from water.make_country_waterways.cut_data import make_directory_gdf
import shapely
from rasterio.warp import Resampling
import matplotlib.pyplot as plt
import numpy as np


def get_file_paths_intersecting_bbox(bbox, base_dir: Path):
    box = shapely.box(*bbox)
    try:
        directory_gdf = make_directory_gdf(base_dir, use_name=True)
    except:
        directory_gdf = make_directory_gdf(base_dir, use_name=False)
    intersects_bbox = directory_gdf[directory_gdf.intersects(box.buffer(0.01))]
    file_paths = [base_dir/file_name for file_name in intersects_bbox.file_name]
    # print(file_paths)
    return file_paths


def cut_and_merge_files(file_paths: list, bbox):
    w, s, e, n = bbox
    temp_arrays = [rxr.open_rasterio(file_path) for file_path in file_paths]
    arrays = []
    for array in temp_arrays:
        if array.dtype in [np.float32, np.float64, np.float16]:
            array = array.rio.set_nodata(np.nan)
        else:
            array = array.rio.set_nodata(0)
        # array = array.rio.pad_box(w, s, e, n)
        subarray = array[:, (s <= array.y) & (array.y <= n), (w <= array.x) & (array.x <= e)]
        if 0 not in subarray.shape:
            subarray = subarray.where(subarray < 1e30, other=array.rio.nodata)
            arrays.append(subarray)
    return merge_arrays(arrays, bounds=(w, s, e, n))


def make_bbox_raster(bbox, base_dir):
    file_paths = get_file_paths_intersecting_bbox(bbox, base_dir=base_dir)
    # print(file_paths)
    raster = cut_and_merge_files(file_paths, bbox)
    return raster


if __name__ == '__main__':
    base = ppaths.country_data/'elevation'
    bbox = (34.77833333333335, 10.00750000000003, 34.86750000000003, 10.080000000000016)
    ras = make_bbox_raster(bbox, base_dir=base)
    # print(ras)
    # fps = get_file_paths_intersecting_bbox(bbox=bbox, base_dir=base)
    # print(fps)
    # ar = cut_and_merge_files(fps, bbox)
    # print(ar)
    # data = ar.to_numpy()
    plt.imshow(ras[0])