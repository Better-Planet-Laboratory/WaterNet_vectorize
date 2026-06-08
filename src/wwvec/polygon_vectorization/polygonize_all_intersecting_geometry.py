import shapely
from wwvec.width_calculation.basin_polygonizer import basin_polygonizer
from wwvec.polygon_vectorization._tools import tt, time_elapsed, delete_directory_contents, SharedMemoryPool
from wwvec.paths import PolygonizedPaths, ppaths
import numpy as np
import geopandas as gpd
import pandas as pd
import warnings
import os
import time
from pathlib import Path
import gc
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def _merge_dfs(input_list: list) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    input_list : list
        A list of dictionaries containing 'hydro2_id' and 'stream_id' values.

    Returns
    -------
    merged_df : DataFrame
        A merged DataFrame containing the data from the specified paths.
    """
    paths = [val['paths'] for val in input_list]
    dfs_to_merge = []
    for path in paths:
        if path.save_path.exists():
            try:
                gdf = gpd.read_parquet(path.save_path)
                gdf['tdx_stream_id'] = path.stream_id
                gdf = gdf[gdf.threshold==0.2].reset_index(drop=True)
                dfs_to_merge.append(gdf)
            except Exception as e:
                print(e)
                print(f'Issue with {path}')
                continue

    if len(paths) > 0:
        merged_df = pd.concat(dfs_to_merge, ignore_index=True)
        return merged_df

def _run_for_basin_list(input_list):
    """
    Runs the mergining process for a list of basins (and the necessary inputs)
    """
    for inputs in input_list:
        # width_calculation(**inputs)
        try:
            basin_polygonizer(**inputs)
        except:
            x, y = inputs['basin_geometry'].centroid.coords[0]
            print('Unexplained Error, investigate further.')
            print(f'Stream ID: {inputs["stream_id"]}, Centroid: {x}, {y}')
        #     _save_exception_waterway(**inputs)
    time.sleep(.1)
    temp_df = _merge_dfs(input_list)
    if temp_df is not None:
        temp_path = ppaths.merged_temp/f'temp_{os.getpid()}.parquet'
        temp_df.to_parquet(temp_path)


def _open_hydro2_id_tdx_data(hydro2_id, polygon=None):
    print('Opening Basins')
    all_basins = gpd.read_file(
        ppaths.get_tdx_basin_file(hydro2_id), mask=polygon, include_fields=['streamID', 'geometry']
    ).reset_index(drop=True)
    all_basins['area'] = all_basins.area
    # all_basins['geometry'] = all_basins.buffer(0.001)
    all_basins = all_basins.sort_values(by='area', ascending=False).drop_duplicates('streamID', keep='first')
    all_basins = all_basins.set_index('streamID')
    return all_basins


def _make_basin_list_input_data(
        basin_gdf, input_list: list, overwrite: bool = False, hydro2_id: int = 0, thresholds: list[float] = None
) -> list[dict]:
    """
    Parameters
    ----------
    basin_gdf : GeoDataFrame

    input_list : list
        List that the input data will be appended to.

    overwrite : bool, optional
        Flag indicating whether to overwrite existing input data.
        Default is False.

    hydro2_id : int, optional
        ID of the hydrologic region. Default is 0.

    Returns
    -------
    list of dict
        List containing input data in the form of dictionaries.
        Each dictionary represents the inputs for a stream in the basin.
        The dictionary contains the following keys: basin_geometry, stream_geometry,
        old_target_id, old_source_ids, old_stream_order, hydro2_id, stream_id, overwrite.

    Notes
    -----
    This method iterates over the rows of the basin_stream_gdf GeoDataFrame and generates input data for each stream.
    The input data is then added to the input_list, which is returned at the end.
    """
    for stream_id, basin_geometry in zip(basin_gdf.index, basin_gdf.geometry):
        try:
            inputs = dict(
                basin_geometry=basin_geometry, thresholds=thresholds,
                paths=PolygonizedPaths(hydro2_id, stream_id), overwrite=overwrite,
                stream_id=stream_id
            )
            input_list.append(inputs)
        except:
            print(f'issue with {hydro2_id}, {stream_id}')
            continue
    return input_list


def make_all_intersecting_polygon(
        polygon: shapely.Polygon, save_path: Path, overwrite=False, num_proc=30
):
    """
    Makes all of the TDX-Hydro basins intersecting the input polygon, then merges that data.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon used for masking the hydrobasins data.

    save_path : Path
        The path where the final merged and fixed dataframe will be saved in Parquet format.

    overwrite : bool, optional
        Whether to overwrite existing files in the temporary merged temporary directory. Default is False.

    num_proc : int, optional
        Number of processes to use for parallelization. Default is 30.
    """
    shapely.prepare(polygon)
    delete_directory_contents(ppaths.merged_temp) if ppaths.merged_temp.exists() else ppaths.merged_temp.mkdir()
    hydro_level2 = gpd.read_file(ppaths.hydrobasins, mask=polygon)
    input_list = []
    print("Making inputs")
    s = tt()
    for hydro2_id in hydro_level2.HYBAS_ID:
        s = tt()
        all_streams = _open_hydro2_id_tdx_data(hydro2_id, polygon)
        input_list = _make_basin_list_input_data(
            all_streams, overwrite=overwrite, input_list=input_list, hydro2_id=hydro2_id
        )
        time_elapsed(s, 2)
    np.random.shuffle(input_list)
    input_chunks = np.array_split(input_list, max(len(input_list)//500, 4*num_proc))
    time_elapsed(s, 2)
    print(f"Making vectorized waterways, number of inputs {len(input_list)}")
    SharedMemoryPool(
        num_proc=num_proc, func=_run_for_basin_list, input_list=input_chunks,
        use_kwargs=False, sleep_time=0, terminate_on_error=False, print_progress=True
    ).run()
    print('Merging dataframes')
    merged_df = pd.concat([gpd.read_parquet(file) for file in ppaths.merged_temp.iterdir()], ignore_index=True)
    merged_df.to_parquet(save_path)
    return merged_df


def make_all_intersecting_hydrobasin_level_2_polygon(hydrobasin_id: int, save_path: Path, overwrite=False, num_proc=30):
    """
    Makes all of the TDX-Hydro basins that intersect the input hydrobasin level 2 id, then merges that data.

    Parameters
    ----------
    hydrobasin_id : int
        The ID of the hydrobasin.
    save_path : Path
        The path to save the polygon data.
    overwrite : bool, optional
        Whether to overwrite existing data at the save path. Default is False.
    num_proc : int, optional
        The number of processes to use for parallel execution. Default is 30.

    Returns
    -------
    pandas.DataFrame
        The merged dataframe containing the intersecting hydrobasin level 2 polygons.
    """
    s = tt()
    delete_directory_contents(ppaths.merged_temp) if ppaths.merged_temp.exists() else ppaths.merged_temp.mkdir()
    all_streams = _open_hydro2_id_tdx_data(hydrobasin_id)
    inputs_list = []
    inputs_list = _make_basin_list_input_data(
        all_streams, overwrite=overwrite, input_list=inputs_list, hydro2_id=hydrobasin_id
    )
    time_elapsed(s, 2)
    np.random.shuffle(inputs_list)
    input_chunks = np.array_split(inputs_list, max(len(inputs_list)//500, min(4*num_proc, len(inputs_list))))
    time_elapsed(s, 2)
    print(f"Making vectorized waterways, number of inputs {len(inputs_list)}")
    SharedMemoryPool(
        num_proc=num_proc, func=_run_for_basin_list, input_list=input_chunks,
        use_kwargs=False, sleep_time=0, terminate_on_error=False, print_progress=True
    ).run()
    print('Merging dataframes')
    del(all_streams, inputs_list, input_chunks)
    gc.collect()
    s = tt()
    merged_df = pd.concat([gpd.read_parquet(file) for file in ppaths.merged_temp.iterdir()], ignore_index=True)
    # temp_files = list(ppaths.merged_temp.iterdir())
    # merged_df = pd.concat([gpd.read_parquet(file) for file in temp_files[:200]], ignore_index=True)
    # for i in range(200, len(temp_files), 200):
    #     print(i, i + 200, len(temp_files))
    #     merged_df = pd.concat(
    #         [merged_df] + [gpd.read_parquet(file) for file in temp_files[i: min(i + 200, len(temp_files))]],
    #         ignore_index=True
    #     )
    time_elapsed(s, 2)
    merged_df.to_parquet(save_path)
    return merged_df
