import shapely
from wwvec.basin_vectorization.make_basin import run_for_basin
from wwvec.polygon_vectorization.clean_merged_data import fix_merged_dfs
from wwvec.polygon_vectorization._tools import tt, time_elapsed, delete_directory_contents, SharedMemoryPool
from wwvec.paths import BasinPaths, ppaths
import numpy as np
import geopandas as gpd
import pandas as pd
import warnings
import os
import time
from pathlib import Path
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def merge_dfs(input_list: list) -> gpd.GeoDataFrame:
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
    paths = [BasinPaths(hydro2_id=val['hydro2_id'], stream_id=val['stream_id']).save_path for val in input_list]
    if len(paths) > 0:
        merged_df = pd.concat([gpd.read_parquet(path) for path in paths if path.exists()], ignore_index=True)
        return merged_df


def save_exception_waterway(
        stream_id: int, hydro2_id: int, old_stream_order: int, old_target_id: int,
        old_source_ids: list[int], stream_geometry: shapely.LineString, overwrite: bool, **kwargs
):
    """Saves waterway in case of an exception in run_for_basin"""
    save_path = BasinPaths(stream_id=stream_id, hydro2_id=hydro2_id).save_path
    if not save_path.exists() or overwrite:
        gpd.GeoDataFrame(
            [{
                'stream_id': stream_id, 'from_tdx': True,
                'source_stream_ids': np.array([], dtype=np.int32), 'target_stream_id': -1,
                'stream_order': old_stream_order, 'tdx_stream_id': stream_id,
                'tdx_target_id': old_target_id, 'geometry': stream_geometry,
                'tdx_source_ids': np.array(old_source_ids, dtype=np.int32),
            }], crs=4326
        ).to_parquet(save_path)


def run_for_basin_list(input_list):
    """
    Runs the mergining process for a list of basins (and the necessary inputs)
    """
    for inputs in input_list:
        # run_for_basin(**inputs)
        try:
            run_for_basin(**inputs)
        except IndexError:
            x, y = inputs['basin_geometry'].centroid.coords[0]
            print('Index Error, likely no waterway or elevation data at this basin.')
            print(f'Stream ID: {inputs["stream_id"]}, Centroid: {x}, {y}')
            save_exception_waterway(**inputs)
        except:
            x, y = inputs['basin_geometry'].centroid.coords[0]
            print('Unexplained Error, investigate further.')
            print(f'Stream ID: {inputs["stream_id"]}, Centroid: {x}, {y}')
        #     save_exception_waterway(**inputs)
    time.sleep(.1)
    temp_df = merge_dfs(input_list)
    if temp_df is not None:
        temp_path = ppaths.merged_temp/f'temp_{os.getpid()}.parquet'
        temp_df.to_parquet(temp_path)


def open_hydro2_id_tdx_data(hydro2_id, polygon=None):
    """
    Parameters
    ----------
    hydro2_id : str
        The Hydrobasins level 2 ID of the data to be opened.

    polygon : optional
        The polygon mask to be applied to the data. If left as None, all of the data will be used.

    Returns
    -------
    all_streams : GeoDataFrame
        The streams data extracted from the TDX stream network file for the given Hydro2 ID and within the specified polygon mask.
        The data includes the following fields: LINKNO, strmOrder, DSLINKNO, USLINKNO1, USLINKNO2, and geometry.

    Notes
    -----
    - The function reads the TDX stream network file and the TDX basin file for the given Hydro2 ID and applies the polygon mask if provided.
    - The function joins the streams data with the basins data based on the LINKNO field.
    - The resulting GeoDataFrame contains the joined data with an added 'basin' field representing the basin geometry.

    """
    print('Opening Streams')
    all_streams = gpd.read_file(
        ppaths.get_tdx_stream_network_file(hydro2_id), mask=polygon,
        include_fields=['LINKNO', 'strmOrder', 'DSLINKNO', 'USLINKNO1', 'USLINKNO2', 'geometry']
    )
    all_streams = all_streams[all_streams.length > 0]
    all_streams = all_streams.reset_index(drop=True)
    print('Opening Basins')
    all_basins = gpd.read_file(
        ppaths.get_tdx_basin_file(hydro2_id), mask=polygon, include_fields=['streamID', 'geometry']
    ).reset_index(drop=True)
    all_basins['area'] = all_basins.area
    all_basins = all_basins.sort_values(by='area', ascending=False).drop_duplicates('streamID', keep='first')
    all_basins = all_basins.set_index('streamID')
    all_basins['basin'] = all_basins.geometry
    all_streams = all_streams.set_index('LINKNO')
    all_streams = all_streams.join(all_basins[['basin']], how='inner')
    return all_streams


def make_basin_list_input_data(
        basin_stream_gdf, input_list: list, overwrite: bool = False, hydro2_id: int = 0
) -> list[dict]:
    """
    Parameters
    ----------
    basin_stream_gdf : GeoDataFrame
        GeoDataFrame containing information about basin streams.
        Required columns: index, strmOrder, DSLINKNO, USLINKNO1, USLINKNO2, basin, geometry.

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
    for stream_id, old_stream_order, old_target, old_source1, old_source2, basin_geometry, stream_geometry in zip(
            basin_stream_gdf.index, basin_stream_gdf.strmOrder, basin_stream_gdf.DSLINKNO, basin_stream_gdf.USLINKNO1,
            basin_stream_gdf.USLINKNO2,
            basin_stream_gdf.basin, basin_stream_gdf.geometry
    ):
        old_sources = [old_source1, old_source2] if old_source1 != -1 else []
        try:
            stream_geometry = basin_stream_gdf.loc[stream_id, 'geometry']
            inputs = dict(
                basin_geometry=basin_geometry, stream_geometry=stream_geometry, old_target_id=old_target,
                old_source_ids=old_sources, old_stream_order=old_stream_order,
                hydro2_id=hydro2_id, stream_id=stream_id, overwrite=overwrite
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
    Makes all of the basins intersecting the input polygon, then merges that data.

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
        all_streams = open_hydro2_id_tdx_data(hydro2_id, polygon)
        input_list = make_basin_list_input_data(
            all_streams, overwrite=overwrite, input_list=input_list, hydro2_id=hydro2_id
        )
        time_elapsed(s, 2)
    np.random.shuffle(input_list)
    input_chunks = np.array_split(input_list, max(len(input_list)//500, 4*num_proc))
    time_elapsed(s, 2)
    print(f"Making vectorized waterways, number of inputs {len(input_list)}")
    SharedMemoryPool(
        num_proc=num_proc, func=run_for_basin_list, input_list=input_chunks,
        use_kwargs=False, sleep_time=0, terminate_on_error=False, print_progress=True
    ).run()
    print('Merging dataframes')
    merged_df = pd.concat([gpd.read_parquet(file) for file in ppaths.merged_temp.iterdir()], ignore_index=True)
    fixed_df = fix_merged_dfs(merged_df)
    fixed_df.to_parquet(save_path)
    return fixed_df


def make_all_intersecting_hydrobasin_level_2_polygon(hydrobasin_id: int, save_path: Path, overwrite=False, num_proc=30):
    """
    Makes all of the basins in the input hydrobasin level 2 id, then merges that data.

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
    all_streams = open_hydro2_id_tdx_data(hydrobasin_id)
    inputs_list = []
    inputs_list = make_basin_list_input_data(
        all_streams, overwrite=overwrite, input_list=inputs_list, hydro2_id=hydrobasin_id
    )
    time_elapsed(s, 2)
    np.random.shuffle(inputs_list)
    input_chunks = np.array_split(inputs_list, max(len(inputs_list)//500, 4*num_proc))
    time_elapsed(s, 2)
    print(f"Making vectorized waterways, number of inputs {len(inputs_list)}")
    SharedMemoryPool(
        num_proc=num_proc, func=run_for_basin_list, input_list=input_chunks,
        use_kwargs=False, sleep_time=0, terminate_on_error=False, print_progress=True
    ).run()
    print('Merging dataframes')
    s = tt()
    merged_df = pd.concat([gpd.read_parquet(file) for file in ppaths.merged_temp.iterdir()], ignore_index=True)
    time_elapsed(s, 2)
    print('Fixing dataframes')
    s = tt()
    fixed_df = fix_merged_dfs(merged_df)
    time_elapsed(s, 2)
    fixed_df.to_parquet(save_path)
    return merged_df


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # ctry = 'rwanda'
    # countries = ['rwanda', 'ivory_coast', 'uganda', 'ethiopia', 'zambia', 'tanzania', 'kenya']
    # countries = ['ethiopia', 'zambia', 'tanzania', 'kenya']

    # countries = ['equatorial_guinea']
    # countries = ['rwanda']
    # from water.basic_functions import get_country_polygon
    # save_dir = ppaths.data/'africa_basins_countries'
    # save_dir.mkdir(exist_ok=True)
    # for ctry in countries:
    #     ctry_parquet_path = save_dir/f'{ctry}_waterways_new.parquet'
    #     polygon = get_country_polygon(ctry)
    #     gdf = make_all_intersecting_polygon(polygon=polygon, save_path=ctry_parquet_path, overwrite=False, num_proc=20)
    #     # ctry_gpkg_path = ppaths.data/f'africa_basins_countries/{ctry}_waterways.gpkg'
        # gdf = gpd.read_parquet(ctry_parquet_path)
    # ctry = countries[0]
    # gdf = gpd.read_parquet(ppaths.data/f'africa_basins_countries/{ctry}_waterways_new.parquet')
    # for file in ppaths.tdx_streams.glob("*.gpkg"):
    #     print(file.name)
    #     print(gpd.read_file(file, rows=1).columns)
    #     print('')
        # gdf.to_file(ctry_gpkg_path, driver='GPKG')
    # countries = ['uganda']
    import pandas as pd
    pd.options.display.float_format = '{:,.0f}'.format
    countries = ['ethiopia', 'zambia', 'tanzania', 'kenya', 'rwanda', 'ivory_coast', 'uganda']
    # countries = ['rwanda', 'uganda']

    #
    from pyproj import Geod
    import warnings
    warnings.filterwarnings(category=UserWarning, action='ignore')
    geod = Geod(ellps='WGS84')
    dfs = []
    waterbodies = gpd.read_file(ppaths.data / 'waterbodies_africa.zip')
    waterbodies_array = waterbodies.geometry.to_list()
    for ctry in countries:
        print(f'opening {ctry} data')
        gdf = gpd.read_parquet(ppaths.data/f'africa_basins_countries/{ctry}_waterways_new.parquet')
        print(len(gdf))
        gdf_tree = shapely.STRtree(gdf.geometry.to_numpy())
        water_inds, tree_inds = gdf_tree.query(waterbodies_array, 'intersects')
        unique_gdf_inds = set(np.unique(tree_inds))
        keep_gdf_inds = [ind for ind in gdf.index if ind not in unique_gdf_inds]
        gdf = gdf.loc[keep_gdf_inds].reset_index(drop=True)
        print(len(gdf))

        gdf['country'] = ctry
        s = tt()
        geom_length_km = np.frompyfunc(lambda x: geod.geometry_length(x)/1000, nin=1, nout=1)
        gdf['length_km'] = geom_length_km(gdf.geometry.to_numpy())
        print(gdf.groupby(['from_tdx'])['length_km'].sum())
        print(gdf.groupby(['from_tdx', 'stream_order'])['length_km'].sum())
        print('')
        dfs.append(gdf)
    df = pd.concat(dfs, ignore_index=True)
    df1 = df.groupby(['country', 'from_tdx', 'stream_order'])[['length_km']].sum()
    df2 = df.groupby(['country', 'from_tdx'])[['length_km']].sum()
    df3 = df.groupby(['country'])[['length_km']].sum()

    # africa_basins = gpd.read_file(ppaths.data/'basins/hybas_af_lev01-12_v1c/hybas_af_lev02_v1c.shp')
    # from multiprocessing import Process
    # world_basins = gpd.read_parquet(ppaths.data/'basins/hybas_all_level_2.parquet')
    # st = tt()
    # for ind, (id, poly) in enumerate(zip(world_basins.HYBAS_ID, world_basins.geometry)):
    #     s = tt()
    #     print(f'Working on {id} ({ind+1}/{len(world_basins)})')
    #     poly = poly.buffer(-.00001)
    #     save_path = ppaths.data/f'basins_level_2/{id}_waterways.parquet'
    #     if not save_path.exists():
            # p = Process(
            #     target=make_all_intersecting_polygon,
            #     kwargs=dict(polygon=poly, save_path=save_path, num_proc=28, overwrite=False)
            # )
            # p.start()
            # p.join()
            # p.close()
            # break
            # make_all_intersecting_polygon(polygon=poly, save_path=save_path, num_proc=20, overwrite=False)
    #         make_all_intersecting_hydrobasin_level_2_polygon(id, save_path, overwrite=False)
    #     time_elapsed(s, 2)
    # time_elapsed(st, 2)

    #
    # countries = ['uganda']
    # from pathlib import Path
    # for file in (ppaths.data/f'africa_basins_level_2').iterdir():
    #     gdf = gpd.read_parquet(file)
    #     save_path = Path(str(file).replace('.parquet', '.gpkg'))
    #     gdf.to_file(save_path, driver='GPKG')
    #
    #
    # for ctry in countries:
    #     ctry_parquet_path = ppaths.data/f'africa_basins_countries/{ctry}_waterways.parquet'
    #     ctry_gpkg_path = ppaths.data/f'africa_basins_countries/{ctry}_waterways.gpkg'
    #     gdf = gpd.read_parquet(ctry_parquet_path)
    #     gdf.to_file(ctry_gpkg_path, driver='GPKG')