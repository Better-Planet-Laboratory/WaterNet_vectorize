import shapely
from wwvec.basin_vectorization.make_basin import run_for_basin
from wwvec.polygon_vectorization.clean_merged_data import StreamOrderFixer
from wwvec.polygon_vectorization._tools import tt, time_elapsed, delete_directory_contents, SharedMemoryPool
from wwvec.paths import BasinPaths, ppaths
import numpy as np
import geopandas as gpd
import pandas as pd
import warnings
import os
import time
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
    merged_df = pd.concat([gpd.read_parquet(path) for path in paths if path.exists()], ignore_index=True)
    return merged_df


def run_for_basin_list(input_list):
    for inputs in input_list:
        try:
            run_for_basin(**inputs)
        except:
            print(inputs['basin_geometry'].centroid)
    time.sleep(.1)
    temp_df = merge_dfs(input_list)
    temp_path = ppaths.merged_temp/f'temp_{os.getpid()}.parquet'
    temp_df.to_parquet(temp_path)


def make_all_intersecting_polygon(
        polygon: shapely.Polygon, save_path, overwrite=False, num_proc=30
):
    shapely.prepare(polygon)
    west, south, east, north = polygon.bounds
    bbox = [west-.5, south-.5, east+.5, north+.5]
    delete_directory_contents(ppaths.merged_temp) if ppaths.merged_temp.exists() else ppaths.merged_temp.mkdir()
    hydro_level2 = gpd.read_file(ppaths.hydrobasins)
    hydro_level2 = hydro_level2[hydro_level2.intersects(polygon)]
    inputs_list = []
    print("Making inputs")
    s = tt()
    for hydro2_id in hydro_level2.HYBAS_ID:
        streams_path = ppaths.tdx_streams/f'basin_{hydro2_id}.gpkg'
        basins_path = ppaths.tdx_basins/f'basin_{hydro2_id}.gpkg'
        all_streams = gpd.read_file(streams_path, bbox=bbox)
        all_basins = gpd.read_file(basins_path, bbox=bbox).reset_index(drop=True)
        basins_tree = shapely.STRtree(all_basins.geometry)
        all_basins = all_basins.loc[basins_tree.query(polygon, predicate='intersects')]
        stream_ids = all_basins.streamID.unique()
        all_streams = all_streams[all_streams.LINKNO.isin(stream_ids)]
        all_streams = all_streams.set_index('LINKNO')
        for stream_id in all_basins.streamID.unique():
            basin_geometries = all_basins[all_basins.streamID == stream_id].reset_index(drop=True)
            if len(basin_geometries) > 0:
                basin_geometries['area'] = basin_geometries.area
                basin_geometries = basin_geometries.sort_values(by='area', ascending=False).reset_index(drop=True)
            basin_geometry = basin_geometries.geometry[0]
            old_stream_order = all_streams.loc[stream_id, 'strmOrder']
            old_target = all_streams.loc[stream_id, 'DSLINKNO']
            old_sources = [all_streams.loc[stream_id, 'USLINKNO1'], all_streams.loc[stream_id, 'USLINKNO2']]
            try:
                stream_geometry = all_streams.loc[stream_id, 'geometry']
                inputs = dict(
                    basin_geometry=basin_geometry, stream_geometry=stream_geometry, old_target_id=old_target,
                    old_source_ids=old_sources, old_stream_order=old_stream_order,
                    hydro2_id=hydro2_id, stream_id=stream_id, overwrite=overwrite
                )
                inputs_list.append(inputs)
            except:
                print(f'issue with {hydro2_id}, {stream_id}')
                continue
    np.random.shuffle(inputs_list)
    input_chunks = np.array_split(inputs_list, max(len(inputs_list)//500, 4*num_proc))
    time_elapsed(s, 2)
    print(f"Making vectorized waterways, number of inputs {len(inputs_list)}")
    SharedMemoryPool(
        num_proc=num_proc, func=run_for_basin_list, input_list=input_chunks,
        use_kwargs=False, sleep_time=0, terminate_on_error=False, print_progress=True
    ).run()
    print('Merging dataframes')
    merged_df = pd.concat([gpd.read_parquet(file) for file in ppaths.merged_temp.iterdir()], ignore_index=True)
    stream_order_fixer = StreamOrderFixer(merged_df)
    stream_order_fixer.investigate_all()
    stream_order_fixer.add_fixed_stream_order()
    stream_order_fixer.init_df.to_parquet(save_path)
    return merged_df



#
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     ctry = 'rwanda'
#     # countries = ['rwanda', 'ivory_coast', 'ethiopia', 'zambia', 'tanzania', 'kenya']
#     # countries = ['equatorial_guinea']
#     # countries = ['rwanda']
#     # from water.basic_functions import get_country_polygon
#     # for ctry in countries:
#     #     ctry_parquet_path = ppaths.data/f'africa_basins_countries/{ctry}_waterways_new.parquet'
#     #     polygon = get_country_polygon(ctry)
#     #     gdf = make_all_intersecting_polygon(polygon=polygon, save_path=ctry_parquet_path, overwrite=True)
#         # ctry_gpkg_path = ppaths.data/f'africa_basins_countries/{ctry}_waterways.gpkg'
#         # gdf = gpd.read_parquet(ctry_parquet_path)
#
#         # gdf.to_file(ctry_gpkg_path, driver='GPKG')
#     # countries = ['uganda']
#     from pyproj import Geod
#     geod = Geod(ellps='WGS84')
#     # print('opening')
#     # gdf = gpd.read_parquet(ppaths.data/f'africa_basins_countries/{ctry}_waterways_new.parquet')
#     gdf = gpd.read_parquet(ppaths.data/f'basins_level_2/1020000010_waterways.parquet')
#     # s = tt()
#     # print('making stream_order_fixer')
#     # stream_order_fixer = StreamOrderFixer(gdf)
#     # time_elapsed(s)
#     # s = tt()
#     # print('investigating')
#     # stream_order_fixer.investigate_all()
#     # time_elapsed(s)
#     # s = tt()
#     # print('adding fixed')
#     # stream_order_fixer.add_fixed_stream_order()
#     # time_elapsed(s)
#     # df = stream_order_fixer.init_df
#     geom_length = np.frompyfunc(lambda x: geod.geometry_length(x), nin=1, nout=1)
#     gdf['length'] = geom_length(gdf.geometry.to_numpy())
#     print(gdf.stream_order.max(), gdf.fixed_stream_order.max())
#
#     print(gdf.groupby(['from_tdx', 'stream_order'])['length'].sum()/1000)
#     # africa_basins = gpd.read_file(ppaths.data/'basins/hybas_af_lev01-12_v1c/hybas_af_lev02_v1c.shp')
#     # from multiprocessing import Process
#     # world_basins = gpd.read_parquet(ppaths.data/'basins/hybas_all_level_2.parquet')
#     # st = tt()
#     # for ind, (id, poly) in enumerate(zip(world_basins.HYBAS_ID, world_basins.geometry)):
#     #     s = tt()
#     #     print(f'Working on {id} ({ind+1}/{len(world_basins)})')
#     #     poly = poly.buffer(-.00001)
#     #     save_path = ppaths.data/f'basins_level_2/{id}_waterways.parquet'
#     #     if not save_path.exists():
#     #         p = Process(
#     #             target=make_all_intersecting_polygon,
#     #             kwargs=dict(polygon=poly, save_path=save_path, num_proc=28, overwrite=False)
#     #         )
#     #         p.start()
#     #         p.join()
#     #         p.close()
#     #         break
#     #         # make_all_intersecting_polygon(polygon=poly, save_path=save_path, num_proc=20, overwrite=False)
#     #     time_elapsed(s, 2)
#     # time_elapsed(st, 2)
#
#
#     # countries = ['uganda']
#     # from pathlib import Path
#     # for file in (ppaths.data/f'africa_basins_level_2').iterdir():
#     #     gdf = gpd.read_parquet(file)
#     #     save_path = Path(str(file).replace('.parquet', '.gpkg'))
#     #     gdf.to_file(save_path, driver='GPKG')
#
#
#     # for ctry in countries:
#     #     ctry_parquet_path = ppaths.data/f'africa_basins_countries/{ctry}_waterways.parquet'
#     #     ctry_gpkg_path = ppaths.data/f'africa_basins_countries/{ctry}_waterways.gpkg'
#     #     gdf = gpd.read_parquet(ctry_parquet_path)
#     #     gdf.to_file(ctry_gpkg_path, driver='GPKG')