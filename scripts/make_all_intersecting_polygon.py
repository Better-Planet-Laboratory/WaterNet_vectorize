import shapely
from water.basic_functions import my_pool, ppaths, get_country_polygon, tt, time_elapsed, delete_directory_contents
from wwvec.basin_vectorization.make_basin import run_for_basin, merge_dfs
from wwvec.paths import BasinPaths
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import warnings
import os
import time
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_for_basin_list(input_list):
    for inputs in input_list:
        try:
            run_for_basin(**inputs)
        except:
            print(inputs)
            continue
    time.sleep(.1)
    temp_df = merge_dfs(input_list)
    temp_path = ppaths.country_data/f'temp_basin_dir/temp_{os.getpid()}.parquet'
    temp_df.to_parquet(temp_path)


def make_all_intersecting_polygon(
        polygon: shapely.Polygon, save_path, overwrite=False, num_proc=30
):
    shapely.prepare(polygon)
    west, south, east, north = polygon.bounds
    bbox = [west-.5, south-.5, east+.5, north+.5]
    temp_basin_path = ppaths.country_data/'temp_basin_dir'
    delete_directory_contents(temp_basin_path) if temp_basin_path.exists() else temp_basin_path.mkdir()
    hydro_level2 = gpd.read_parquet(ppaths.country_data/'basins/hybas_all_level_2.parquet')
    hydro_level2 = hydro_level2[hydro_level2.intersects(polygon)]
    inputs_list = []
    print("Making inputs")
    s = tt()
    for hydro2_id in hydro_level2.HYBAS_ID:
        streams_path = ppaths.country_data/f'tdx_streams/basin_{hydro2_id}.gpkg'
        basins_path = ppaths.country_data/f'tdx_basins/basin_{hydro2_id}.gpkg'
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
            try:
                stream_geometry = all_streams.loc[stream_id, 'geometry']
                inputs = dict(
                    basin_geometry=basin_geometry, stream_geometry=stream_geometry,
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
    my_pool(
        num_proc=num_proc, func=run_for_basin_list, input_list=input_chunks,
        use_kwargs=False, sleep_time=0, terminate_on_error=False
    )
    print('Merging dataframes')
    merged_df = pd.concat([gpd.read_parquet(file) for file in temp_basin_path.iterdir()], ignore_index=True)
    merged_df.to_parquet(save_path)


if __name__ == "__main__":
#     continents = ['af', 'ar', 'as', 'au', 'eu', 'gr', 'na', 'sa', 'si']
#     base_dir = ppaths.country_data/'basins'
#     dfs = []
#     for cont in continents:
#         cont_dir = base_dir / f'hybas_lake_{cont}_lev01-12_v1c'
#         basin_file = cont_dir / f'hybas_lake_{cont}_lev02_v1c.shp'
#         dfs.append(gpd.read_file(basin_file))
#     df = pd.concat(dfs, ignore_index=True)
#     df.to_parquet(base_dir/'hybas_all_level_2.parquet')
    # ctry = 'rwanda'
    # countries = ['rwanda', 'ivory_coast', 'ethiopia', 'zambia', 'tanzania', 'kenya']
    # countries = ['equatorial_guinea']

    # for ctry in countries:
    #     ctry_parquet_path = ppaths.country_data/f'africa_basins_countries/{ctry}_waterways.parquet'
    #     ctry_gpkg_path = ppaths.country_data/f'africa_basins_countries/{ctry}_waterways.gpkg'
    #     gdf = gpd.read_parquet(ctry_parquet_path)
    #     gdf.to_file(ctry_gpkg_path, driver='GPKG')
    # countries = ['uganda']

    # africa_basins = gpd.read_file(ppaths.country_data/'basins/hybas_af_lev01-12_v1c/hybas_af_lev02_v1c.shp')
    from multiprocessing import Process
    world_basins = gpd.read_parquet(ppaths.country_data/'basins/hybas_all_level_2.parquet')
    st = tt()
    for ind, (id, poly) in enumerate(zip(world_basins.HYBAS_ID, world_basins.geometry)):
        s = tt()
        print(f'Working on {id} ({ind+1}/{len(world_basins)})')
        poly = poly.buffer(-.00001)
        save_path = ppaths.country_data/f'basins_level_2/{id}_waterways.parquet'
        if not save_path.exists():
            p = Process(
                target=make_all_intersecting_polygon,
                kwargs=dict(polygon=poly, save_path=save_path, num_proc=20, overwrite=False)
            )
            p.start()
            p.join()
            p.close()
            # make_all_intersecting_polygon(polygon=poly, save_path=save_path, num_proc=20, overwrite=False)
        time_elapsed(s, 2)
    time_elapsed(st, 2)
    # countries = ['uganda']
    # from pathlib import Path
    # for file in (ppaths.country_data/f'africa_basins_level_2').iterdir():
    #     gdf = gpd.read_parquet(file)
    #     save_path = Path(str(file).replace('.parquet', '.gpkg'))
    #     gdf.to_file(save_path, driver='GPKG')


    # for ctry in countries:
    #     ctry_parquet_path = ppaths.country_data/f'africa_basins_countries/{ctry}_waterways.parquet'
    #     ctry_gpkg_path = ppaths.country_data/f'africa_basins_countries/{ctry}_waterways.gpkg'
    #     gdf = gpd.read_parquet(ctry_parquet_path)
    #     gdf.to_file(ctry_gpkg_path, driver='GPKG')