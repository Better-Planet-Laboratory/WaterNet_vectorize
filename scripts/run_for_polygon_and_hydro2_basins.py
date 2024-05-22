import shapely
from wwvec.polygon_vectorization.make_all_intersecting_polygon import (make_all_intersecting_polygon,
                                                                       make_all_intersecting_hydrobasin_level_2_polygon)
from wwvec.paths import ppaths
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import warnings
import os
import time
from functools import cached_property
warnings.filterwarnings("ignore", category=RuntimeWarning)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # ctry = 'rwanda'
    # countries = ['rwanda', 'ivory_coast', 'uganda', 'ethiopia', 'zambia', 'tanzania', 'kenya']
    # # # countries = ['ethiopia', 'zambia', 'tanzania', 'kenya']
    # #
    # # # countries = ['equatorial_guinea']
    # # # countries = ['egypt']
    countries = ['rwanda']

    from water.basic_functions import get_country_polygon, printdf, tt, time_elapsed
    from pyproj import Geod
    from pathlib import Path
    usa_states = gpd.read_parquet(ppaths.country_lookup_data/'usa_states.parquet')
    hu4_hulls = gpd.read_parquet(Path('/ilus/data/waterway_data/hu4_hulls.parquet'))

    # printdf(usa_states)
    # geod = Geod(ellps='WGS84')
    # save_dir = ppaths.data/'usa_states_waterways'


    save_dir = Path('/ilus/data/waterway_data/hu4_model')
    save_dir.mkdir(exist_ok=True)
    for name, polygon in zip(usa_states.hu4_index, usa_states.geometry):
        ctry_parquet_path = save_dir/f'hu4_{name}_model.parquet'
        # polygon = get_country_polygon(ctry)
        if not ctry_parquet_path.exists() and name != 'Hawaii':
            print(f'Working on {name}')
            gdf = make_all_intersecting_polygon(
                polygon=polygon, save_path=ctry_parquet_path, overwrite=False, num_proc=28
            )
            print('\n')
        # gdf = gpd.read_parquet(ctry_parquet_path)
        # print(ctry)
        # # printdf(gdf)
        # gdf['length_km'] = gdf['geometry'].apply(geod.geometry_length)/1000
        # printdf(gdf.groupby(['from_tdx'])[['length_km']].sum())
        # print('')
        # printdf(gdf.groupby(['from_tdx', 'stream_order'])[['length_km']].sum(), 100)
        # print('\n'*2)
    # basin_dir = ppaths.data/'basins_level_2'
    # basin_dir_with_len = basin_dir/'basins_level_2_with_length'
    # basin_dir_with_len.mkdir(exist_ok=True)
    # length_gdfs = []
    # basin_paths = list(basin_dir.iterdir())
    # geod_geom_len = np.frompyfunc(lambda x: geod.geometry_length(x), 1, 1)
    # st = tt()
    # for ind, basin_path in enumerate(basin_paths):
    #     print(f'Working on {basin_path.name} ({ind+1}/{len(basin_paths)})')
    #     s = tt()
    #     print('  Opening data')
    #     gdf = gpd.read_parquet(basin_path)
    #     time_elapsed(s, 4)
    #     print('  Finding Length')
    #     gdf['length_m'] = np.round(geod_geom_len(gdf.geometry)).astype(np.uint32)
    #     gdf.to_parquet(basin_dir_with_len/basin_path.name)
    #     gdf['length_km'] = gdf['length_m']/1000
    #
    #     time_elapsed(s, 4)
    #     printdf(gdf.groupby(['from_tdx'])[['length_km']].sum())
    #     print('')
    #     gdf1 = gdf.groupby(['from_tdx', 'stream_order'])[['length_km']].sum()
    #     printdf(gdf1, 100)
    #     gdf1 = gdf1.reset_index()
    #     gdf1['basin_id'] = int(basin_path.stem)
    #     length_gdfs.append(gdf1)
    #     time_elapsed(st)
    #     print('\n'*2)
    # length_gdf = pd.concat(length_gdfs)


    # save_dir = ppaths.data/'basins_level_2'
    # save_dir.mkdir(exist_ok=True)
    # world_basins = gpd.read_parquet(ppaths.data/'basins/hybas_all_level_2.parquet')
    # from water.basic_functions import printdf
    # world_basins['area_percent'] = 100*world_basins.UP_AREA / world_basins.UP_AREA.sum()
    # world_basins['region'] = world_basins.HYBAS_ID.apply(lambda x: int(str(x)[0]))
    # printdf(world_basins[['HYBAS_ID', 'UP_AREA', 'area_percent']], 100)
    # printdf(world_basins.groupby('region')[['area_percent']].sum(), 100)
    # printdf(world_basins, 100)
    # for id in [id for id in world_basins.HYBAS_ID if 7020000000<id<8000000010]:
    #     print(f'working on {id}')
    #     gdf = gpd.read_parquet(ppaths.tdx_basins/f'basin_{id}.parquet')
    #     gdf.to_file(ppaths.tdx_basins/f'basin_{id}.gpkg', driver='GPKG')
    # for id in world_basins.HYBAS_ID:
    #     save_path = save_dir/f'{id}.parquet'
    #     if not save_path.exists():
    #         print(f'\nWorking on {id}')
    #         try:
    #             make_all_intersecting_hydrobasin_level_2_polygon(
    #                 hydrobasin_id=id, save_path=save_path, num_proc=30
    #             )
    #         except:
    #             print(f'Issue with {id}')
    #         print('')
    #     else:
    #         print(f'{id} already Complete')



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