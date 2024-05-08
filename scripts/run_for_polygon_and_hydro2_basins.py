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
    # # countries = ['ethiopia', 'zambia', 'tanzania', 'kenya']
    #
    # # countries = ['equatorial_guinea']
    # # countries = ['egypt']
    # from water.basic_functions import get_country_polygon
    # save_dir = ppaths.data/'africa_basins_countries'
    # save_dir.mkdir(exist_ok=True)
    # for ctry in countries:
    #     ctry_parquet_path = save_dir/f'{ctry}_waterways_new.parquet'
    #     polygon = get_country_polygon(ctry)
    #     gdf = make_all_intersecting_polygon(polygon=polygon, save_path=ctry_parquet_path, overwrite=False, num_proc=30)

    save_dir = ppaths.data/'basins_level_2'
    save_dir.mkdir(exist_ok=True)
    world_basins = gpd.read_parquet(ppaths.data/'basins/hybas_all_level_2.parquet')
    # from water.basic_functions import printdf
    # world_basins['area_percent'] = 100*world_basins.UP_AREA / world_basins.UP_AREA.sum()
    # world_basins['region'] = world_basins.HYBAS_ID.apply(lambda x: str(x)[0])
    # printdf(world_basins[['HYBAS_ID', 'UP_AREA', 'area_percent']], 100)
    # printdf(world_basins.groupby('region')[['area_percent']].sum(), 100)
    print(world_basins)
    for id in world_basins.HYBAS_ID:
        save_path = save_dir/f'{id}.parquet'
        if not save_path.exists():
            print(f'\nWorking on {id}')
            make_all_intersecting_hydrobasin_level_2_polygon(
                hydrobasin_id=id, save_path=save_path, num_proc=30
            )
            print('')
        else:
            print(f'{id} already Complete')



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