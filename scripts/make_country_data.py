from wwvec.paths import ppaths
import shapely
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from water.basic_functions import tt, time_elapsed
import pyarrow as pa
pa.jemalloc_set_decay_ms(0)

if __name__ == "__main__":
    from water.basic_functions import get_country_polygon, printdf, tt, time_elapsed
    from pyproj import Geod
    from pathlib import Path
    country_boundaries = gpd.read_file(
        ppaths.data/'natural_earth_country_boundaries.zip', include_fields=['NAME', 'CONTINENT', 'ISO_A3', 'geometry']
    )
    country_boundaries = country_boundaries.sort_values(by='CONTINENT').reset_index(drop=True)
    save_dir = ppaths.data / 'country_waterways'
    st = tt()
    for ind, (name, iso_a3, polygon) in enumerate(
            zip(country_boundaries.NAME, country_boundaries.ISO_A3, country_boundaries.geometry)
    ):
        print(f'Working on {name} ({ind+1}/{len(country_boundaries)})')
        ctry_parquet_path = save_dir / f'{iso_a3}_model_waterways.parquet'
        if not ctry_parquet_path.exists():
            hydro_level2 = gpd.read_file(ppaths.hydrobasins, mask=polygon)
            waterway_gdfs = []
            s = tt()
            for id in hydro_level2.HYBAS_ID:
                id_path = ppaths.data/f'basins_level_2/{id}.parquet'
                if id_path.exists():
                    current_gdf = gpd.read_parquet(ppaths.data/f'basins_level_2/{id}.parquet')
                    tree = shapely.STRtree(current_gdf.geometry.to_numpy())
                    current_gdf = current_gdf.loc[tree.query(polygon, predicate='intersects')].reset_index(drop=True)
                    if len(current_gdf) > 0:
                        waterway_gdfs.append(current_gdf)
                else:
                    print(f'    Issue with {id_path}')
            time_elapsed(s, 2)
            if len(waterway_gdfs) > 0:
                waterway_gdfs = pd.concat(waterway_gdfs, ignore_index=True)
                waterway_gdfs.to_parquet(ctry_parquet_path)
    time_elapsed(st)
    # print(hydro_level2)
    # printdf(usa_states)
    # geod = Geod(ellps='WGS84')
    # save_dir = ppaths.data/'usa_states_waterways'

    # printdf(country_boundaries, 300)
    # save_dir = ppaths.data/'country_waterways'
    # save_dir.mkdir(exist_ok=True)
    # for ind, (name, polygon) in enumerate(zip(country_boundaries.ADM0_A3, country_boundaries.geometry)):
    #     ctry_parquet_path = save_dir/f'{name}_model_waterways.parquet'
    #     # polygon = get_country_polygon(ctry)
    #     if not ctry_parquet_path.exists():
    #         print(f'Working on {name} ({ind + 1}/{len(country_boundaries)})')
    #         try:
    #             gdf = make_all_intersecting_polygon(
    #                 polygon=polygon, save_path=ctry_parquet_path, overwrite=False, num_proc=20
    #             )
    #         except Exception as e:
    #             print(e)
    #             print(f'Issue making {name}')
    #         print('\n')
