import shapely
from wwvec.paths import ppaths
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import warnings
import os
import time
import pyarrow as pa
pa.jemalloc_set_decay_ms(0)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_lakes_intersecting_basin(
        lakes_gdf: gpd.GeoDataFrame, init_lakes_tree: shapely.STRtree, basins_gdf: gpd.GeoDataFrame, basin_id: int
) -> shapely.MultiPolygon:
    basin_polygon = basins_gdf[basins_gdf.HYBAS_ID == basin_id].reset_index().geometry[0]
    return shapely.unary_union(lakes_gdf.loc[init_lakes_tree.query(basin_polygon, predicate='intersects'), 'geometry'])


if __name__ == "__main__":
    from water.basic_functions import get_country_polygon, printdf, tt, time_elapsed
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    waterbodies = gpd.read_parquet(ppaths.data/'hydrolakes.parquet')
    init_waterbodies_tree = shapely.STRtree(waterbodies.geometry.to_numpy())
    level_2_basins = gpd.read_parquet(ppaths.data/'basins/hybas_all_level_2.parquet')
    basin_dir = ppaths.data/'basins_level_2'
    basin_dir_with_len = ppaths.data/'basins_level_2_with_length'
    basin_dir_with_len.mkdir(exist_ok=True)
    length_dir = ppaths.data/'basin_waterway_lengths'
    length_dir.mkdir(exist_ok=True)
    length_gdfs = []
    basin_paths = list(basin_dir.iterdir())
    geod_geom_len = np.frompyfunc(lambda x: geod.geometry_length(x), 1, 1)
    st = tt()
    for ind, basin_path in enumerate(basin_paths):
        if not (length_dir/basin_path.name).exists():
            print(f'Working on {basin_path.name} ({ind+1}/{len(basin_paths)})')
            s = tt()
            print('  Opening data')
            gdf = gpd.read_parquet(basin_path)
            lakes_in_basin = get_lakes_intersecting_basin(
                basin_id=int(basin_path.stem), basins_gdf=level_2_basins,
                lakes_gdf=waterbodies, init_lakes_tree=init_waterbodies_tree
            )
            ww_tree = shapely.STRtree(gdf.geometry.to_numpy())
            wws_intersecting_lake = ww_tree.query(lakes_in_basin, predicate='intersects')
            gdf['intersects_lake'] = gdf.index.isin(wws_intersecting_lake)
            time_elapsed(s, 4)
            print('  Finding Length')
            gdf['length_m'] = geod_geom_len(gdf.geometry).astype(np.float32)
            gdf.to_parquet(basin_dir_with_len/basin_path.name)
            gdf['length_km'] = gdf['length_m']/1000

            printdf(gdf.groupby(['intersects_lake', 'from_tdx'])[['length_km']].sum())
            print('')
            gdf1 = gdf.groupby(['intersects_lake', 'from_tdx', 'stream_order'])[['length_km']].sum()
            gdf1['length_km'] = np.round(gdf1['length_km'])
            printdf(gdf1, 100)
            gdf1 = gdf1.reset_index()
            gdf1['basin_id'] = int(basin_path.stem)
            gdf1.to_parquet(length_dir/basin_path.name)
            length_gdfs.append(gdf1)
            time_elapsed(s, 4)
            time_elapsed(st)
        else:
            length_gdfs.append(pd.read_parquet(length_dir/basin_path.name))
        # print('\n'*2)
        # break
    length_gdf = pd.concat(length_gdfs)
    length_gdf['source'] = length_gdf.from_tdx.apply(lambda x: 'TDX-Hydro' if x else 'WaterNet')
    length_gdf['length_km'] = length_gdf['length_km'].astype(np.uint32)
    # length_gdf_no_lakes = length_gdf[(~length_gdf.intersects_lake) & (length_gdf.length_km > 0)]
    length_gdf_no_lakes = length_gdf[(length_gdf.length_km > 0)]

    total_no_lakes = length_gdf_no_lakes.groupby(['source'])[['length_km']].sum()
    # total_no_lakes_by_stream_order = length_gdf_no_lakes.groupby(['source', 'stream_order'])[['length_km']].sum()
    # total_by_basin_id = length_gdf_no_lakes.groupby(['basin_id', 'source'])[['length_km']].sum()
    # # total_by_bain_id_and_stream_order =
    printdf(total_no_lakes, 100)
    # printdf(total_no_lakes_by_stream_order, 100)
    # printdf(total_by_basin_id, 200)