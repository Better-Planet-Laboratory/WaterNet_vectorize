from wwvec.width_calculation.drop_and_reorder_streams import run_stream_remover, tt, time_elapsed
from wwvec.paths import ppaths, Path
import geopandas as gpd
import pandas as pd
import numpy as np
from multiprocessing import Process
from pyproj import Geod

pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

def calculate_length(gdf_path: Path, save_path: Path, hyrdobasin_id: int, remove_lakes: bool = True):
    gdf = gpd.read_parquet(gdf_path)
    if remove_lakes:
        gdf = gdf[~gdf.intersects_lake]
    gdf1 = gdf.groupby(['from_tdx', 'stream_order'])[['length_m']].sum().reset_index()
    gdf1['hydrobasin_id'] = hyrdobasin_id
    gdf1.to_csv(save_path, index=False)
    return gdf1


def calculate_individual_basin_lengths(
        waterways_dir=ppaths.data/'basins_level_2_removed_new',
        save_dir=ppaths.data/'basins_level_2_calculated_lengths',
        remove_lakes=True
):
    hydro = gpd.read_file(ppaths.hydrobasins)
    # hydro = hydro[hydro.HYBAS_ID.isin([4020034510])]
    # hydro.to_file(ppaths.data/f'hydrobasins_level_2.fgb')
    save_dir.mkdir(exist_ok=True)
    for idx, hydro_id in enumerate(hydro.HYBAS_ID):
        gdf_path = waterways_dir/f'{hydro_id}.parquet'
        save_path = save_dir/f'{hydro_id}.csv'
        print(f'Working on {hydro_id} ({idx+1}/{len(hydro)})')
        # if not save_path.exists():
        s = tt()
        p = Process(target=calculate_length, args=(gdf_path, save_path, hydro_id, remove_lakes))
        p.start()
        p.join()
        p.close()
        time_elapsed(s, 2)
        print('\n')

def merge_individual_basin_lengths(lengths_dir = ppaths.data/'basins_level_2_calculated_lengths'):
    dfs = []
    for file in lengths_dir.glob('*.csv'):
        dfs.append(pd.read_csv(file))
    df = pd.concat(dfs, ignore_index=True)
    return df


if __name__ == '__main__':
    geod = Geod(ellps='WGS84')
    length_func = np.frompyfunc(lambda geometry: geod.geometry_length(geometry), 1, 1)
    waterways_dir = ppaths.data/'basins_level_2_removed_new'
    lengths_dir = ppaths.data/'basins_level_2_calculated_lengths_new'
    basin_lengths_csv_save_path = ppaths.data/'length_by_hydrobasin_id_new.csv'
    stream_order_lengths_csv_save_path = ppaths.data/'length_by_stream_order_new.csv'
    calculate_individual_basin_lengths(waterways_dir=waterways_dir, save_dir=lengths_dir, remove_lakes=True)
    df = merge_individual_basin_lengths(lengths_dir)
    df['length_km'] = np.round(df['length_m']/1000, 3)
    basin_df = df.groupby(['hydrobasin_id', 'from_tdx'])['length_km'].sum().reset_index()
    stream_order_df = df.groupby(['from_tdx', 'stream_order'])['length_km'].sum().reset_index()
    basin_df.to_csv(basin_lengths_csv_save_path, index=False)
    stream_order_df.to_csv(stream_order_lengths_csv_save_path, index=False)
    # waterways_dir_old = ppaths.data/'basins_level_2_with_length_fixed'

    # lengths_dir = ppaths.data/'basins_level_2_calculated_lengths_with_lakes_old'
    # df = merge_individual_basin_lengths(lengths_dir)
    # df['length_km'] = np.round(df['length_m']/1000, 3)
    # basin_df_old = df.groupby(['hydrobasin_id', 'from_tdx'])['length_km'].sum().reset_index()
    # # basin_df.to_csv(basin_lengths_csv_save_path, index=False)
    # # stream_order_df.to_csv(stream_order_lengths_csv_save_path, index=False)
    # basin_df = basin_df[basin_df.from_tdx]
    # basin_df_old = basin_df_old[basin_df_old.from_tdx].set_index('hydrobasin_id')
    # basin_df = basin_df.join(basin_df_old['length_km'], on='hydrobasin_id', rsuffix='_old')
    # gdf1 = gpd.read_parquet(waterways_dir/'5020082270.parquet')
    # gdf2 = gpd.read_parquet(waterways_dir_old/'5020082270.parquet')
    #
    # gdf1 = gdf1[gdf1.from_tdx].reset_index(drop=True)
    # gdf1['length_check'] = gdf1.geometry.apply(length_func)
    # gdf2 = gdf2[gdf2.from_tdx].reset_index(drop=True)
    # gdf2['length_check'] = gdf2.geometry.apply(length_func)
    #
    # length1 = gdf1.length_check.sum()/1000
    # length2 = gdf2.length_check.sum()/1000
    # print(length1, length2)