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
from functools import cached_property
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_for_basin_list(input_list):
    for inputs in input_list:
        # run_for_basin(**inputs)
        try:
            run_for_basin(**inputs)
        except:
            print(inputs['basin_geometry'].centroid)
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
    my_pool(
        num_proc=num_proc, func=run_for_basin_list, input_list=input_chunks,
        use_kwargs=False, sleep_time=0, terminate_on_error=False
    )
    print('Merging dataframes')
    merged_df = pd.concat([gpd.read_parquet(file) for file in temp_basin_path.iterdir()], ignore_index=True)
    stream_order_fixer = StreamOrderFixer(merged_df)
    stream_order_fixer.investigate_all()
    stream_order_fixer.add_fixed_stream_order()
    stream_order_fixer.init_df.to_parquet(save_path)
    return merged_df


def new_stream_order(old_stream_order, source_1_order, source_2_order):
    new_source_order = source_1_order + 1 if source_1_order == source_2_order else max(source_1_order, source_2_order)
    return max(new_source_order, old_stream_order)


class StreamOrderFixer:
    def __init__(self, df):
        self.init_df = df

    def apply_new_stream_order(self, row):
        stream_order = row.stream_order
        tdx_id = row.tdx_stream_id
        if row.from_tdx:
            stream_order = max(stream_order, self.new_stream_orders[tdx_id])
        return stream_order

    def investigate_all(self):
        for index, id in enumerate(self.old_stream_orders):
            if id not in self.new_stream_orders:
                self.investigate_id(id)

    def add_fixed_stream_order(self):
        self.init_df['fixed_stream_order'] = self.init_df[['from_tdx', 'tdx_stream_id', 'stream_order']].apply(
            lambda row: self.apply_new_stream_order(row), axis=1
        )

    @cached_property
    def reference_df(self) -> gpd.GeoDataFrame:
        reference_df = self.init_df.groupby('tdx_stream_id')[['stream_order']].agg('max')
        df_tdx_info = (self.init_df[['tdx_stream_id', 'tdx_source_ids', 'tdx_target_id']].
                       drop_duplicates('tdx_stream_id').set_index('tdx_stream_id'))
        reference_df = reference_df.join(df_tdx_info, how='outer').reset_index()
        return reference_df

    @cached_property
    def new_stream_orders(self) -> dict:
        new_stream_orders = {
            stream_id: stream_order for (stream_id, stream_order, source_id) in
            zip(self.reference_df.tdx_stream_id, self.reference_df.stream_order, self.reference_df.tdx_source_ids)
            if -1 in source_id
        }
        return new_stream_orders

    @cached_property
    def old_stream_orders(self) -> dict:
        old_stream_orders = {
            stream_id: stream_order for (stream_id, stream_order) in
            zip(self.reference_df.tdx_stream_id, self.reference_df.stream_order)
        }
        return old_stream_orders

    @cached_property
    def id_to_target(self) -> dict:
        id_to_target = {
            stream_id: target_id for stream_id, target_id in
            zip(self.reference_df.tdx_stream_id, self.reference_df.tdx_target_id)
        }
        return id_to_target

    @cached_property
    def id_to_sources(self) -> dict:
        id_to_sources = {
            stream_id: source_ids for stream_id, source_ids in
            zip(self.reference_df.tdx_stream_id, self.reference_df.tdx_source_ids)
        }
        return id_to_sources

    @cached_property
    def ids_to_check(self) -> set:
        ids_to_check = {id for id in self.new_stream_orders if self.check_sources_investigated(id)}
        return ids_to_check

    def check_sources_investigated(self, id):
        sources = self.id_to_sources[id]
        for source in sources:
            if source not in self.new_stream_orders and source in self.old_stream_orders:
                return False
        return True

    @staticmethod
    def _calculate_new_stream_order(old_stream_order, source_1_order, source_2_order):
        source_order = source_1_order + 1 if source_1_order == source_2_order else max(source_1_order, source_2_order)
        return max(source_order, old_stream_order)

    def get_new_stream_order(self, id):
        source_1, source_2 = self.id_to_sources[id]
        old_stream_order = self.old_stream_orders[id]
        source_1_order = self.new_stream_orders.get(source_1, old_stream_order)
        source_2_order = self.new_stream_orders.get(source_2, old_stream_order)
        if source_1 not in self.old_stream_orders or source_2 not in self.old_stream_orders:
            self.new_stream_orders[id] = max(old_stream_order, source_1_order, source_2_order)
            return self.new_stream_orders[id]
        else:
            self.new_stream_orders[id] = self._calculate_new_stream_order(
                old_stream_order, source_1_order, source_2_order
            )
            return self.new_stream_orders[id]

    def investigate_id(self, id):
        if self.check_sources_investigated(id):
            self.get_new_stream_order(id)
            target_id = self.id_to_target[id]
            if target_id in self.old_stream_orders:
                self.investigate_id(target_id)



if __name__ == "__main__":
    ctry = 'rwanda'
    # countries = ['rwanda', 'ivory_coast', 'ethiopia', 'zambia', 'tanzania', 'kenya']
    # countries = ['equatorial_guinea']
    # countries = ['rwanda']
    # from water.basic_functions import get_country_polygon
    # for ctry in countries:
    #     ctry_parquet_path = ppaths.country_data/f'africa_basins_countries/{ctry}_waterways_new.parquet'
    #     polygon = get_country_polygon(ctry)
    #     gdf = make_all_intersecting_polygon(polygon=polygon, save_path=ctry_parquet_path, overwrite=True)
        # ctry_gpkg_path = ppaths.country_data/f'africa_basins_countries/{ctry}_waterways.gpkg'
        # gdf = gpd.read_parquet(ctry_parquet_path)

        # gdf.to_file(ctry_gpkg_path, driver='GPKG')
    # countries = ['uganda']
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    # print('opening')
    # gdf = gpd.read_parquet(ppaths.country_data/f'africa_basins_countries/{ctry}_waterways_new.parquet')
    gdf = gpd.read_parquet(ppaths.country_data/f'basins_level_2/1020000010_waterways.parquet')
    # s = tt()
    # print('making stream_order_fixer')
    # stream_order_fixer = StreamOrderFixer(gdf)
    # time_elapsed(s)
    # s = tt()
    # print('investigating')
    # stream_order_fixer.investigate_all()
    # time_elapsed(s)
    # s = tt()
    # print('adding fixed')
    # stream_order_fixer.add_fixed_stream_order()
    # time_elapsed(s)
    # df = stream_order_fixer.init_df
    geom_length = np.frompyfunc(lambda x: geod.geometry_length(x), nin=1, nout=1)
    gdf['length'] = geom_length(gdf.geometry.to_numpy())
    print(gdf.stream_order.max(), gdf.fixed_stream_order.max())

    print(gdf.groupby(['from_tdx', 'stream_order'])['length'].sum()/1000)
    # africa_basins = gpd.read_file(ppaths.country_data/'basins/hybas_af_lev01-12_v1c/hybas_af_lev02_v1c.shp')
    # from multiprocessing import Process
    # world_basins = gpd.read_parquet(ppaths.country_data/'basins/hybas_all_level_2.parquet')
    # st = tt()
    # for ind, (id, poly) in enumerate(zip(world_basins.HYBAS_ID, world_basins.geometry)):
    #     s = tt()
    #     print(f'Working on {id} ({ind+1}/{len(world_basins)})')
    #     poly = poly.buffer(-.00001)
    #     save_path = ppaths.country_data/f'basins_level_2/{id}_waterways.parquet'
    #     if not save_path.exists():
    #         p = Process(
    #             target=make_all_intersecting_polygon,
    #             kwargs=dict(polygon=poly, save_path=save_path, num_proc=28, overwrite=False)
    #         )
    #         p.start()
    #         p.join()
    #         p.close()
    #         break
    #         # make_all_intersecting_polygon(polygon=poly, save_path=save_path, num_proc=20, overwrite=False)
    #     time_elapsed(s, 2)
    # time_elapsed(st, 2)


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