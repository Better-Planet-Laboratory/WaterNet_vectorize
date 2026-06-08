import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
from markdown_it.rules_block import reference
from pyproj import Geod

from wwvec.paths import BasinPaths, PolygonizedPaths, Path, ppaths
from wwvec.polygon_vectorization._tools import printdf, tt, time_elapsed, SharedMemoryPool, delete_directory_contents
from collections import defaultdict
from multiprocessing import Process

# Some files are missing intersects_lake
# Issue with target_stream_id=-1 for streams without from_tdx. I believe due to intersecting stream at an endpoint.
# Issue with -1 in source_stream_ids, due to intersection of 3 streams in tdx data, and their system only allows
# 2 streams to intersect. They artificially add a point and intersect two streams at that point.

class WaterwayFixer:
    def __init__(
            self, waterways_gdf: gpd.GeoDataFrame, tdx_gdf: gpd.GeoDataFrame, lakes_tree: shapely.STRtree,
            hydrobasins_gdf: gpd.GeoDataFrame, hydrobasin_id: int
    ):
        self.waterways_gdf = waterways_gdf
        if not hasattr(self.waterways_gdf, 'intersects_lake'):
            self.waterways_gdf['intersects_lake'] = False
        self.tdx_gdf = tdx_gdf
        self.tdx_gdf = self.remove_length_0_tdx_hydro()
        self.make_tdx_stream_id_to_upstream_downstream_ids()
        self.lakes_tree = lakes_tree
        self.hydrobasins_gdf = hydrobasins_gdf
        self.hydrobasin_id = hydrobasin_id
        self.hydrobasin_polygon = hydrobasins_gdf[hydrobasins_gdf.HYBAS_ID == hydrobasin_id].reset_index().geometry[0]

    def check_fix_intersects_lake(self):
        return not np.any(self.waterways_gdf.intersects_lake)

    def fix_missing_intersects_lake(self):
        lakes_inds = self.lakes_tree.query(self.hydrobasin_polygon, predicate="intersects")
        lakes_polygons = self.lakes_tree.geometries[lakes_inds]
        waterways_tree = shapely.STRtree(self.waterways_gdf.geometry.to_numpy())
        _, waterways_inds = waterways_tree.query(lakes_polygons, predicate="intersects")
        intersects_lake_dict = defaultdict(lambda: False)
        intersects_lake_dict.update({ind: True for ind in waterways_inds})
        self.waterways_gdf['intersects_lake'] = self.waterways_gdf.index.map(intersects_lake_dict)

    def make_tdx_stream_id_to_upstream_downstream_ids(self):
        dicts = self.waterways_gdf[
            self.waterways_gdf.from_tdx
        ].groupby('tdx_stream_id')['stream_id'].agg(['min', 'max']).to_dict()
        self.tdx_to_upstream_id = dicts['min']
        self.tdx_to_downstream_id = dicts['max']

    def remove_length_0_tdx_hydro(self):
        tdx_gdf = self.tdx_gdf
        tdx_gdf['source_stream_ids'] = tdx_gdf[['USLINKNO1', 'USLINKNO2']].apply(
            lambda x: [linkno for linkno in (x.USLINKNO1, x.USLINKNO2) if linkno not in (-1, 0)],
            axis=1
        )
        tdx_gdf = tdx_gdf.sort_values(by='LINKNO').reset_index(drop=True)
        tdx_gdf['LINKNO_index'] = tdx_gdf['LINKNO'].copy()
        tdx_gdf.set_index('LINKNO_index', inplace=True)
        to_remove = tdx_gdf[(tdx_gdf.Length==0) & (tdx_gdf.DSLINKNO == -1)]
        for _, row in to_remove.iterrows():
            linkno = row['LINKNO']
            source_ids = row['source_stream_ids']
            target_id = row['DSLINKNO']
            #replace source ids target id with ds target id
            for source_id in source_ids:
                tdx_gdf.loc[source_id, 'DSLINKNO'] = target_id
            if target_id != -1:
                target_source_ids = tdx_gdf.loc[target_id, 'source_stream_ids']
                tdx_gdf.at[target_id, 'source_stream_ids'] = source_ids + [id for id in target_source_ids if id != linkno]
        tdx_gdf = tdx_gdf[~tdx_gdf.index.isin(to_remove.index)]
        to_remove = tdx_gdf[tdx_gdf.Length==0]
        for _, row in to_remove.iterrows():
            linkno = row['LINKNO']
            source_ids = tdx_gdf.loc[linkno, 'source_stream_ids']
            target_id = tdx_gdf.loc[linkno, 'DSLINKNO']
            #replace source ids target id with ds target id
            for source_id in source_ids:
                tdx_gdf.loc[source_id, 'DSLINKNO'] = target_id
            if target_id != -1:
                target_source_ids = tdx_gdf.loc[target_id, 'source_stream_ids']
                tdx_gdf.at[target_id, 'source_stream_ids'] = source_ids + [id for id in target_source_ids if id != linkno]
        return tdx_gdf[tdx_gdf.Length != 0]

    def check_fix_source_stream_issues(self):
        needs_fixed = self.waterways_gdf['source_stream_ids'].apply(lambda x: -1 in x or len(x)==1)
        return self.waterways_gdf[needs_fixed]

    def fix_source_streams_issue(self, streams_to_fix: gpd.GeoDataFrame):
        streams_to_fix = streams_to_fix.sort_values(by='stream_order')
        for _, row in streams_to_fix.iterrows():
            stream_id = row['stream_id']
            current_source_ids = [int(id) for id in row['source_stream_ids'] if id != -1]
            tdx_id = row['tdx_stream_id']
            tdx_source_ids = self.tdx_gdf.loc[tdx_id, 'source_stream_ids']
            ww_source_ids = [
                self.tdx_to_downstream_id[id] for id in tdx_source_ids
                if self.tdx_to_downstream_id[id] not in current_source_ids and id != 0
            ]
            self.waterways_gdf.at[stream_id, 'source_stream_ids'] = ww_source_ids + list(current_source_ids)
            self._fix_downstream_stream_orders(stream_id)

    def check_fix_target_streams_issues_from_tdx(self):
        tdx_not_neg_1 = set(self.tdx_gdf[self.tdx_gdf.DSLINKNO!=-1].LINKNO)
        to_return = (
                (self.waterways_gdf['target_stream_id'] == -1)
                & (self.waterways_gdf.from_tdx)
                & (self.waterways_gdf.tdx_stream_id.isin(tdx_not_neg_1))
        )
        return to_return

    def check_fix_target_streams_issues_not_tdx(self):
        return (self.waterways_gdf['target_stream_id'] == -1) & (~self.waterways_gdf.from_tdx)

    def check_fix_target_streams_issues(self):
        return self.waterways_gdf[
            self.check_fix_target_streams_issues_from_tdx() + self.check_fix_target_streams_issues_not_tdx()
        ]

    def add_stream_id_to_target_source_ids(self, stream_id: int, target_stream_id: int):
        target_source_ids = self.waterways_gdf.loc[target_stream_id, 'source_stream_ids']
        if stream_id not in target_source_ids:
            # print(stream_id, target_stream_id, self.waterways_gdf.loc[stream_id, 'from_tdx'])
            self.waterways_gdf.at[target_stream_id, 'source_stream_ids'] = [stream_id] + list(target_source_ids)

    def _calculate_stream_order(self, stream_id):
        source_ids = self.waterways_gdf.loc[stream_id, 'source_stream_ids']
        if -1 in source_ids:
            source_ids = [sid for sid in source_ids if sid != -1]
        source_stream_orders = self.waterways_gdf.loc[source_ids, 'stream_order'].to_list()
        source_stream_orders.sort(reverse=True)
        if len(source_stream_orders) == 0:
            return 1
        stream_order = source_stream_orders[0]
        if len(source_stream_orders) > 1:
            if source_stream_orders[1] == stream_order:
                stream_order += 1
        return stream_order

    def _fix_downstream_stream_orders(self, stream_id):
        current_stream_order = self.waterways_gdf.loc[stream_id, 'stream_order']
        calculated_stream_order = self._calculate_stream_order(stream_id)
        while calculated_stream_order != current_stream_order:
            self.waterways_gdf.loc[stream_id, 'stream_order'] = calculated_stream_order
            stream_id = self.waterways_gdf.loc[stream_id, 'target_stream_id']
            if stream_id == -1:
                break
            current_stream_order = self.waterways_gdf.loc[stream_id, 'stream_order']
            calculated_stream_order = self._calculate_stream_order(stream_id)

    def fix_target_streams_issues(self, streams_to_fix: gpd.GeoDataFrame):
        streams_to_fix = streams_to_fix.sort_values(by='stream_order')
        for _, row in streams_to_fix.iterrows():
            stream_id = row['stream_id']
            tdx_stream_id = row['tdx_stream_id']
            tdx_target_stream_id = self.tdx_gdf.loc[tdx_stream_id, 'DSLINKNO']
            target_stream_id = self.tdx_to_upstream_id[tdx_target_stream_id] if tdx_target_stream_id != -1 else -1
            self.waterways_gdf.loc[stream_id, 'target_stream_id'] = target_stream_id
            if tdx_target_stream_id != -1:
                self.add_stream_id_to_target_source_ids(stream_id, target_stream_id)
                self._fix_downstream_stream_orders(target_stream_id)

    def run(self):
        if self.check_fix_intersects_lake():
            s = tt()
            print('Adding intersects lake')
            self.fix_missing_intersects_lake()
            time_elapsed(s, 2)

        print(f'Finding waterways with issues')
        s = tt()
        waterways_with_source_id_issues = self.check_fix_source_stream_issues()
        waterways_with_target_id_issues = self.check_fix_target_streams_issues()
        time_elapsed(s, 2)
        if len(waterways_with_source_id_issues)>0:
            s = tt()
            print(f'Fixing {len(waterways_with_source_id_issues)} waterways with source stream issues')
            self.fix_source_streams_issue(waterways_with_source_id_issues)
            time_elapsed(s, 2)

        if len(waterways_with_target_id_issues) > 0:
            s = tt()
            print(f'Fixing {len(waterways_with_target_id_issues)} waterways with target stream issues')
            self.fix_target_streams_issues(waterways_with_target_id_issues)
            time_elapsed(s, 2)
        return self.waterways_gdf


def run_for_id(hybas_id: int, waterway_dir: Path, save_dir: Path):
    s = tt()
    lines_path = waterway_dir/f'{hybas_id}.parquet'
    tdx_path = ppaths.tdx_streams/f'basin_{hybas_id}.parquet'
    lines_gdf = gpd.read_parquet(lines_path)
    tdxgdf = gpd.read_parquet(tdx_path)
    hydro2 = gpd.read_file(ppaths.hydrobasins)
    lakes_gdf = gpd.read_parquet(ppaths.data/'hydrolakes.parquet')
    laketree = shapely.STRtree(lakes_gdf.geometry.to_numpy())
    wwfixer = WaterwayFixer(lines_gdf, tdxgdf, laketree, hydro2, hybas_id)
    fixed_gdf = wwfixer.run()
    fixed_gdf.to_parquet(save_dir/f'{hybas_id}.parquet')

    # fixed_gdf.to_parquet(ppaths.data/f'basins_level_2_with_length_fixed/{hybas_id}.parquet')
    time_elapsed(s)


def run_for_all(waterway_dir: Path, save_dir: Path=None, force: bool = False):
    if save_dir is None:
        save_dir = ppaths.data/f'basins_level_2_with_length_fixed'
    if waterway_dir is None:
        waterway_dir = ppaths.data/'basins_level_2_with_length'
    hydro_level_2 = gpd.read_file(ppaths.hydrobasins)
    for hybas_id in hydro_level_2.HYBAS_ID:
        save_name = f'{hybas_id}.parquet'
        save_path = save_dir/save_name
        print(f'Working on {hybas_id}')
        if (not save_path.exists() or force) and (waterway_dir/save_name).exists():
            p = Process(target=run_for_id, args=(hybas_id, waterway_dir, save_dir,))
            p.start()
            p.join()
            p.close()
        print('\n')


if __name__ == '__main__':
    run_for_all(waterway_dir=ppaths.data/'basins_level_2', force=True)
    # hybas_id = 5020049720
    # hybas_id = 4020015090
    # hybas_id = 1020000010
    # hybas_id = 2020071190
    # # 4020050290, 3020024310, 3020005240
    # gdf1 = gpd.read_parquet(ppaths.data/'basins_level_2_with_width/4020034510.parquet')
    # lines_path = ppaths.data/f'basins_level_2_with_length/{hybas_id}.parquet'
    # tdx_path = ppaths.tdx_streams/f'basin_{hybas_id}.parquet'
    # s = tt()
    # lines_gdf = gpd.read_parquet(lines_path)
    # time_elapsed(s)
    # s=tt()
    # tdxgdf = gpd.read_parquet(tdx_path)
    # hydro2 = gpd.read_file(ppaths.hydrobasins)
    # time_elapsed(s)
    # s=tt()
    # lakes_gdf = gpd.read_parquet(ppaths.data/'hydrolakes.parquet')
    # time_elapsed(s)
    # s=tt()
    # lines_tree = shapely.STRtree(lines_gdf.geometry.to_numpy())
    # time_elapsed(s)
    # s=tt()
    # lake_inds, _ = lines_tree.query(lakes_gdf.geometry.to_numpy(), predicate='intersects')
    # del lines_tree
    # lakes_gdf = lakes_gdf[lakes_gdf.index.isin(lake_inds)].reset_index(drop=True)
    # time_elapsed(s)
    # s=tt()
    # laketree = shapely.STRtree(lakes_gdf.geometry.to_numpy())
    # wwfixer = WaterwayFixer(lines_gdf, tdxgdf, laketree, hydro2, hybas_id)
    # fixed_gdf = wwfixer.run()