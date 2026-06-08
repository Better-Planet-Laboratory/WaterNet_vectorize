import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
from pyproj import Geod

from wwvec.paths import BasinPaths, PolygonizedPaths, Path, ppaths
from wwvec.polygon_vectorization._tools import printdf, tt, time_elapsed, SharedMemoryPool, delete_directory_contents
from collections import defaultdict


# import warnings
# warnings.filterwarnings("ignore")

# Filter out streams with a length/width<2 & stream_order == 1,
# stream_order == 1 & target stream order >=6, (need to add target stream order)
# stream_order == 1 and length_m < 100
# recalculate stream order and merge downstream, and repeat until no changes are made.
# Two scenarios during downstream merging:
#   1.) Target stream order doesn't change: merge with upstream geometry, recalculate values.
#   2.) Target stream order changes: merge with upstream geometry, recalculate values,
#           and recalculate all downstream target stream orders


def add_target_stream_order(streams_df: gpd.GeoDataFrame):
    stream_id_to_order = defaultdict(int)
    stream_id_to_order.update({
        stream_id: stream_order for stream_id, stream_order in zip(streams_df.stream_id, streams_df.stream_order)
    })
    streams_df['target_stream_order'] = streams_df.target_stream_id.map(stream_id_to_order)
    return streams_df


class StreamRemover:
    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf
        self.gdf['stream_id_index'] = self.gdf.stream_id.astype(int)
        self.gdf.set_index(self.gdf.stream_id_index, inplace=True)
        self.ids_to_remove = []

    def _get_source_rows(self, target_stream_id, stream_id_remove):
        source_ids = list(self.gdf.loc[target_stream_id, 'source_stream_ids'])
        source_id = [int(id) for id in source_ids if int(id) != stream_id_remove]
        return source_id

    def _fix_target_stream_and_stream_orders_single_source(self, stream_id_fix, source_id):
        # merge geometry with source if single source
        # add lengths
        # average widths
        # check if need to add downstream to fix list
        fix_row = self.gdf.loc[stream_id_fix]
        source_row = self.gdf.loc[source_id]
        target_stream_id = fix_row['target_stream_id']
        self.gdf.loc[source_id, 'geometry'] = shapely.line_merge(
            shapely.MultiLineString([source_row.geometry, fix_row.geometry])
        )
        self.gdf.loc[source_id, 'target_stream_id'] = target_stream_id
        new_length = fix_row['length_m'] + source_row['length_m']
        new_width = fix_row['width_m']*fix_row['length_m'] + source_row['width_m']*source_row['length_m']
        new_width = new_width/new_length
        new_length_width_ratio = fix_row['length_width_ratio']*fix_row['length_m'] + source_row['length_width_ratio']*source_row['length_m']
        new_length_width_ratio = new_length_width_ratio/new_length
        self.gdf.loc[source_id, 'length_m'] = new_length
        self.gdf.loc[source_id, 'width_m'] = new_width
        # self.gdf.loc[source_id, 'length_width_ratio'] = new_length/(new_width + 0.00000000000001)
        self.gdf.loc[source_id, 'length_width_ratio'] = new_length_width_ratio
        self.gdf.loc[source_id, 'intersects_lake'] = fix_row['intersects_lake'] or source_row['intersects_lake']
        self._fix_target_source_stream_ids(fix_row['target_stream_id'], fix_row['stream_id'], source_row['stream_id'])
        self.ids_to_remove.append(fix_row['stream_id'])
        if fix_row['stream_order'] != source_row['stream_order'] and target_stream_id != -1:
            self._fix_downstream_stream_orders(fix_row['target_stream_id'])
        # if new_length/new_width < 2 and self.gdf.loc[source_id, 'stream_order'] == 1:
        #     print(source_id)

    def _fix_target_stream_and_stream_orders(self, stream_id_remove):
        stream_id_fix = self.gdf.loc[stream_id_remove, 'target_stream_id']
        if stream_id_fix != -1:
            source_ids = self._get_source_rows(stream_id_fix, stream_id_remove)
            if len(source_ids) == 1:
                if self.gdf.loc[source_ids[0], 'from_tdx'] == self.gdf.loc[stream_id_fix, 'from_tdx']:
                    self._fix_target_stream_and_stream_orders_single_source(stream_id_fix, source_ids[0])
                else:
                    self._fix_target_source_stream_ids(stream_id_fix, stream_id_remove)
                    self._fix_downstream_stream_orders(stream_id_fix)
            else:
                self._fix_target_source_stream_ids(stream_id_fix, stream_id_remove)
                self._fix_downstream_stream_orders(stream_id_fix)

    def _fix_target_source_stream_ids(self, target_stream_id, stream_id_remove, stream_id_add=None):
        if target_stream_id != -1:
            source_rows = self._get_source_rows(target_stream_id, stream_id_remove)
            if stream_id_add:
                source_rows.append(stream_id_add)
            self.gdf.at[target_stream_id, 'source_stream_ids'] = np.array(source_rows, dtype='int32')

    def _calculate_stream_order(self, stream_id):
        source_ids = self.gdf.loc[stream_id, 'source_stream_ids']
        if -1 in source_ids:
            source_ids = [sid for sid in source_ids if sid != -1]
            print(f'Issue with {stream_id} source ids, tdx steam id: {self.gdf.loc[stream_id, "tdx_stream_id"]}')
        source_stream_orders = self.gdf.loc[source_ids, 'stream_order'].to_list()
        source_stream_orders.sort(reverse=True)
        if len(source_stream_orders) == 0:
            return 1
        stream_order = source_stream_orders[0]
        if len(source_stream_orders) > 1:
            if source_stream_orders[1] == stream_order:
                stream_order += 1
        return stream_order

    def _fix_downstream_stream_orders(self, stream_id):
        current_stream_order = self.gdf.loc[stream_id, 'stream_order']
        calculated_stream_order = self._calculate_stream_order(stream_id)
        while calculated_stream_order != current_stream_order:
            self.gdf.loc[stream_id, 'stream_order'] = calculated_stream_order
            stream_id = self.gdf.loc[stream_id, 'target_stream_id']
            if stream_id == -1:
                break
            current_stream_order = self.gdf.loc[stream_id, 'stream_order']
            calculated_stream_order = self._calculate_stream_order(stream_id)

    def _get_streams_to_remove(self):
        stream_ids = self.gdf[
            ~self.gdf.from_tdx & (self.gdf.stream_order == 1) & (self.gdf.length_width_ratio<2)
        ].sort_values(by='length_m').stream_id
        # print(len(stream_ids))
        self.ids_to_remove = []
        return stream_ids

    def remove_streams(self):
        start = tt()
        streams_to_remove = self._get_streams_to_remove()
        while len(streams_to_remove) > 0:
            num_to_remove = len(self.ids_to_remove)
            # print(f'Removing {len(streams_to_remove)} streams')
            s = tt()
            for idx, stream_id in enumerate(streams_to_remove):
                if self.gdf.loc[stream_id, 'length_width_ratio'] < 2:
                    self.ids_to_remove.append(stream_id)
                    self._fix_target_stream_and_stream_orders(stream_id)
                # if idx % max(len(streams_to_remove)//10, 1) == 0:
                #     print(f'  {idx/len(streams_to_remove):.2%} ({idx}/{len(streams_to_remove)})')
                #     time_elapsed(s, 2)
            # time_elapsed(s, 2)
            self.gdf.drop(self.ids_to_remove, axis=0, inplace=True)
            streams_to_remove = self._get_streams_to_remove()
        # time_elapsed(start)
        return self.gdf


def run_stream_remover(stream_path, save_path):
    gdf = gpd.read_parquet(stream_path)
    gdf['length_m'] = gdf['length_m'].astype(np.float32)
    gdf['width_m'] = gdf['width_m'].astype(np.float32)
    gdf['length_width_ratio'] = gdf['length_width_ratio'].astype(np.float32)
    gdf['stream_order'] = gdf['stream_order'].astype(np.int32)
    gdf = gdf.sort_values(by='stream_id').reset_index(drop=True)
    stream_remover = StreamRemover(gdf)
    gdf = stream_remover.remove_streams()
    gdf.to_parquet(save_path)


if __name__ == '__main__':
    gdf = gpd.read_parquet(ppaths.data/'basins_level_2_with_width/5020082270.parquet')
    # gdf = gdf.sort_values(by=['stream_order', 'length_m']).reset_index(drop=True)
    gdf = gdf.sort_values(by='stream_id').reset_index(drop=True)
    # gdf = gdf.sort_values(by='stream_id').reset_index(drop=True)
    # printdf(gdf[(gdf.target_stream_id==-1) & ~gdf.from_tdx])
    stream_remover = StreamRemover(gdf)
    gdf1 = stream_remover.remove_streams()
    print(len(gdf1))