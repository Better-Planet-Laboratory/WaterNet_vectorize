import geopandas as gpd
from functools import cached_property
from collections import defaultdict
import numpy as np

# noinspection PyShadowingBuiltins
class StreamOrderFixer:
    """
    :class: StreamOrderFixer

    This class is responsible for fixing the stream order in a given DataFrame.

    Attributes:
        - init_df : pd.DataFrame
            The initial DataFrame with stream order information.

    Methods:
        - __init__(self, df)
            Initializes a StreamOrderFixer object and performs the necessary operations to fix the stream order.

        - apply_new_stream_order(self, row)
            Applies a new stream order to a given row based on the row's attributes.

        - investigate_all(self)
            Investigates all the stream IDs in the old_stream_orders dictionary.

        - add_fixed_stream_order(self)
            Adds a 'fixed_stream_order' column to the init_df DataFrame using the apply_new_stream_order function.

        - reference_df(self) -> gpd.GeoDataFrame
            Returns a cached property that represents a reference DataFrame containing relevant stream order information.

        - new_stream_orders(self) -> dict
            Returns a cached property that represents a dictionary containing new stream orders for specific stream IDs.

        - old_stream_orders(self) -> dict
            Returns a cached property that represents a dictionary containing old stream orders for specific stream IDs.

        - id_to_target(self) -> dict
            Returns a cached property that represents a dictionary mapping stream IDs to their target IDs.

        - id_to_sources(self) -> dict
            Returns a cached property that represents a dictionary mapping stream IDs to their source IDs.

        - ids_to_check(self) -> set
            Returns a cached property that represents a set of stream IDs to be checked for further investigation.

        - check_sources_investigated(self, id)
            Checks whether the sources of a given stream ID have already been investigated.

        - _calculate_new_stream_order(self, old_stream_order, source_1_order, source_2_order)
            Calculates a new stream order based on the old stream order and the orders of two sources.

        - get_new_stream_order(self, id)
            Returns the new stream order for a given stream ID based on its sources.

        - investigate_id(self, id)
            Investigates a given stream ID, assigns a new stream order to it, and recursively investigates its target ID if necessary.
    """
    def __init__(self, df):
        self.init_df = df
        self.investigate_all()
        self.add_fixed_stream_order()

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
            if -1 in source_id or len(source_id) == 0
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


class PathingFixer:
    """
    Class: PathingFixer

    This class is used to fix pathing issues in a given GeoDataFrame.

    Attributes:
    - df: gpd.GeoDataFrame
        The input GeoDataFrame containing information about streams and their connections.
    - basin_stream_id_to_stream_id: defaultdict
        Dictionary mapping basin_stream_id to stream_id.
    - stream_id_and_tdx_id_to_basin_stream_id: defaultdict
        Dictionary mapping (stream_id, tdx_id) to basin_stream_id.
    - tdx_id_to_source_id: defaultdict
        Dictionary mapping tdx_id to source_id.
    - tdx_id_to_target_id: defaultdict
        Dictionary mapping tdx_id to target_id.
    - basin_stream_id_to_sources: defaultdict
        Dictionary mapping basin_stream_id to an array of source_ids.
    - basin_stream_id_to_targets: defaultdict
        Dictionary mapping basin_stream_id to target_id.

    Methods:
    - __init__(self, df: gpd.GeoDataFrame)
        Initializes the PathingFixer object with the given GeoDataFrame and performs necessary setup.
    - make_dicts(self)
        Creates the initial dictionaries for mapping stream_ids, tdx_ids, source_ids, and target_ids.
    - new_source_and_target_dicts(self)
        Updates the dictionaries for mapping source_ids and target_ids.
    - add_fixed_targets_and_sources(self)
        Adds fixed target_ids and source_ids to the GeoDataFrame.

    Usage:
    # Create PathingFixer object
    fixer = PathingFixer(df)

    # Access attributes
    fixer.df
    fixer.basin_stream_id_to_stream_id
    fixer.stream_id_and_tdx_id_to_basin_stream_id
    fixer.tdx_id_to_source_id
    fixer.tdx_id_to_target_id
    fixer.basin_stream_id_to_sources
    fixer.basin_stream_id_to_targets

    # Call methods
    fixer.make_dicts()
    fixer.new_source_and_target_dicts()
    fixer.add_fixed_targets_and_sources()
    """
    def __init__(self, df: gpd.GeoDataFrame):
        self.df = df
        self.df['basin_stream_id'] = df.index
        self.basin_stream_id_to_stream_id = defaultdict(lambda: -1)
        self.stream_id_and_tdx_id_to_basin_stream_id = defaultdict(lambda: -1)
        self.tdx_id_to_source_id = defaultdict(lambda: -1)
        self.tdx_id_to_target_id = defaultdict(lambda: -1)
        self.basin_stream_id_to_sources = defaultdict(lambda: np.array([], np.int32))
        self.basin_stream_id_to_targets = defaultdict(lambda: -1)
        self.make_dicts()
        self.add_fixed_targets_and_sources()

    def make_dicts(self):
        for (
                basin_stream_id, stream_id, tdx_stream_id, from_tdx,
                target_stream_id, tdx_target_id, source_stream_ids, tdx_source_ids
        ) in zip(
            self.df.basin_stream_id,  self.df.stream_id, self.df.tdx_stream_id, self.df.from_tdx,
            self.df.target_stream_id, self.df.tdx_target_id, self.df.source_stream_ids, self.df.tdx_source_ids
        ):
            self.basin_stream_id_to_stream_id[basin_stream_id] = stream_id
            self.stream_id_and_tdx_id_to_basin_stream_id[(stream_id, tdx_stream_id)] = basin_stream_id
            if from_tdx:
                if target_stream_id == -1 and tdx_target_id != -1:
                    self.tdx_id_to_source_id[tdx_stream_id] = basin_stream_id

                if len(source_stream_ids) == 0 and len(tdx_source_ids) != 0:
                    self.tdx_id_to_target_id[tdx_stream_id] = basin_stream_id
        self.new_source_and_target_dicts()

    def new_source_and_target_dicts(self):
        for (
                basin_stream_id, stream_id, tdx_stream_id, from_tdx,
                target_stream_id, tdx_target_id, source_stream_ids, tdx_source_ids
        ) in zip(
            self.df.basin_stream_id,  self.df.stream_id, self.df.tdx_stream_id, self.df.from_tdx,
            self.df.target_stream_id, self.df.tdx_target_id, self.df.source_stream_ids, self.df.tdx_source_ids
        ):
            if target_stream_id == -1 and tdx_target_id != -1 and from_tdx:
                self.basin_stream_id_to_targets[basin_stream_id] = self.tdx_id_to_target_id[tdx_target_id]
            else:
                self.basin_stream_id_to_targets[basin_stream_id] = self.stream_id_and_tdx_id_to_basin_stream_id[
                    (target_stream_id, tdx_stream_id)
                ]

            if len(source_stream_ids) == 0 and len(tdx_source_ids) != 0 and from_tdx:
                self.basin_stream_id_to_sources[basin_stream_id] = np.array([
                    self.tdx_id_to_source_id[tdx_source_id] for tdx_source_id in tdx_source_ids
                ], np.int32)
            elif len(source_stream_ids) != 0:
                self.basin_stream_id_to_sources[basin_stream_id] = np.array([
                    self.stream_id_and_tdx_id_to_basin_stream_id[(source_stream_id, tdx_stream_id)]
                    for source_stream_id in source_stream_ids
                ], np.int32)
    def add_fixed_targets_and_sources(self):
        self.df['basin_target_id'] = self.df.basin_stream_id.map(self.basin_stream_id_to_targets)
        self.df['basin_source_ids'] = self.df.basin_stream_id.map(self.basin_stream_id_to_sources)


def fix_merged_dfs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to be fixed.

    Returns
    -------
    gpd.GeoDataFrame
        The fixed GeoDataFrame with the following columns:
        - 'stream_id'
        - 'target_stream_id'
        - 'source_stream_ids'
        - 'stream_order'
        - 'from_tdx'
        - 'tdx_stream_id'
        - 'geometry'
    """
    stream_order_fixer = StreamOrderFixer(gdf)
    path_fixer = PathingFixer(stream_order_fixer.init_df)
    df = path_fixer.df
    df['stream_id'] = df['basin_stream_id']
    df['target_stream_id'] = df['basin_target_id']
    df['source_stream_ids'] = df['basin_source_ids']
    df['stream_order'] = df['fixed_stream_order']
    return df[[
        'stream_id', 'target_stream_id', 'source_stream_ids', 'stream_order', 'from_tdx', 'tdx_stream_id', 'geometry'
    ]]

#
# if __name__ == '__main__':
#     from wwvec.paths import ppaths
#     from water.basic_functions import tt, time_elapsed
#     s = tt()
#     print('Loading data')
#     gdf = gpd.read_parquet(ppaths.data / f'basins_level_2/1020000010_waterways.parquet')
#     time_elapsed(s)
#     # print('Stream order fixer')
#     # s = tt()
#     stream_order_fixer = StreamOrderFixer(gdf)
#     stream_order_fixer.investigate_all()
#     stream_order_fixer.add_fixed_stream_order()
#     # time_elapsed(s)
#
#     s = tt()
#     print('Pathing fixer')
#     path_fixer = PathingFixer(gdf)
#     time_elapsed(s)
#     path_fixer.add_fixed_targets_and_sources()
#     time_elapsed(s)
#     print(gdf.columns)