import geopandas as gpd
from functools import cached_property


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
