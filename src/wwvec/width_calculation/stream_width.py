import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
from pyproj import Geod
import pyarrow as pa
from wwvec.paths import BasinPaths, PolygonizedPaths, Path, ppaths
from wwvec.polygon_vectorization._tools import printdf, tt, time_elapsed, SharedMemoryPool, delete_directory_contents
import warnings
import gc

warnings.filterwarnings("ignore")
# For each basin, select all polygons within 10km of the basin, union the polygons.
#   To do this,
# To determine width, make lines every nth percentile.
# to do this, determine endpoints, use those to choose direction, center perpendicular line at the centroid.
# Take the length to be the width.

# This should be run with multiprocessing, probably at the basin level, or rather on a list of basins.


class StreamWidthCalculator:
    def __init__(
            self, basin_gdf: gpd.GeoDataFrame,
            polygons_gdf: gpd.GeoDataFrame,
            tdx_tree: shapely.STRtree,
            tdx_tree_id_to_tdx_id: dict,
            number_of_segments: int=9,
            buffer_distance: float=.1,
            max_steam_width: float=None
    ):
        self.basin_gdf = basin_gdf.copy()
        self.basin_gdf['width_m'] = 1
        self.basin_gdf['length_width_ratio'] = 2
        if not hasattr(self.basin_gdf, 'intersects_lake'):
            self.basin_gdf['intersects_lake'] = False
        self.number_of_segments = number_of_segments
        self.segment_step = 1/self.number_of_segments
        self.half_segment_step = self.segment_step/2
        self.polygons_gdf = polygons_gdf
        self.tdx_tree = tdx_tree
        self.tdx_tree_id_to_tdx_id = tdx_tree_id_to_tdx_id
        self.tdx_gdf = self.basin_gdf[self.basin_gdf.from_tdx & ~self.basin_gdf.intersects_lake].reset_index(drop=True)
        self.lake_gdf = self.basin_gdf[self.basin_gdf.intersects_lake].reset_index(drop=True)
        self.lake_gdf['width_m'] = self.lake_gdf['length_m']
        self.lake_gdf['length_width_ratio'] = 1
        self.basin_gdf = self.basin_gdf[~(self.basin_gdf.from_tdx | self.basin_gdf.intersects_lake)].reset_index(drop=True)
        self.polygon_time = 0
        self.width_time = 0
        if len(self.basin_gdf)>0:
            s = tt()
            self.polygon = self._make_polygon()
            self.polygon_time = tt() - s
            # time_elapsed(s, 2)
        geod = Geod(ellps='WGS84')
        self.length_func = np.frompyfunc(lambda geometry: geod.geometry_length(geometry), 1, 1)

    def _make_polygon(self):
        centroids = self.basin_gdf.centroid
        # s = tt()
        buffer_amounts = shapely.length(self.basin_gdf.geometry)
        buffered_centroids = shapely.unary_union(
            [centroid.buffer(buffer_amount, cap_style='square') for centroid, buffer_amount in zip(centroids, buffer_amounts)]
        )
        # print(buffer_amounts)
        # time_elapsed(s, 4)
        # s = tt()
        # buffered_centroid1 = shapely.unary_union(self.basin_gdf.centroid).buffer(self.buffer_distance)
        tree_ids = self.tdx_tree.query(buffered_centroids, predicate='intersects')
        tdx_ids = {self.tdx_tree_id_to_tdx_id[id] for id in tree_ids}
        # time_elapsed(s, 4)
        # s = tt()
        # print(len(tdx_ids))
        self.polygon = shapely.unary_union(
            self.polygons_gdf[self.polygons_gdf.tdx_stream_id.isin(tdx_ids)].reset_index(drop=True).geometry
        )
        # time_elapsed(s, 4)
        return self.polygon

    def _generate_perpendicular_lines(
            self, lines: np.ndarray[shapely.LineString], number_of_segments: int=None
    ):
        if number_of_segments is None:
            segment_step = self.segment_step
            number_of_segments = self.number_of_segments
        else:
            segment_step = 1/number_of_segments
        num_lines = len(lines)
        line_lengths = shapely.length(lines)
        extended_lengths = []
        for length in line_lengths:
            extended_lengths += [length]*number_of_segments
        line_lengths = np.array(extended_lengths)
        endpoints = np.array([
            shapely.line_interpolate_point(lines, ((i + 1) // 2) * segment_step, normalized=True)
            for i in range(2 * number_of_segments)
        ]).T.reshape(num_lines * number_of_segments, -1)
        centroids = np.array([
            shapely.line_interpolate_point(lines, i * segment_step + segment_step / 2, normalized=True)
            for i in range(number_of_segments)
        ]).T.flatten()

        heads = shapely.get_coordinates(endpoints[:, 0])
        tails = shapely.get_coordinates(endpoints[:, 1])
        centroid_coords = shapely.get_coordinates(centroids)
        midpoints = (heads + tails) / 2
        heads = heads - midpoints
        tails = tails - midpoints

        rotation_matrix = np.array([[0, -1], [1, 0]])
        heads = np.matmul(rotation_matrix, heads.T).T
        tails = np.matmul(rotation_matrix, tails.T).T
        lengths = line_lengths/np.sqrt((heads[:, 0] - tails[:, 0]) ** 2 + (heads[:, 1] - tails[:, 1]) ** 2)
        lengths = lengths.reshape(-1, 1)
        # lengths = line_lengths
        heads = lengths * heads
        tails = lengths * tails
        heads = heads + centroid_coords
        tails = tails + centroid_coords
        x = np.stack([heads[:, 0], tails[:, 0]], axis=1)
        y = np.stack([heads[:, 1], tails[:, 1]], axis=1)
        segments = shapely.linestrings(x, y)
        segments = shapely.intersection(segments, self.polygon)
        return np.array_split(segments, len(lines))

    def _find_segment_intersecting_lines(
            self, lines: np.ndarray[shapely.LineString], segments: np.ndarray[np.ndarray[shapely.GeometryType]]
    ):
        segments_intersecting_lines = []
        for line, segment_mls in zip(lines, segments):
            segment_geoms = []
            segments_intersecting_lines.append(segment_geoms)
            for segment in segment_mls:
                if hasattr(segment, "geoms"):
                    geoms_array = np.array(segment.geoms)
                    geoms_intersecting = geoms_array[shapely.intersects(geoms_array, line)]
                    if len(geoms_intersecting) > 0:
                        if len(geoms_intersecting)==1:
                            segment_geoms.append(geoms_intersecting[0])
                        else:
                            segment_geoms.append(shapely.unary_union(geoms_intersecting))
                    else:
                        segment_geoms.append(shapely.LineString())
                else:
                    segment_geoms.append(segment)
        return segments_intersecting_lines

    def _calculate_segment_width(self, segments_list):
        segment_lengths = []
        for segments in segments_list:
            median_length = 0
            if len(segments) > 0:
                median_length = np.median(self.length_func(np.array(segments)))
            segment_lengths.append(median_length)
        return segment_lengths

    def run(self):
        lines = self.basin_gdf.geometry.to_numpy()
        s = tt()
        if len(lines)>0:
            # s = tt()
            perpendicular_lines = self._generate_perpendicular_lines(lines)
            # time_elapsed(s, 2)
            # s = tt()
            perpendicular_lines = self._find_segment_intersecting_lines(lines, perpendicular_lines)
            # time_elapsed(s, 2)
            # s = tt()
            widths = self._calculate_segment_width(perpendicular_lines)
            # time_elapsed(s, 2)
            self.basin_gdf['width_m'] = widths
            self.basin_gdf[f'length_width_ratio'] = self.basin_gdf['length_m'] / (self.basin_gdf['width_m'] + 1)
        self.basin_gdf = pd.concat([self.basin_gdf, self.tdx_gdf, self.lake_gdf], ignore_index=True)
        self.width_time = tt() - s
        return self.basin_gdf

def _run_on_waterway_subset(
        save_dir: Path,
        waterways_subset_gdf: gpd.GeoDataFrame,
        polygons_gdf: gpd.GeoDataFrame,
        tdx_tree: shapely.STRtree,
        tdx_tree_index_to_tdx_id: dict,
        number_of_segments: int = 9,
        buffer_distance: float = .1,
        max_steam_width: float = None,
        proc_id: int = 0
):
    unique_ids = waterways_subset_gdf.tdx_stream_id.unique()
    waterways_with_width = []
    start = tt()
    times_list = []
    # tdx_tree = shapely.STRtree(polygons_gdf.geometry)
    # tdx_tree_index_to_tdx_id = {ind: tdx_ind for (ind, tdx_ind) in enumerate(polygons_gdf.tdx_stream_id)}
    for idx, id in enumerate(unique_ids):
        s = tt()
        # try:
        basin_gdf = waterways_subset_gdf[waterways_subset_gdf.tdx_stream_id == id].reset_index(drop=True)
        width_calculator = StreamWidthCalculator(
            basin_gdf=basin_gdf, polygons_gdf=polygons_gdf, tdx_tree=tdx_tree,
            tdx_tree_id_to_tdx_id=tdx_tree_index_to_tdx_id, number_of_segments=number_of_segments,
            buffer_distance=buffer_distance, max_steam_width=max_steam_width
        )
        waterways_with_width.append(width_calculator.run())
        times_list.append(tt()-s)
        if tt()-s>60*2:
            poly_time = round(width_calculator.polygon_time/60)
            run_time = round(width_calculator.width_time/60)
            print(f'    Basin {id} took longer than 2 minutes ({round((tt()-s)/60)} minutes, polygon time {poly_time}, width time {run_time})')
        # if idx == len(unique_ids)//2:
        #     time_elapsed(start, 2)
        #     print(f'  {np.mean(times_list): .4f}')
        # except:
        #     print(f'Issue with {id}')
    if proc_id % 75 == 0:
        mean_length = waterways_subset_gdf.length_m.mean()
        mean_times = np.mean(times_list)
        print(f'  ({proc_id}) Average time: {mean_times:.4f}, mean waterway length: {mean_length:.2f}')
    waterways_with_width_gdf = pd.concat(waterways_with_width, ignore_index=True)
    waterways_with_width_gdf.to_parquet(save_dir / f'temp_{unique_ids[0]}', index=False)


def remove_holes(polygon: shapely.Polygon):
    if polygon.interiors:
        return shapely.Polygon(polygon.exterior.coords)
    else:
        return polygon


def _make_inputs_list(
        waterways, polygons, temp_dir, number_of_segments: int = 5, buffer_distance: float = .1,
        max_steam_width: float = None, total_processes: int=120
):
    # tdx_lines = waterways[waterways.from_tdx].reset_index(drop=True)
    # tdx_lines = waterways
    print('    Removing unnecessary polygons')
    s = tt()
    init_num_polygons = len(polygons)
    lines_tree = shapely.STRtree(waterways[~waterways.intersects_lake & ~waterways.from_tdx].geometry)
    poly_inds, _ = lines_tree.query(polygons.geometry, predicate='intersects')
    polygons = polygons[polygons.index.isin(poly_inds)].reset_index(drop=True)
    num_polygons = len(polygons)
    print(f'    init num polygons: {init_num_polygons}, num polygons: {num_polygons}')
    time_elapsed(s, 4)
    s = tt()
    print('    Removing polygon holes')
    polygons['geometry'] = polygons.geometry.apply(remove_holes)
    time_elapsed(s, 3)
    print('    Making input lists')
    tdx_tree = shapely.STRtree(polygons.geometry)
    tdx_tree_index_to_tdx_id = {ind: tdx_ind for (ind, tdx_ind) in enumerate(polygons.tdx_stream_id)}
    unique_indices = waterways.tdx_stream_id.unique()
    np.random.shuffle(unique_indices)
    tdx_ids_split = np.array_split(unique_indices, total_processes)
    inputs_list = []
    for ind, tdx_ids in enumerate(tdx_ids_split):
        waterways_subset = waterways[waterways.tdx_stream_id.isin(tdx_ids)].reset_index(drop=True)
        inputs_list.append(dict(
            save_dir=temp_dir, waterways_subset_gdf=waterways_subset,
            polygons_gdf=polygons, tdx_tree=tdx_tree, tdx_tree_index_to_tdx_id=tdx_tree_index_to_tdx_id,
            number_of_segments=number_of_segments, buffer_distance=buffer_distance, max_steam_width=max_steam_width,
            proc_id=ind
        ))
    print(f'    Number of inputs {len(inputs_list)}, Number of basins per process {len(tdx_ids_split[0])}')
    print(f'    Mean waterway length: {waterways.length_m.mean():.2f}')
    return inputs_list


def calculate_widths_for_hydro2_index(
        hydro2_id: int, waterways_dir: Path, polygon_dir: Path, temp_dir: Path, save_dir: Path,
        number_of_segments: int = 5, buffer_distance: float = .1, max_steam_width: float = 0.05,
        num_proc: int=30, processes_per_proc: int = 5
):
    pa.jemalloc_set_decay_ms(0)
    print('  Opening Data')
    s = tt()
    waterways = gpd.read_parquet(waterways_dir/f'{hydro2_id}.parquet')
    polygons = gpd.read_parquet(polygon_dir/f'{hydro2_id}.parquet')
    print(f'    Polygon Thresholds: {polygons.threshold.unique()}')
    time_elapsed(s, 2)
    delete_directory_contents(temp_dir)
    s = tt()
    print('  Making inputs list')
    inputs_list = _make_inputs_list(
        waterways, polygons, temp_dir, number_of_segments, buffer_distance, max_steam_width, num_proc*processes_per_proc
    )
    del waterways
    pa.default_memory_pool().release_unused()
    time_elapsed(s, 2)
    print('  Starting width calculation')
    if num_proc == 1:
        for inputs in inputs_list:
            _run_on_waterway_subset(**inputs)
    else:
        SharedMemoryPool(
            func=_run_on_waterway_subset, input_list=inputs_list, max_memory_usage_percent=88,
            terminate_memory_usage_percent=95, num_proc=num_proc, use_kwargs=True, print_progress=True
        ).run()
    del inputs_list
    gc.collect()
    pa.default_memory_pool().release_unused()
    gdf = pd.concat([gpd.read_parquet(file_path) for file_path in temp_dir.iterdir()], ignore_index=True)
    gdf.to_parquet(save_dir/f'{hydro2_id}.parquet', index=False)



if __name__ == '__main__':
    # hybas_id = 4020034510
    hybas_id = 2020071190

    # gdf1 = gpd.read_parquet(ppaths.data/'basins_level_2_with_width/4020034510.parquet')
    # gdf2 = gpd.read_parquet(ppaths.data/'4020034510_og_width.parquet')
    # gdf3 = gdf1.join(gdf2.set_index('stream_id')[['width_m', 'length_width_ratio']], on='stream_id', rsuffix='_og')
    # gdf3['difference'] = np.abs(gdf3['width_m'] - gdf3['width_m_og'])
    #
    199626
    lines_dir = ppaths.data/'basins_level_2_with_length_fixed'
    poly_dir = ppaths.data/'polygonized_level_2'
    calculate_widths_for_hydro2_index(
        hydro2_id=hybas_id, waterways_dir=lines_dir, polygon_dir=poly_dir, temp_dir=ppaths.merged_temp,
        save_dir=ppaths.data/'basins_level_2_with_width',
        number_of_segments=5, processes_per_proc=50, num_proc=30
    )
    # s = tt()
    # lines_gdf = gpd.read_parquet(lines_dir/f'{hybas_id}.parquet')
    # time_elapsed(s)
    # s = tt()
    # poly_gdf = gpd.read_parquet(poly_dir/f'{hybas_id}.parquet')
    # time_elapsed(s)
    # s = tt()
    # # tdx_lines = lines_gdf[lines_gdf.from_tdx].reset_index(drop=True)
    # lines_tree = shapely.STRtree(lines_gdf[~lines_gdf.intersects_lake & ~lines_gdf.from_tdx].geometry)
    # poly_inds, _ = lines_tree.query(poly_gdf.geometry)
    # time_elapsed(s)
    # s = tt()
    # poly_gdf = poly_gdf[poly_gdf.index.isin(poly_inds)].reset_index(drop=True)
    # tdx_tree = shapely.STRtree(poly_gdf.geometry)
    # s = tt()
    # time_elapsed(s)
    # tdx_tree_index_to_tdx_id = {ind: tdx_ind for (ind, tdx_ind) in enumerate(poly_gdf.tdx_stream_id)}
    # basin_gdf = lines_gdf[lines_gdf.tdx_stream_id == 470727]
    # time_elapsed(s)
    # s = tt()
    # stream_width = StreamWidthCalculator(
    #     basin_gdf, poly_gdf, tdx_tree, tdx_tree_index_to_tdx_id, number_of_segments=5
    # )
    # stream_width.run()
    # print('done')
    # print(stream_width.polygon_time, stream_width.width_time)
    # time_elapsed(s)
    # tdx_tree_index_to_tdx_id = {ind: tdx_ind for (ind, tdx_ind) in enumerate(tdx_lines.tdx_stream_id)}
    # # tree_ids = tdx_tree.query(tdx_centroids[0], predicate='intersects')
    # # tdx_ids = {tdx_tree_index_to_tdx_id[id] for id in tree_ids}
    # count = 0
    # # start = tt()
    # geod = Geod(ellps='WGS84')
    # length_func = np.frompyfunc(lambda geometry: geod.geometry_length(geometry), 1, 1)
    # num_segments_to_widths = {}
    # # tdx_ids = lines_gdf.tdx_stream_id.unique()[:10]
    # tdx_ids = [46332, 70472]
    # num_segments_list = [5]
    # for num_segments in num_segments_list:
    #     start = tt()
    #     basins_with_widths = []
    #     for tdx_id in tdx_ids:
    #         basin_gdf = lines_gdf[(lines_gdf.tdx_stream_id==tdx_id)].copy().reset_index(drop=True)
    #         for bd in [.1]:
    #             print('  ', tdx_id, bd)
    #             s = tt()
    #             # s1 = tt()
    #             stream_width = StreamWidthCalculator(
    #                 basin_gdf, poly_gdf, tdx_tree, tdx_tree_index_to_tdx_id,
    #                 number_of_segments=num_segments, buffer_distance=bd, max_steam_width=.05
    #             )
    #             stream_width.run()
                # # time_elapsed(s, 4)
                # input_lines = basin_gdf.geometry.to_numpy()
                # segs = stream_width._generate_perpendicular_lines(input_lines)
                # time_elapsed(s, 4)
                # s1 = tt()
                # lsegs = stream_width._find_segment_intersecting_lines(input_lines, segs)
                # time_elapsed(s1, 4)
                # s1 = tt()
                # lseg_lengths = []
                # for lseg in lsegs:
                #     median_length = 0
                #     if len(lseg)>0:
                #         try:
                #             median_length = np.median(length_func(np.array(lseg)))
                #         except:
                #             print(lseg)
                #             print('')
                #     lseg_lengths.append(median_length)
    #             time_elapsed(s1, 4)
    #             basin_gdf[f'width_{num_segments}'] = lseg_lengths
    #             basin_gdf[f'length_width_ratio_{num_segments}'] = basin_gdf[f'length_m']/(basin_gdf[f'width_{num_segments}'] + .00000000001)
    #             basins_with_widths.append(basin_gdf)
    #             time_elapsed(s, 2)
    #     time_elapsed(start)
    #     num_segments_to_widths[num_segments] = pd.concat(basins_with_widths, ignore_index=True)
    # joined_df = None
    # for num_segments, width_df in num_segments_to_widths.items():
    #     if joined_df is None:
    #         joined_df = width_df
    #     else:
    #         joined_df[f'width_{num_segments}'] = width_df[f'width_{num_segments}']
    #     joined_df[f'length_width_ratio_{num_segments}'] = joined_df[f'length_m']/(joined_df[f'width_{num_segments}'] + .00000000001)
    #     print(
    #         num_segments,
    #         len(joined_df[(joined_df[f'length_width_ratio_{num_segments}']>2)]),
    #         len(joined_df[(joined_df[f'length_width_ratio_{num_segments}'] > 2) & (joined_df.stream_order==1)])
    #     )
    # joined_df['diff_3_5'] = np.abs(joined_df['width_3'] - joined_df['width_5'])
    # joined_df['diff_3_10'] = np.abs(joined_df['width_3'] - joined_df['width_10'])
    # joined_df['diff_5_10'] = np.abs(joined_df['width_10'] - joined_df['width_5'])
    # joined_df[['diff_3_5', 'diff_3_10', 'diff_5_10']].describe([.001*i for i in range(990, 1000)])
    # joined_df[joined.]
    # time_elapsed(s, 2)
    # s = tt()
    # tree_ids = tdx_tree.query(tdx_centroids[0], predicate='intersects')
    # tdx_ids = {tdx_tree_index_to_tdx_id[id] for id in tree_ids}
    # polygon = shapely.unary_union(poly_gdf[poly_gdf.tdx_stream_id.isin(tdx_ids)].reset_index(drop=True).geometry)
    # time_elapsed(s, 2)

    #
    # line = lines_gdf.geometry[:10].to_numpy()
    # num_lines = len(line)
    # buffer_distance = 0.001
    # number_of_segments = 9
    # segment_step = 1/(number_of_segments)
    # segment_length = shapely.length(line)
    # # scale_factor = buffer_distance / segment_length
    # scale_factor = 1
    # endpoints = np.array([
    #     shapely.line_interpolate_point(line, ((i+1)//2) * segment_step, normalized=True)
    #     for i in range(2*number_of_segments)
    # ]).T.reshape(num_lines*number_of_segments, -1)
    # centroids = np.array([
    #     shapely.line_interpolate_point(line, i * segment_step + segment_step/2, normalized=True)
    #     for i in range(number_of_segments)
    # ]).T.flatten()
    #
    # heads = shapely.get_coordinates(endpoints[:, 0])
    # tails = shapely.get_coordinates(endpoints[:, 1])
    # x1 = np.stack([heads[:, 0], tails[:, 0]], axis=1)
    # y1 = np.stack([heads[:, 1], tails[:, 1]], axis=1)
    # seg1 = shapely.linestrings(x1, y1)
    # centroid_coords = shapely.get_coordinates(centroids)
    # midpoints = (heads + tails)/2
    # heads = heads - midpoints
    # tails = tails - midpoints
    #
    # rotation_matrix = np.array([[0, -1], [1, 0]])
    # heads = np.matmul(rotation_matrix, heads.T).T
    # tails = np.matmul(rotation_matrix, tails.T).T
    # lengths = buffer_distance/np.sqrt((heads[:, 0] - tails[:, 0]) ** 2 + (heads[:, 1] - tails[:, 1]) ** 2)
    # lengths = lengths.reshape(-1, 1)
    # # heads = lengths*heads
    # # tails = lengths*tails
    # heads = heads + centroid_coords
    # tails = tails + centroid_coords
    # # heads = heads + midpoints
    # # tails = tails + midpoints
    # x = np.stack([heads[:, 0], tails[:, 0]], axis=1)
    # y = np.stack([heads[:, 1], tails[:, 1]], axis=1)
    # segments = shapely.linestrings(x, y)
    # geoms = [line[0]] + list(segments[:number_of_segments]) + list(endpoints[:number_of_segments, 0])
    # # geoms = list(seg1[:number_of_segments]) + list(segments[:number_of_segments]) + list(endpoints[:number_of_segments, 0])
    # # geoms = list(seg1[:number_of_segments]) + list(segments[:number_of_segments])
    #
    # gdf = gpd.GeoDataFrame(geoms, columns=['geometry'], geometry='geometry', crs='EPSG:4326')
    # gdf.plot(aspect='equal')
    # geod = Geod(ellps='WGS84')
    # length_func = np.frompyfunc(lambda geometry: geod.geometry_length(geometry), 1, 1)