import shapely
from wwvec.basin_vectorization.basin_class import BasinData, post_connections_clean
from wwvec.basin_vectorization.connect import Connector
from wwvec.basin_vectorization.cycle_remover import CycleRemover
from wwvec.basin_vectorization.vectorize import Vectorizer
from wwvec.basin_vectorization.local_stream_order import NodeGenerator, StreamGenerator
from wwvec.paths import BasinPaths
import numpy as np
import geopandas as gpd
import pandas as pd
import os
from typing import Union
PolygonType = Union[shapely.Polygon, shapely.MultiPolygon]
LineStringType = Union[shapely.LineString, shapely.MultiLineString]


def connect_disconnected_components(basin_data: BasinData, **kwargs) -> np.ndarray:
    """
    Connects disconnected components in the input basin_data.

    Parameters
    ----------
    basin_data : BasinData
        The data representing the basin.

    Returns
    -------
    connected_grid : ndarray
        The grid after connecting the disconnected components.
    """
    connected_grid = basin_data.connected_grid
    if basin_data.main_component > 0:
        connector = Connector(basin_data, **kwargs)
        paths, targets, components_seen = connector.get_paths()
        unique_components = np.unique(connector.component_grid)
        unseen_components = [component for component in unique_components if component not in components_seen]
        connected_grid[np.where(np.isin(connector.component_grid, unseen_components))] = 0
        for node in targets:
            path, index = paths.get(node, {'path': [], 'i': 0}).values()
            if len(path) > 0:
                path = np.array(path)
                connected_grid[path[:, 0], path[:, 1]] = 1
    connected_grid[basin_data.waterway_grid == 1] = 2
    return connected_grid


def remove_cycles_and_make_gdfs(
        vectorizer: Vectorizer, basin_data: BasinData
) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    vectorizer : Vectorizer
        Object that contains information about the waterway network.

    basin_data : BasinData
        Object that contains information about the basin data.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the waterways with cycles removed.

    """
    init_new_waterways = vectorizer.line_strings
    new_waterways = init_new_waterways.copy()
    if len(vectorizer.intersection_points) > 0:
        base_gdf = new_waterways[new_waterways.from_tdx]
        cycle_remover = CycleRemover(
            waterways=init_new_waterways, elevation_data=basin_data.basin_elevation,
            intersection_points=vectorizer.intersection_points
        )
        new_waterways = cycle_remover.find_paths()
        num_new = len(new_waterways)
        new_waterways = gpd.GeoDataFrame(
            {'from_tdx': [False]*num_new, 'geometry': new_waterways},
            crs=4326
        )
        new_waterways = pd.concat([base_gdf, new_waterways], ignore_index=True)
    return new_waterways


def run_for_basin(
        basin_geometry: PolygonType, stream_geometry: LineStringType,
        old_target_id: int, old_source_ids: list[int],
        hydro2_id: int, stream_id: int, old_stream_order: int, overwrite=False, **kwargs
) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    basin_geometry : PolygonType
        The geometry representing the basin.
    stream_geometry : LineStringType
        The geometry representing the stream.
    old_target_id : int
        The tdx target stream id.
    old_source_ids : list[int]
        The tdx ids for the source streams.
    hydro2_id : int
        The hydro2 id for the basin.
    stream_id : int
        The stream id for the basin.
    old_stream_order : int
        The stream order for the old stream.
    overwrite : bool, optional
        Indicates whether to overwrite the existing data in the save path. The default is False.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame containing the generated stream data for the basin.

    """
    basin_paths = BasinPaths(stream_id=stream_id, hydro2_id=hydro2_id)
    if basin_paths.save_path.exists() and not overwrite:
        return gpd.GeoDataFrame()
    elif basin_paths.save_path.exists() and overwrite:
        os.remove(basin_paths.save_path)
    basin_data = BasinData(
        basin_geometry=basin_geometry, stream_geometry=stream_geometry, paths=basin_paths, **kwargs
    )
    connected_grid = connect_disconnected_components(basin_data)
    thin_grid = post_connections_clean(
        connected_grid, elevation_grid=basin_data.elevation_grid, waterway_grid=basin_data.waterway_grid
    )
    waterway_line_strings = [stream_geometry] if not hasattr(stream_geometry, 'geoms') else list(stream_geometry.geoms)
    vectorizer = Vectorizer(thin_grid, waterway_line_strings, basin_data)
    new_waterways = remove_cycles_and_make_gdfs(vectorizer, basin_data)
    node_generator = NodeGenerator(
        new_line_strings=new_waterways[~new_waterways.from_tdx].geometry,
        old_line_strings=new_waterways[new_waterways.from_tdx].geometry, old_stream_order=old_stream_order
    )
    stream_generator = StreamGenerator(
        node_generator, tdx_stream_id=stream_id, old_target=old_target_id, old_sources=old_source_ids
    )
    stream_generator.gdf.to_parquet(basin_paths.save_path)
    return stream_generator.gdf


def merge_dfs(input_list: list) -> gpd.GeoDataFrame:
    """
    Parameters
    ----------
    input_list : list
        A list of dictionaries containing 'hydro2_id' and 'stream_id' values.

    Returns
    -------
    merged_df : DataFrame
        A merged DataFrame containing the data from the specified paths.
    """
    paths = [BasinPaths(hydro2_id=val['hydro2_id'], stream_id=val['stream_id']).save_path for val in input_list]
    merged_df = pd.concat([gpd.read_parquet(path) for path in paths if path.exists()], ignore_index=True)
    return merged_df




if __name__ == "__main__":
    from water.basic_functions import ppaths, get_country_polygon, tt, time_elapsed
    import warnings
    import importlib
    import wwvec
    importlib.reload(wwvec.basin_vectorization.basin_class)
    importlib.reload(wwvec.basin_vectorization.connect)
    importlib.reload(wwvec.basin_vectorization.vectorize)
    importlib.reload(wwvec.basin_vectorization.vectorize)
    importlib.reload(wwvec.basin_vectorization.local_stream_order)
    from wwvec.basin_vectorization.local_stream_order import NodeGenerator, StreamGenerator
    from wwvec.basin_vectorization.vectorize import Vectorizer
    from wwvec.basin_vectorization.basin_class import BasinData
    from wwvec.basin_vectorization.connect import Connector
    # warnings.filterwarnings("error", category=RuntimeWarning)
    s = tt()
    # bbox = (32.1, .5, 33.1, 1)
    # bbox = (29.1, -2.5, 30, -1.5)
    # y, x = -1.8248828178023107, 30.424549297438947
    y, x = 6.106896, 35.945688

    bbox = (x - .001, y - .001, x + .001, y + .001)
    # 0.514538, 32.426492
    # shapelybox = shapely.box(32.5, -.5, 33.5, .5)
    hydro_level2 = gpd.read_file(ppaths.country_data/'basins/hybas_af_lev01-12_v1c/hybas_af_lev02_v1c.shp', bbox=bbox)
    h2_id = hydro_level2.reset_index().HYBAS_ID[0]
    # sid = 265071
    streams_path = ppaths.country_data/f'tdx_streams/basin_{h2_id}.gpkg'
    basins_path = ppaths.country_data/f'tdx_basins/basin_{h2_id}.gpkg'
    all_streams = gpd.read_file(streams_path, bbox=bbox)
    all_basins = gpd.read_file(basins_path, bbox=bbox)
    print(len(all_streams))
    # sid = all_streams.LINKNO.to_list()[0]
    # ax = all_streams.plot()
    # all_basins.exterior.plot(ax=ax, color='black')
    times = []
    num_streams = []
    print(len(all_basins))
    sid_list = all_streams.LINKNO.to_list()
    all_streams = all_streams.set_index('LINKNO')
    ind = 10
    # for sid in sid_list[ind:ind+1]:
    for sid in sid_list[:100]:
        basin_geometries = all_basins[all_basins.streamID == sid].reset_index(drop=True)
        if len(basin_geometries) > 0:
            basin_geometries['area'] = basin_geometries.area
            basin_geometries = basin_geometries.sort_values(by='area', ascending=False).reset_index(drop=True)
        basin_geom = basin_geometries.geometry[0]
        stream_geom = all_streams.loc[sid, 'geometry']
        old_order = all_streams.loc[sid, 'strmOrder']
        old_target = all_streams.loc[sid, 'DSLINKNO']
        old_sources = [all_streams.loc[sid, 'USLINKNO1'], all_streams.loc[sid, 'USLINKNO2']]
        # print(old_target, old_sources)
        if old_sources == [-1, -1]:
            old_sources = []
        s = tt()
        # try:
        new = run_for_basin(
            stream_id=sid, hydro2_id=h2_id, basin_geometry=basin_geom, stream_geometry=stream_geom,
            old_target_id=old_target, old_source_ids=old_sources, old_stream_order=old_order, overwrite=True, plot_data=True
        )
        # except:
        #     ax = gpd.GeoSeries([basin_geom]).plot()
        #     gpd.GeoSeries([stream_geom]).plot(ax=ax, color='orange')
        #     break
        # node_gen = NodeGenerator(
        #     new_line_strings=new[~new.from_tdx].geometry, old_line_strings=new[new.from_tdx].geometry, old_stream_order=6
        # )
        # stream_gen = StreamGenerator(node_gen)
        if new.stream_id.max() + 1 == 149:
            ax = new.plot('stream_order', legend=True)
            break
        num_streams.append(new.stream_id.max()+1)
        time_elapsed(s)
        # print(len(new[~new.from_tdx]), len(new[new.from_tdx]))
        times.append(tt()-s)
        # gpd.GeoSeries([basin_geom], crs=4326).boundary.plot(ax=ax)
        # init_new.plot()
    new.plot('from_tdx')
    print(np.mean(times))
    print(max(num_streams))