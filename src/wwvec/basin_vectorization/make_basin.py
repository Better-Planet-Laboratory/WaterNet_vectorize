import shapely
from water.basic_functions import my_pool
from wwvec.basin_vectorization.basin_class import BasinData, post_connections_clean
from wwvec.basin_vectorization.connect import Connector
from wwvec.basin_vectorization.cycle_remover import CycleRemover
from wwvec.basin_vectorization.vectorize import Vectorizer
from wwvec.paths import BasinPaths
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import os
from wwvec.basin_vectorization.local_stream_order import NodeGenerator, StreamGenerator

import time
from typing import Union
PolygonType = Union[shapely.Polygon, shapely.MultiPolygon]
LineStringType = Union[shapely.LineString, shapely.MultiLineString]


def connect_disconnected_components(basin_data: BasinData, **kwargs):
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
            for (row, col) in path:
                if connected_grid[row, col] == 0:
                    connected_grid[row, col] = index
                connected_grid[node] = 10
        connected_grid[connected_grid > 2] = 1
    connected_grid[basin_data.waterway_grid == 1] = 2
    return connected_grid


def remove_cycles_and_make_gdfs(vectorizer: Vectorizer, basin_data: BasinData):
    """
    Parameters
    ----------
    vectorizer : Vectorizer
        The vectorizer object that generates the line strings representing waterways.

    basin_data : BasinData
        The basin data object that contains the basin data.

    Returns
    -------
    tuple
        A tuple containing two objects - the new waterways GeoDataFrame without cycles,
         and the initial waterways GeoDataFrame before removing cycles.

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
    return new_waterways, init_new_waterways


def run_for_basin(
        basin_geometry: PolygonType, stream_geometry: LineStringType,
        hydro2_id: int, stream_id: int, overwrite=False, **kwargs
):
    """
    Parameters
    ----------
    basin_geometry : PolygonType
        The geometry of the basin.
    stream_geometry : LineStringType
        The geometry of the stream.
    hydro2_id : int
        The hydro2 ID of the basin.
    stream_id : int
        The ID of the stream.
    overwrite : bool, optional
        Whether to overwrite the existing saved data for the basin. Default is False.
    kwargs : keyword arguments, optional
        Additional arguments that can be passed to the method.

    Returns
    -------
    new_waterways : GeoDataFrame
        The new waterways intersecting the basin geometry.
    init_new_waterways : GeoDataFrame
        The initial new waterways before intersection with the basin geometry.
    """
    basin_paths = BasinPaths(stream_id=stream_id, hydro2_id=hydro2_id)
    if basin_paths.save_path.exists() and not overwrite:
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    elif basin_paths.save_path.exists() and overwrite:
        os.remove(basin_paths.save_path)
    s = tt()
    print('basin data')
    basin_data = BasinData(
        basin_geometry=basin_geometry, stream_geometry=stream_geometry, paths=basin_paths, **kwargs
    )
    time_elapsed(s, 2)
    print('connect')
    s = tt()
    connected_grid = connect_disconnected_components(basin_data)
    time_elapsed(s, 2)
    s = tt()
    print('thin')
    thin_grid = post_connections_clean(
        connected_grid, elevation_grid=basin_data.elevation_grid, waterway_grid=basin_data.waterway_grid
    )
    time_elapsed(s, 2)
    print('vectorize')
    s = tt()
    waterway_line_strings = [stream_geometry] if not hasattr(stream_geometry, 'geoms') else list(stream_geometry.geoms)
    vectorizer = Vectorizer(thin_grid, waterway_line_strings, basin_data)
    new_waterways, init_new_waterways = remove_cycles_and_make_gdfs(vectorizer, basin_data)
    time_elapsed(s, 2)
    new_waterways['stream_id'] = stream_id
    # new_waterways.to_parquet(basin_paths.save_path)
    return new_waterways, init_new_waterways


def merge_dfs(input_list: list):
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
    y, x = -2.067971, 30.052090
    bbox = (x - .5, y - .5, x + .5, y + .5)
    # 0.514538, 32.426492
    # shapelybox = shapely.box(32.5, -.5, 33.5, .5)
    hydro_level2 = gpd.read_file(ppaths.country_data/'basins/hybas_af_lev01-12_v1c/hybas_af_lev02_v1c.shp', bbox=bbox)
    h2_id = hydro_level2.reset_index().HYBAS_ID[0]
    # sid = 265071
    streams_path = ppaths.country_data/f'tdx_streams/basin_{h2_id}.gpkg'
    basins_path = ppaths.country_data/f'tdx_basins/basin_{h2_id}.gpkg'
    all_streams = gpd.read_file(streams_path, bbox=bbox)
    all_basins = gpd.read_file(basins_path, bbox=bbox)
    # sid = all_streams.LINKNO.to_list()[0]
    # ax = all_streams.plot()
    # all_basins.exterior.plot(ax=ax, color='black')
    times = []
    print(len(all_basins))
    sid_list = all_streams.LINKNO.to_list()
    all_streams = all_streams.set_index('LINKNO')
    for sid in sid_list[:1]:
        basin_geometries = all_basins[all_basins.streamID == sid].reset_index(drop=True)
        if len(basin_geometries) > 0:
            basin_geometries['area'] = basin_geometries.area
            basin_geometries = basin_geometries.sort_values(by='area', ascending=False).reset_index(drop=True)
        basin_geom = basin_geometries.geometry[0]
        stream_geom = all_streams.loc[sid, 'geometry']
        s = tt()
        new, init_new = run_for_basin(
            stream_id=sid, hydro2_id=h2_id, basin_geometry=basin_geom, stream_geometry=stream_geom, overwrite=True,
            plot_data=True
        )
        node_gen = NodeGenerator(
            new_line_strings=new[~new.from_tdx].geometry, old_line_strings=new[new.from_tdx].geometry, old_stream_order=6
        )
        stream_gen = StreamGenerator(node_gen)
        time_elapsed(s)
        # print(len(new[~new.from_tdx]), len(new[new.from_tdx]))
        times.append(tt()-s)
        # ax = new.plot('from_tdx')
        # gpd.GeoSeries([basin_geom], crs=4326).boundary.plot(ax=ax)
        # init_new.plot()