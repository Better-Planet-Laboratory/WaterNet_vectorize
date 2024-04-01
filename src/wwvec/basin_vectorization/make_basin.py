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
import time

def run_for_basin(
        basin_geometry, stream_geometry, hydro2_id, stream_id, overwrite=False, plot_data=False, **kwargs
):
    basin_paths = BasinPaths(stream_id=stream_id, hydro2_id=hydro2_id)
    if basin_paths.save_path.exists() and not overwrite:
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    elif basin_paths.save_path.exists() and overwrite:
        os.remove(basin_paths.save_path)
    basin_data = BasinData(
        basin_geometry=basin_geometry, stream_geometry=stream_geometry, paths=basin_paths, **kwargs
    )
    if plot_data:
        pass
        # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        # ax[0].imshow(basin_data.probability_grid)
        # ax[1].imshow(basin_data.elevation_grid)
    new_grid = basin_data.rounded_grid.copy()
    new_grid[new_grid > 0] = 1
    new_grid = new_grid.astype(np.int8)
    if plot_data:
        pass
        # fig, ax = plt.subplots()
        # ax.imshow(new_grid)
    if basin_data.main_color > 0:
        connector = Connector(basin_data, min_probability=0)
        paths, targets, colors_seen = connector.get_paths()
        unique_colors = np.unique(connector.colored_grid)
        unseen_colors = [color for color in unique_colors if color not in colors_seen]
        new_grid[np.where(np.isin(connector.colored_grid, unseen_colors))] = 0
        for node in targets:
            path, index = paths.get(node, {'path': [], 'i': 0}).values()
            for (row, col) in path:
                if new_grid[row, col] == 0:
                    # new_grid[row, col] = 1
                    new_grid[row, col] = index
                new_grid[node] = 10
        if plot_data:
            pass
            # fig, ax = plt.subplots()
            # ax.imshow(new_grid)
    new_grid[new_grid > 2] = 1
    new_grid[basin_data.waterway_grid == 1] = 2
    thin = post_connections_clean(
        new_grid, elevation_grid=basin_data.elevation_grid, waterway_grid=basin_data.waterway_grid
    )
    if plot_data:
        pass
        # fig, ax = plt.subplots()
        # ax.imshow(thin)
    bbox = basin_data.basin_probability.rio.bounds()
    x_res, y_res = np.abs(basin_data.basin_probability.rio.resolution())
    waterway_line_strings = [stream_geometry] if not hasattr(stream_geometry, 'geoms') else list(stream_geometry.geoms)
    # thin[basin_data.waterway_grid==1] = 2
    vectorizer = Vectorizer(thin, waterway_line_strings, bbox, x_res, y_res)
    # vectorizer.line_strings.plot()
    init_new_waterways = vectorizer.line_strings
    new_waterways = init_new_waterways.copy()
    try:
        base_gdf = new_waterways[new_waterways.from_tdx]
    except:
        print(stream_id, hydro2_id)
        print(new_waterways)
        raise Exception
    if len(vectorizer.intersection_points) > 0:
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
    new_waterways['stream_id'] = stream_id
    new_waterways['geometry'] = new_waterways.intersection(basin_geometry)
    # if save_path is not None:
    new_waterways.to_parquet(basin_paths.save_path)
    return new_waterways, init_new_waterways
    # except:
    #     pass


def merge_dfs(input_list, i=0):
    paths = [BasinPaths(hydro2_id=val['hydro2_id'], stream_id=val['stream_id']).save_path for val in input_list]
    # for path in paths:
    #     if not path.exists():
    #         if i == 10:
    #             raise Exception
    #         time.sleep(.5)
    #         return merge_dfs(input_list, i+1)
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
    from wwvec.basin_vectorization.vectorize import Vectorizer
    from wwvec.basin_vectorization.basin_class import BasinData
    from wwvec.basin_vectorization.connect import Connector
    # warnings.filterwarnings("error", category=RuntimeWarning)
    s = tt()
    # bbox = (32.1, .5, 33.1, 1)
    # bbox = (29.1, -2.5, 30, -1.5)
    y, x = -2.067971,30.052090
    bbox = (x - .5, y - .5, x + .5, y + .5)
    # 0.514538, 32.426492
    # shapelybox = shapely.box(32.5, -.5, 33.5, .5)
    hydro_level2 = gpd.read_file(ppaths.country_data/'basins/hybas_af_lev01-12_v1c/hybas_af_lev02_v1c.shp', bbox=bbox)
    h2_id = hydro_level2.reset_index().HYBAS_ID[0]
    sid = 265071
    streams_path = ppaths.country_data/f'tdx_streams/basin_{h2_id}.gpkg'
    basins_path = ppaths.country_data/f'tdx_basins/basin_{h2_id}.gpkg'
    all_streams = gpd.read_file(streams_path, bbox=bbox)
    all_basins = gpd.read_file(basins_path, bbox=bbox)
    # ax = all_streams.plot()
    # all_basins.exterior.plot(ax=ax, color='black')
    print(len(all_basins))
    basin_geometries = all_basins[all_basins.streamID == sid].reset_index(drop=True)
    if len(basin_geometries) > 0:
        basin_geometries['area'] = basin_geometries.area
        basin_geometries = basin_geometries.sort_values(by='area', ascending=False).reset_index(drop=True)
    basin_geom = basin_geometries.geometry[0]
    # basin_geom = all_basins[all_basins.streamID == sid].reset_index().geometry[0]
    # all_basins[all_basins.streamID == sid].plot()
    # gpd.GeoSeries([basin_geom.buffer(0)]).plot()
    # sid = all_basins.streamID[10]
    stream_geom = all_streams.set_index('LINKNO').loc[sid, 'geometry']
    # time_elapsed(s, 2)
    # print(sid, stream_geom, basin_geom)
    new, init_new = run_for_basin(
        stream_id=sid, hydro2_id=h2_id, basin_geometry=basin_geom, stream_geometry=stream_geom, overwrite=True,
        plot_data=True
    )
    # new.plot()
    init_new.plot()