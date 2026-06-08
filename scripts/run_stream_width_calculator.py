from wwvec.width_calculation.stream_width import calculate_widths_for_hydro2_index, tt, time_elapsed
from wwvec.paths import BasinPaths, PolygonizedPaths, Path, ppaths
import geopandas as gpd
from multiprocessing import Process

hydro = gpd.read_file(ppaths.hydrobasins)
hydro.sort_values(by='HYBAS_ID', inplace=True)
if __name__ == '__main__':
    save_dir = ppaths.data / 'basins_level_2_with_width'
    lines_dir = ppaths.data / 'basins_level_2_with_length_fixed'
    poly_dir = ppaths.data / 'polygonized_level_2'
    for idx, hydro_id in enumerate(hydro.HYBAS_ID):
        s = tt()
        print(f'Working on {hydro_id} ({idx + 1}/{len(hydro.HYBAS_ID)})')
        if not (save_dir/f'{hydro_id}.parquet').exists():
            inputs = dict(
                hydro2_id=hydro_id, waterways_dir=lines_dir, polygon_dir=poly_dir, temp_dir=ppaths.merged_temp,
                save_dir=ppaths.data / 'basins_level_2_with_width', number_of_segments=5,
                processes_per_proc=50, num_proc=30
            )
            p = Process(target=calculate_widths_for_hydro2_index, kwargs=inputs)
            p.start()
            p.join()
            p.close()
            time_elapsed(s)
            print('\n'*2)