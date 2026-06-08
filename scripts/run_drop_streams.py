from wwvec.width_calculation.drop_and_reorder_streams import run_stream_remover, tt, time_elapsed
from wwvec.paths import ppaths
from wwvec.polygon_vectorization._tools import SharedMemoryPool
import geopandas as gpd
import pandas as pd
from multiprocessing import Process
hydro = gpd.read_file(ppaths.hydrobasins)
# hydro = hydro[hydro.HYBAS_ID.isin([4020034510])]
# hydro.to_file(ppaths.data/f'hydrobasins_level_2.fgb')
stream_dir = ppaths.data/'basins_level_2_with_width'
save_dir = ppaths.data/'basins_level_2_removed_new'
save_dir.mkdir(exist_ok=True)
inputs_list = []
for idx, hydro_id in enumerate(hydro.HYBAS_ID):
    stream_path = stream_dir/f'{hydro_id}.parquet'
    save_path = save_dir/f'{hydro_id}.parquet'
    # if not save_path.exists():
    #     print(stream_path)
    # print(f'Working on {hydro_id} ({idx+1}/{len(hydro)})')
    if not save_path.exists():
        inputs_list.append({'stream_path': stream_path, 'save_path':save_path})
        # s = tt()
        # p = Process(target=run_stream_remover, args=(stream_path, save_path,))
        # p.start()
        # p.join()
        # p.close()
        # time_elapsed(s, 2)
        # print('\n\n')
SharedMemoryPool(
    run_stream_remover, inputs_list, 2, use_kwargs=True, print_progress=True,
    terminate_memory_usage_percent=95, terminate_on_error=True, max_memory_usage_percent=75
).run()
