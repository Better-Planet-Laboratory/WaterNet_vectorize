from pathlib import Path
from water.basic_functions import ppaths


class BasinPaths:
    def __init__(self, hydro2_id: int, stream_id: int):
        self.elevation_path = ppaths.country_data/'elevation'
        self.waterways_grid = ppaths.country_data/'all_countries_raster_grids'
        self.tdx_streams = ppaths.country_data/'tdx_streams'
        self.tdx_basins = ppaths.country_data/'tdx_basins'
        self.vectorized = ppaths.country_data/'vectorized'
        self.hydro = self.vectorized/f'hydro2_{hydro2_id}'
        self.hydro.mkdir(exist_ok=True)
        self.save_path = self.hydro / f'stream_id_{stream_id}.parquet'