import os
import sys
from pathlib import Path
import yaml


def save_yaml(file_name: Path, obj) -> None:
    with open(file_name, 'w') as file:
        yaml.dump(obj, file)


def open_yaml(file_name: Path):
    with open(file_name, 'r') as file:
        obj = yaml.safe_load(file)
    return obj


package_path = Path(__file__).parent
src_path = package_path.parent
if (Path(os.getcwd())/'configuration_files').exists():
    base_path = Path(os.getcwd())
elif (Path(os.getcwd()).parent/'configuration_files').exists():
    base_path = Path(os.getcwd()).parent
elif (Path(os.getcwd()).parent.parent/'configuration_files').exists():
    base_path = Path(os.getcwd()).parent.parent
elif (src_path.parent/'configuration_files').exists():
    base_path = src_path.parent
elif (Path(sys.path[0])/'configuration_files').exists():
    base_path = Path(sys.path[0])
elif (Path(sys.path[0]).parent/'configuration_files').exists():
    base_path = Path(sys.path[0]).parent
else:
    raise Exception('Can\'t find configuration directory... '
                    'A directory named "configuration_files" must exist'
                    'in the same directory as this script '
                    'or in this script\'s parent directory'
                    f'Checked:'
                    f'{os.getcwd()}'
                    f'{sys.path[0]}'
                    )


class ProjPaths:
    """
    paths for the project
    """

    def __init__(self,
                 base: Path = base_path
                 ) -> None:
        self.base_path = base
        self.configuration_files = self.base_path/'configuration_files'
        self._path_config = {}
        if (self.configuration_files/'path_configuration.yaml').exists():
            self._path_config = open_yaml(self.configuration_files/'path_configuration.yaml')
            if self._path_config is None:
                self._path_config = {}
            if self._path_config.setdefault('directories', {}) is None:
                self._path_config['directories'] = {}
            if self._path_config.setdefault('files', {}) is None:
                self._path_config['files'] = {}
        self.data = self.add_directory('data', self.base_path)
        self.tdx_streams = self.add_directory('tdx_streams', self.data)
        self.tdx_basins = self.add_directory('tdx_basins', self.data)
        self.merged_temp = self.add_directory('merged_temp', self.data)
        self.hydrobasins = self.add_file(
            'hydrobasins_level2', 'geojson', self.data, 'hydrobasins'
        )
        for key in self._path_config['directories']:
            self.__setattr__(key, self.add_directory(key, self.data))
        for key in self._path_config['files']:
            self.__setattr__(key, self.add_file(key, '', self.data))

    def add_file(self, file_name, extension, file_parent, key=None):
        key = file_name if key is None else key
        if key in self._path_config['files']:
            return Path(self._path_config['files'][key])
        else:
            return file_parent/f'{file_name}.{extension}'

    def add_directory(self, directory_name, directory_parent, key=None):
        key = directory_name if key is None else key
        if key not in self._path_config['directories']:
            directory_path = directory_parent/directory_name
        else:
            directory_path = Path(self._path_config['directories'][key])
        if not directory_path.exists():
            directory_path.mkdir()
        return directory_path

    def get_tdx_stream_network_file(self, hydro2_id):
        stream_network_name = self._path_config['file_names']['tdx_stream_network'].replace('hydroidrpl', hydro2_id)
        return self.tdx_streams/stream_network_name

    def get_tdx_basin_file(self, hydro2_id):
        basin_name = self._path_config['file_names']['tdx_basins'].replace('hydroidrpl', hydro2_id)
        return self.tdx_basins/basin_name


ppaths = ProjPaths()


class BasinPaths:
    """Paths for an individual basin"""
    def __init__(self, hydro2_id: int, stream_id: int):
        self.elevation_grids = self.add_directory('elevation_grids')
        self.waterway_grids = self.add_directory('waterway_grids')
        self.tdx_streams = self.add_directory('tdx_streams')
        self.tdx_basins = self.add_directory('tdx_basins')
        self.vectorized = self.add_directory('vectorized')
        self.hydro = self.vectorized/f'hydro2_{hydro2_id}'
        self.hydro.mkdir(exist_ok=True)
        self.save_path = self.hydro / f'stream_id_{stream_id}.parquet'

    @staticmethod
    def add_directory(directory_name):
        if hasattr(ppaths, directory_name):
            return getattr(ppaths, directory_name)
        else:
            return ppaths.add_directory(directory_name, ppaths.data)


# if __name__ == '__main__':
#     from pprint import pprint
#     # pprint(ppaths.__dict__)
#     for key, value in ppaths.__dict__.items():
#         if hasattr(value, 'exists'):
#             print(key, value, value.exists())
#     basin_paths = BasinPaths(hydro2_id=1020034170, stream_id=1)
#
#     print('\n')
#     # pprint(basin_paths.__dict__)
#     for key, value in basin_paths.__dict__.items():
#         if hasattr(value, 'exists'):
#             print(key, value, value.exists())
