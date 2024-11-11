

# Overview

This repository is associated with the forthcoming paper "Pierson, Matthew., and Mehrabi, Zia. 2024. Deep learning waterways for rural infrastructure development. arXiv." Please cite this paper and attribute the work if using the model or work.

The data outputs of this model (raster and vectorized versions) are stored and available under CC-BY-SA 4.0 from the following source: Pierson, Matthew., Mehrabi. Zia. 2024, WaterNet Outputs and Code, https://doi.org/10.7910/DVN/YY2XMG, Harvard Dataverse.

This repository is used to vectorize WaterNet outputs.

# Data Requirements

To use this module, it is assumed that you have raster probability grids for waterways locations (EG WaterNet outputs),
that you have the necessary TDX-Hydro data (The waterway networks and basins should be stored in different directories),
hydrobasins level 2 geojson file, and that you have the necessary elevation data.

The TDX-Hydro data and hydrobasins level 2
file can be obtained from https://earth-info.nga.mil/ (under the Geosciences tab, hydrobasins level 2 is the Basin GeoJSON File with ID Numbers)

We used COPDEM-GLO30 for our elevation data (obtained from Microsoft Planetary Computer)
and bicubically upsampled COP-DEM-GLO-90 for Armenia and Azerbaijan.

A copy of the path_configuration_template.yaml file should be made and saved as path_configuration.yaml in the
configuration_files directory.


# Installation
You may also want to install [WaterNet Training and Evaluation](https://github.com/Better-Planet-Laboratory/WaterNet_training_and_evaluation) and
[WaterNet Vectorize](https://github.com/Better-Planet-Laboratory/WaterNet_vectorize).

All code was prototyped using python 3.11.4 and pip 23.0.1.

A python environment and version handler such as [pyenv](https://github.com/pyenv/pyenv) should make those easy to obtain.

After getting your environment setup correctly, download this repository and use pip to install:

```
git clone https://github.com/Better-Planet-Laboratory/WaterNet_vectorize.git
cd WaterNet_vectorize
pip install .
```

or if you wish to edit to code

``
pip install -e .
``

# Key Functions

The two important functions in this module are 

wwvec.polygon_vectorization.make_all_intersecting_polygon.make_all_intersecting_polygon

```
def make_all_intersecting_polygon(
        polygon: shapely.Polygon, save_path: Path, overwrite=False, num_proc=30
):
    """
    Makes all of the basins intersecting the input polygon, then merges that data.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon used for masking the hydrobasins data.

    save_path : Path
        The path where the final merged and fixed dataframe will be saved in Parquet format.

    overwrite : bool, optional
        Whether to overwrite existing files in the temporary merged temporary directory. Default is False.

    num_proc : int, optional
        Number of processes to use for parallelization. Default is 30.
    """
```

and

wwvec.polygon_vectorization.make_all_intersecting_polygon.make_all_intersecting_hydrobasin_level_2_polygon
```
def make_all_intersecting_hydrobasin_level_2_polygon(hydrobasin_id: int, save_path: Path, overwrite=False, num_proc=30):
    """
    Makes all of the TDX-Hydro basins that intersect the input hydrobasin level 2 id, then merges that data.

    Parameters
    ----------
    hydrobasin_id : int
        The ID of the hydrobasin.
    save_path : Path
        The path to save the polygon data.
    overwrite : bool, optional
        Whether to overwrite existing data at the save path. Default is False.
    num_proc : int, optional
        The number of processes to use for parallel execution. Default is 30.

    Returns
    -------
    pandas.DataFrame
        The merged dataframe containing the intersecting hydrobasin level 2 polygons.
    """
```

# Related Repositories

 * [WaterNet](https://github.com/Better-Planet-Laboratory/WaterNet)
 * [WaterNet Training and Evaluation](https://github.com/Better-Planet-Laboratory/WaterNet_training_and_evaluation)
