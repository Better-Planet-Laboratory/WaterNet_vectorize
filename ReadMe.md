

# Overview

This repository is associated with the forthcoming paper "Pierson, Matthew., and Mehrabi, Zia. 2024. Mapping waterways worldwide with deep learning. arXiv.  	
https://doi.org/10.48550/arXiv.2412.00050". Please do cite this paper and attribute the work if using the model or work. The data outputs of this model (raster and vectorized versions) are also stored and available from the following source: Pierson, Matthew., Mehrabi. Zia. 2024, WaterNet Outputs and Code, https://doi.org/10.7910/DVN/YY2XMG, Harvard Dataverse.


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


# Outputs

The output dataframes have the following columns:

| Name              | Type       | Description                                                                                                                                                                                                                              |
|-------------------|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| stream_id         | int        | A unique id for the stream segment.                                                                                                                                                                                                      |
| target_stream_id  | int        | The stream_id of the target. IE the stream_id of the next adjacent downstream stream.                                                                                                                                                    |
| source_stream_ids | list(int)  | A list of the source stream_id. IE a list containing the ids of all adjacent streams flowing into this stream. Can contain more than 2 ids.                                                                                              |
| stream_order      | int        | The Strahler stream order of this stream segment.                                                                                                                                                                                        |
| from_tdx          | bool       | True if this stream segment appears in TDX-Hydro, otherwise False                                                                                                                                                                        |
| tdx_stream_id     | int        | Each stream in this dataset falls in a drainage basin in the TDX-Hydro dataset. This value corresponds to the streamID in the TDX-Hydo drainage basins dataset that this stream is in (or LINKNO in the TDX-Hydo stream network dataset) |
| geometry          | LineString | The Geometry of this segment.                                                                                                                                                                                                            |

The outputs in the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YY2XMG) have the following additional columns, but these are not automatically generated by this module.

| Name            | Type  | Description                                                                                                                                                  |
|-----------------|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| intersects_lake | bool  | True if the geometry intersects a lake in the [HydroLakes v 1.0 Dataset](https://www.hydrosheds.org/products/hydrolakes)                                     |
| length_m        | float | The length of the geometry computed using [pyproj.Geod.geometry_length](https://pyproj4.github.io/pyproj/stable/api/geod.html#pyproj.Geod.geometry_length)   |


# Known Issues

Our methodology currently has issues when vectorizing large bodies of water (wide rivers, lakes, swamps, etc). In these locations, 

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


# Citations

```yaml
@misc{pierson2024mappingwaterwaysworldwidedeep,
      title={Mapping waterways worldwide with deep learning}, 
      author={Matthew Pierson and Zia Mehrabi},
      year={2024},
      eprint={2412.00050},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00050}, 
}
```

```yaml
@data{DVN/YY2XMG_2024,
author = {Pierson, Matthew and Mehrabi, Zia},
publisher = {Harvard Dataverse},
title = {{WaterNet Outputs and Code}},
year = {2024},
version = {V1},
doi = {10.7910/DVN/YY2XMG},
url = {https://doi.org/10.7910/DVN/YY2XMG}
}
```
