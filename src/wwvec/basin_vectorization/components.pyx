#cython: language_level=3
import cython as c
import numpy as np
cimport numpy as np
from libc.math cimport floor
np.import_array()
ctypedef np.int8_t npint8
ctypedef np.int16_t npint16
ctypedef np.int32_t npint
ctypedef np.float_t npfloat


cdef int get_start_index(int init_index):
    """
    Obtain the start index based on the initial index.

    Parameters
    ----------
    init_index : int
        The initial index.

    Returns
    -------
    int
        The start index.

    """
    cdef int to_return = 0
    if init_index > 0:
        to_return = init_index - 1
        return to_return
    else:
        return to_return


cdef int get_end_index(int init_index, int max_index):
    """    
    Parameters
    ----------
    init_index : int
        The starting index.
    max_index : int
        The maximum index.

    Returns
    -------
    int
        The ending index. If init_index is less than max_index - 1, returns 
        init_index + 2. Otherwise, returns max_index.
    """
    cdef int to_return = max_index
    if init_index < max_index - 1:
        to_return = init_index + 2
        return to_return
    else:
        return to_return


cdef find_component(
        np.ndarray[npint8, ndim=2] grid,
        np.ndarray[npint, ndim=2] component_grid,
        np.ndarray[npint16, ndim=2] elevation_grid,
        npint component,
        npint8 value,
        int row,
        int col,
        int num_rows,
        int num_cols,
):
    """
    Parameters
    ----------
    grid : np.ndarray[np.int8, ndim=2]
        The grid containing the binary values.
    component_grid : np.ndarray[np.int, ndim=2]
        The grid with the connected component labeled.
    elevation_grid : np.ndarray[np.int16, ndim=2]
        The grid containing elevation values.
    component : np.int
        The component label to assign to the connected component.
    value : np.int8
        The binary value to consider when labeling the connected component.
    row : int
        The starting row index of the connected component.
    col : int
        The starting column index of the connected component.
    num_rows : int
        The total number of rows in the grid.
    num_cols : int
        The total number of columns in the grid.

    Returns
    -------
    component_grid : np.ndarray[np.int, ndim=2]
        The grid with labeled connected components.
    component_count : int
        The total number of cells in the connected component.
    elevation_range : int
        The difference between the maximum and minimum elevation values within the connected component.

    """
    cdef int other_row
    cdef int other_col
    cdef npint8 grid_val
    cdef npint component_grid_val
    cdef int start_row
    cdef int start_col
    cdef int end_row
    cdef int end_col
    cdef npint16 min_elevation = 32767
    cdef npint16 max_elevation = -32766
    cdef npint16 elevation
    cdef list row_col = [(row, col)]
    cdef int current_index = 0
    cdef long component_count = 0
    cdef long current_len = 1
    component_grid[row, col] = component
    while current_len > component_count:
        row, col = row_col[current_index]
        elevation = elevation_grid[row, col]
        if elevation < min_elevation:
            min_elevation = elevation
        if elevation > max_elevation:
            max_elevation = elevation
        component_count += 1
        # Find the start and end row/ col in the grid for the 3x3 subgrid surrounding the current cell.
        start_row = get_start_index(row)
        start_col = get_start_index(col)
        end_row = get_end_index(row, num_rows)
        end_col = get_end_index(col, num_cols)
        for other_row in range(start_row, end_row):
            for other_col in range(start_col, end_col):
                # value=0 has 4 connectivity, value=1 has 8 connectivity
                if other_col == col or other_row == row or value == 1:
                    grid_val = grid[other_row, other_col]
                    component_grid_val = component_grid[other_row, other_col]
                    # if the grid values agree and the cell component is unlabeled
                    if grid_val == value and component_grid_val == 0:
                        current_len += 1
                        component_grid[other_row, other_col] = component
                        row_col.append((other_row, other_col))
        current_index += 1
        # If the grid is large, this list can become very large, so to save memory list entries should be removed.
        # Removing a lot of entries from the list all at once is faster than removing entries after each iteration.
        if current_index > 10000000:
            row_col = row_col[current_index:]
            current_index = 0
    return component_grid, component_count, max_elevation - min_elevation


cdef find_all_components(
        np.ndarray[npint8, ndim=2] grid,
        np.ndarray[npint, ndim=2] component_grid,
        np.ndarray[npint16, ndim=2] elevation_grid
):
    """
    Finds all the connected components in the given grid.

    Parameters
    ----------
    grid : numpy.ndarray[np.int8, ndim=2]
        The grid representing the binary values.
    component_grid : numpy.ndarray[np.int, ndim=2]
        The grid representing the connected components, where each cell contains the component number it belongs to.
    elevation_grid : numpy.ndarray[np.int16, ndim=2]
        The grid representing the elevation, where each cell contains the elevation value.

    Returns
    -------
    tuple
        A tuple containing the updated component grid, positive component count list, negative component count list,
        positive elevation difference list, and negative elevation difference list.

    """
    cdef int max_row = grid.shape[0]
    cdef int max_col = grid.shape[1]
    cdef npint current_positive_component = 1
    cdef list positive_component_count_list = [0]
    cdef npint current_negative_component= -1
    cdef list negative_component_count_list = [0]
    cdef list positive_elevation_difference = [0]
    cdef list negative_elevation_difference = [0]
    cdef long component_count
    cdef npint16 elevation_difference

    for row in range(max_row):
        for col in range(max_col):
            if component_grid[row, col] == 0 and grid[row, col] == 1:
                component_grid, component_count, elevation_difference = find_component(
                    grid, component_grid, elevation_grid, current_positive_component, 1, row, col, max_row, max_col
                )
                positive_component_count_list.append(component_count)
                positive_elevation_difference.append(elevation_difference)
                current_positive_component += 1
            elif component_grid[row, col] == 0 and grid[row, col] == 0:
                component_grid, component_count, elevation_difference = find_component(
                    grid, component_grid, elevation_grid, current_negative_component, 0, row, col, max_row, max_col
                )
                current_negative_component -= 1
                negative_component_count_list.append(component_count)
                negative_elevation_difference.append(elevation_difference)
    return (component_grid, positive_component_count_list, negative_component_count_list,
            positive_elevation_difference, negative_elevation_difference)


def find_raster_components(
        np.ndarray[npint8, ndim=2] grid, np.ndarray[npint16, ndim=2] elevation_grid
):
    """
    Parameters
    ----------
    grid : np.ndarray[np.int8, ndim=2]
        A 2-dimensional NumPy array.

    elevation_grid : np.ndarray[np.int16, ndim=2]
        A 2-dimensional NumPy array representing the elevation.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - component_grid : np.ndarray[np.int32, ndim=2]
            A 2-dimensional NumPy array with the component labels.
        - positive_component_count_list : list
            A list of the number of cells in each positive (water) component.
        - negative_component_count_list : list
            A list of the number of cells in each negative (land) component
        - positive_elevation_difference : int
            The positive elevation difference.
        - negative_elevation_difference : int
            The negative elevation difference.
    """
    component_grid = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.int32)
    (component_grid, positive_component_count_list, negative_component_count_list,
     positive_elevation_difference, negative_elevation_difference) = find_all_components(
        grid, component_grid, elevation_grid
    )
    return (component_grid, positive_component_count_list, negative_component_count_list,
            positive_elevation_difference, negative_elevation_difference)