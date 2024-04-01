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
    cdef int to_return = 0
    if init_index > 0:
        to_return = init_index - 1
        return to_return
    else:
        return to_return


cdef int get_end_index(int init_index, int max_index):
    cdef int to_return = max_index
    if init_index < max_index - 1:
        to_return = init_index + 2
        return to_return
    else:
        return to_return




cdef color_grid_values_cython(
        np.ndarray[npint8, ndim=2] grid,
        np.ndarray[npint, ndim=2] color_grid,
        np.ndarray[npint16, ndim=2] elevation_grid,
        npint color,
        npint8 value,
        int row,
        int col,
        int num_rows,
        int num_cols,
):
    cdef int other_row
    cdef int other_col
    cdef npint8 grid_val
    cdef npint color_grid_val
    cdef int start_row
    cdef int start_col
    cdef int end_row
    cdef int end_col
    cdef npint16 min_elevation = 32767
    cdef npint16 max_elevation = -32766
    cdef npint16 elevation
    cdef list row_col = [(row, col)]
    cdef int current_index = 0
    cdef long color_count = 0
    cdef long current_len = 1
    color_grid[row, col] = color
    while current_len > color_count:
        row, col = row_col[current_index]
        elevation = elevation_grid[row, col]
        if elevation < min_elevation:
            min_elevation = elevation
        if elevation > max_elevation:
            max_elevation = elevation
        # row, col = row_col[-1]
        # row_col = row_col[:-1]
        color_count += 1
        start_row = get_start_index(row)
        start_col = get_start_index(col)
        end_row = get_end_index(row, num_rows)
        end_col = get_end_index(col, num_cols)
        for other_row in range(start_row, end_row):
            for other_col in range(start_col, end_col):
                if other_col == col or other_row == row or value == 1:
                    grid_val = grid[other_row, other_col]
                    color_grid_val = color_grid[other_row, other_col]
                    if grid_val == value and color_grid_val == 0:
                        current_len += 1
                        color_grid[other_row, other_col] = color
                        row_col.append((other_row, other_col))
        current_index += 1
        if current_index > 10000000:
            current_index = 0
            row_col = row_col[10000000:]
    return color_grid, color_count, max_elevation - min_elevation


cdef color_grid_values(
        np.ndarray[npint8, ndim=2] grid, 
        np.ndarray[npint, ndim=2] color_grid,
        np.ndarray[npint16, ndim=2] elevation_grid,
        int row, 
        int col,
        int color,
        npint8 value):
    # print('in cgv')
    cdef int num_rows
    cdef int num_cols
    num_rows, num_cols = grid.shape[0], grid.shape[1]
    color_grid, color_count, elevation_difference = color_grid_values_cython(
        grid, color_grid, elevation_grid, color, value, row, col, num_rows, num_cols
    )
    # print('leaving cython')
    return color_grid, color_count, elevation_difference


cdef color_all(
        np.ndarray[npint8, ndim=2] grid,
        np.ndarray[npint, ndim=2] color,
        np.ndarray[npint16, ndim=2] elevation_grid
):
    cdef int max_row = grid.shape[0]
    cdef int max_col = grid.shape[1]
    cdef npint current_pos = 1
    cdef list pos_count_list = [0]
    cdef npint current_neg = -1
    cdef list neg_count_list = [0]
    cdef list pos_elevation_difference = [0]
    cdef list neg_elevation_difference = [0]
    cdef npint16 elevation_difference

    for row in range(max_row):
        for col in range(max_col):
            if color[row, col] == 0 and grid[row, col] == 1:
                color, count, elevation_difference = color_grid_values(grid, color, elevation_grid, row, col, current_pos, 1)
                pos_count_list.append(count)
                pos_elevation_difference.append(elevation_difference)
                current_pos += 1
            elif color[row, col] == 0 and grid[row, col] == 0:
                color, count, elevation_difference= color_grid_values(grid, color, elevation_grid, row, col, current_neg, 0)
                current_neg -= 1
                neg_count_list.append(count)
                neg_elevation_difference.append(elevation_difference)
    return color, pos_count_list, neg_count_list, pos_elevation_difference, neg_elevation_difference

def color_raster(np.ndarray[npint8, ndim=2] grid, np.ndarray[npint16, ndim=2] elevation_grid):
    color = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.int32)
    color, pos_list, neg_list, pos_elevation_difference, neg_elevation_difference = color_all(grid, color, elevation_grid)
    return color, pos_list, neg_list, pos_elevation_difference, neg_elevation_difference
