#cython: language_level=3
import cython as c
import numpy as np
cimport numpy as cnp
cnp.import_array()
ctypedef cnp.int8_t npint8
ctypedef cnp.int16_t npint16
ctypedef cnp.int32_t npint
ctypedef cnp.float_t npfloat



cdef class ElevationHeap:
    cdef npint row
    cdef npint col
    cdef npint index
    cdef npint16 index_elevation
    cdef npint parent_index
    cdef npint16 parent_elevation
    cdef npint child_0_index
    cdef npint16 child_0_elevation
    cdef npint child_1_index
    cdef npint16 child_1_elevation
    cdef npint row_1
    cdef npint col_1
    cdef npint row_2
    cdef npint col_2
    cdef npint index_1
    cdef npint index_2
    cdef npint next_index
    cdef npint16[:,:] elevation_grid
    cdef npint[:,:] array
    cdef npint[:,:] old_array
    cdef int current_length
    cdef int tail_index
    cdef int free_index
    cdef int down_index
    def __init__(self, cnp.ndarray[npint, ndim=2] array, cnp.ndarray[npint16, ndim=2] elevation_grid):
        self.elevation_grid = elevation_grid.copy()
        self.array = array.copy()
        self.current_length = array.shape[0]
        self.tail_index = self.current_length - 1
        self.free_index = 0
        self.down_index = -1


    cdef heapify_up(self, npint index):
        parent_index = (index + 1)//2 - 1
        index_elevation = self.get_index_elevation(index)
        parent_elevation = self.get_index_elevation(parent_index)
        if parent_elevation < index_elevation:
            self.swap_indices(index, parent_index)
            if parent_index > 0:
                self.heapify_up(parent_index)

    cdef reset_free_index(self):
        self.free_index = 0
        self.down_index = -1

    cdef npint get_elevation(self, npint row, npint col):
        if row >= 0 and col >= 0:
            return self.elevation_grid[row, col]
        return -32767

    cdef npint16 get_index_elevation(self, npint index):
        if index < self.current_length:
            row = self.array[index, 0]
            col = self.array[index, 1]
        else:
            row, col = -1, -1
        return self.get_elevation(row, col)

    cdef swap_indices(self, npint index_1, npint index_2):
        row_1 = self.array[index_1, 0]
        col_1 = self.array[index_1, 1]
        row_2 = self.array[index_2, 0]
        col_2 = self.array[index_2, 1]
        self.add_row_col_at_index(row_2, col_2, index_1)
        self.add_row_col_at_index(row_1, col_1, index_2)

    cdef add_row_col_at_index(self, npint row, npint col, npint index):
        self.array[index, 0] = row
        self.array[index, 1] = col

    cdef get_row_col_at_index(self, npint index):
        row = self.array[index, 0]
        col = self.array[index, 1]
        return row, col

    def heapify_down(self, npint index):
        self.down_index = index
        child_0_index = 2*(index + 1)
        child_1_index = 2*(index + 1) - 1
        index_elevation = self.get_index_elevation(index)
        child_0_elevation = self.get_index_elevation(child_0_index)
        child_1_elevation = self.get_index_elevation(child_1_index)
        if index_elevation < child_0_elevation or index_elevation < child_1_elevation:
            if child_0_elevation > child_1_elevation:
                self.swap_indices(index, child_0_index)
                next_index = child_0_index
            else:
                self.swap_indices(index, child_1_index)
                next_index = child_1_index
            self.down_index = next_index
            if 2*(next_index+1) < self.current_length:
                self.heapify_down(next_index)

    cdef remove_top(self):
        self.array[0, 0] = self.array[2, 0]
        self.array[0, 1] = self.array[2, 1]
        self.array[2, 0] = -1
        self.array[2, 1] = -1
        self.heapify_down(0)
        self.heapify_down(2)

    cdef add_element(self, npint row, npint col):
        if self.down_index < 0 or self.down_index > self.free_index:
            if self.get_row_col_at_index(self.free_index)[0] >= 0:
                print('issue adding element: ', self.free_index, self.down_index)
                raise Exception
            self.add_row_col_at_index(row, col, self.free_index)
            if self.free_index > 0:
                self.heapify_up(self.free_index)
            self.free_index += 1
            if self.free_index > self.tail_index:
                self.double_array()
        else:
            if self.get_row_col_at_index(self.down_index)[0] >= 0:
                print('issue adding element: ', self.free_index, self.down_index)
                raise Exception
            self.add_row_col_at_index(row, col, self.down_index)
            self.heapify_up(self.down_index)
            self.down_index = -1

    cdef double_array(self):
        old_array = self.array
        self.array = np.zeros((2*self.current_length, 2), dtype=np.int32) - 1
        self.array[:self.current_length] = old_array
        self.current_length *= 2
        self.tail_index = self.current_length - 1


cdef cnp.ndarray[npint8, ndim=2] embed_in_larger_grid8(cnp.ndarray[npint8, ndim=2] grid):
    cdef int num_rows = grid.shape[0]
    cdef int num_cols = grid.shape[1]
    cdef cnp.ndarray[npint8, ndim=2] larger_grid = np.zeros((num_rows + 2, num_cols + 2), dtype=grid.dtype)
    larger_grid[1:-1, 1:-1] = grid
    return larger_grid


cdef cnp.ndarray[npint16, ndim=2] embed_in_larger_grid16(cnp.ndarray[npint16, ndim=2] grid):
    cdef int num_rows = grid.shape[0]
    cdef int num_cols = grid.shape[1]
    cdef cnp.ndarray[npint16, ndim=2] larger_grid = np.zeros((num_rows + 2, num_cols + 2), dtype=grid.dtype)
    larger_grid[1:-1, 1:-1] = grid
    return larger_grid


cdef cnp.ndarray[npint, ndim=2] add_row_col_to_array(
        cnp.ndarray[npint, ndim=2] array, int current_index, npint row, npint col
):
    array[current_index, 0] = row
    array[current_index, 1] = col
    return array


cdef npint8 get_counts(cnp.ndarray[npint8, ndim=2] local_grid):
    """
    This function takes in a 3x3 grid and calculates the sum of the values at
    specific adjacent positions. The four adjacent positions considered are
    (1, 0), (0, 1), (1, 2), and (2, 1). The sum is then returned as an int8 value.
    """
    cdef npint8 dir_count = local_grid[1, 0] + local_grid[0, 1] + local_grid[1, 2] + local_grid[2, 1]
    return dir_count


cdef bint is_merge_type_1(cnp.ndarray[npint8, ndim=2] local_grid):
    """
    Merge type 1 are the rotations of:
    |X| | |
    | |X|X|
    | |X| |
    """
    cdef npint8 merge_type_11 = local_grid[1, 0] + local_grid[2, 1] + local_grid[0, 2]
    cdef npint8 merge_type_12 = local_grid[1, 0] + local_grid[0, 1] + local_grid[2, 2]
    cdef npint8 merge_type_13 = local_grid[0, 1] + local_grid[1, 2] + local_grid[2, 0]
    cdef npint8 merge_type_14 = local_grid[1, 2] + local_grid[2, 1] + local_grid[0, 0]
    if merge_type_11 == 3 or merge_type_12 == 3 or merge_type_13 == 3 or merge_type_14 == 3:
        return True
    return False


cdef bint is_merge_type_2(cnp.ndarray[npint8, ndim=2] local_grid):
    """
    Merge type 2 are the rotations of:
    | |X| |
    | |X| |
    |Z| |Z|
    Where at least 1 Z is a waterway
    """
    cdef npint8 sum1 = local_grid[2, 0] + local_grid[2, 2]
    cdef npint8 sum2 = local_grid[0, 2] + local_grid[2, 2]
    cdef npint8 sum3 = local_grid[0, 0] + local_grid[0, 2]
    cdef npint8 sum4 = local_grid[0, 0] + local_grid[2, 0]

    cdef bint merge_type_21 = local_grid[0, 1] == 1 and sum1 >= 1
    cdef bint merge_type_22 = local_grid[1, 0] == 1 and sum2 >= 1
    cdef bint merge_type_23 = local_grid[2, 1] == 1 and sum3 >= 1
    cdef bint merge_type_24 = local_grid[1, 2] == 1 and sum4 >= 1

    if merge_type_21 or merge_type_22 or merge_type_23 or merge_type_24:
        return True
    return False


cdef check_2_dir_start_points(cnp.ndarray[npint8, ndim=2] local_grid):
    cdef npint8 sum1 = local_grid[2, 0] + local_grid[2, 1] + local_grid[2, 2]
    cdef npint8 sum2 = local_grid[0, 2] + local_grid[1, 2] + local_grid[2, 2]
    cdef npint8 sum3 = local_grid[0, 0] + local_grid[0, 1] + local_grid[0, 2]
    cdef npint8 sum4 = local_grid[0, 0] + local_grid[1, 0] + local_grid[2, 0]

    cdef bint start_type_1 = sum1 <= 2 and local_grid[2, 1] == 1
    cdef bint start_type_2 = sum2 <= 2 and local_grid[1, 2] == 1
    cdef bint start_type_3 = sum3 <= 2 and local_grid[0, 1] == 1
    cdef bint start_type_4 = sum4 <= 2 and local_grid[1, 0] == 1

    if start_type_1 or start_type_2 or start_type_3 or start_type_4:
        return True
    return False


cdef check_1_dir_start_points(cnp.ndarray[npint8, ndim=2] local_grid):
    cdef npint8 sum1 = local_grid[2, 0] + local_grid[2, 1] + local_grid[2, 2]
    cdef npint8 sum2 = local_grid[0, 2] + local_grid[1, 2] + local_grid[2, 2]
    cdef npint8 sum3 = local_grid[0, 0] + local_grid[0, 1] + local_grid[0, 2]
    cdef npint8 sum4 = local_grid[0, 0] + local_grid[1, 0] + local_grid[2, 0]

    cdef bint start_type_1 = sum1 <= 2 and local_grid[2, 1] == 1
    cdef bint start_type_2 = sum2 <= 2 and local_grid[1, 2] == 1
    cdef bint start_type_3 = sum3 <= 2 and local_grid[0, 1] == 1
    cdef bint start_type_4 = sum4 <= 2 and local_grid[1, 0] == 1

    if start_type_1 or start_type_2 or start_type_3 or start_type_4:
        return True
    return False


cdef double_simple(cnp.ndarray[npint, ndim=2] simple):
    cdef int simple_size = simple.shape[0]
    cdef new_simple = np.zeros((simple_size*2, 2), dtype=np.int32)
    new_simple[:simple_size] = simple
    return new_simple


cdef label_points(
        list points_list,
        cnp.ndarray[npint8, ndim=2] grid,
        cnp.ndarray[npint8, ndim=2] interior_grid,
        bint init_label,
        ElevationHeap elevation_heap
):
    cdef npint row = 1
    cdef npint col
    cdef int index
    cdef int num_remaining = len(points_list)
    simple_size = num_remaining
    cdef cnp.ndarray[npint8, ndim=2] local_grid
    cdef npint8 dir_count
    for index in range(num_remaining):
        row, col = points_list[index]
        # 3x3 grid centered at row, col
        local_grid = grid[row - 1:row + 2, col - 1:col + 2]
        dir_count = get_counts(local_grid)
        if dir_count == 4:
            # If there is water in all 4 directions, (row, col) is currently an interior point
            interior_grid[row, col] = 1
        elif dir_count == 3:
            # If there is water in 1 of 3 
            elevation_heap.add_element(row, col)
            interior_grid[row, col] = 0
        elif dir_count == 2:
            interior_grid[row, col] = 0
            if local_grid[1, 0] + local_grid[1, 2] != 2 and local_grid[0, 1] + local_grid[2, 1] != 2:
                # If not straight across, check if it is a merge point
                if not is_merge_type_1(local_grid):
                    elevation_heap.add_element(row, col)
        elif dir_count == 1:
            interior_grid[row, col] = 0
            if local_grid[0, 0] + local_grid[0, 2] + local_grid[2, 0] + local_grid[2, 2] != 0:
                if init_label:
                    if not is_merge_type_2(local_grid) and not check_1_dir_start_points(local_grid):
                        elevation_heap.add_element(row, col)
                else:
                    if not is_merge_type_2(local_grid):
                        elevation_heap.add_element(row, col)
    return elevation_heap, interior_grid


cdef list check_for_interior_points(cnp.ndarray[npint8, ndim=2] interior_grid,
                                    npint row,
                                    npint col,
                                    list points_list
                                    ):
    cdef npint other_row, other_col
    for other_row in range(row-1, row+2):
        for other_col in range(col-1, col+2):
            if interior_grid[other_row, other_col] == 1:
                interior_grid[other_row, other_col] = 0
                points_list.append((other_row, other_col))
    return points_list


cdef evaluate_simple(list points_list,
        cnp.ndarray[npint8, ndim=2] grid,
                     ElevationHeap elevation_heap,
):
    cdef npint row = 1
    cdef npint col
    cdef int index
    cdef cnp.ndarray[npint8, ndim=2] local_grid
    cdef cnp.ndarray[npint8, ndim=2] interior_grid = grid.copy()
    cdef cnp.ndarray[npint, ndim=2] itergrid = np.zeros(
        shape=(interior_grid.shape[0], interior_grid.shape[1]), dtype=np.int32
    )
    interior_grid[interior_grid == 2] = 0
    grid[grid == 2] = 1
    cdef int simple_index = 0
    cdef npint8 dir_count
    # np.random.shuffle(points_list)
    elevation_heap, interior_grid = label_points(points_list, grid, interior_grid, False, elevation_heap)
    cdef int count = 0
    cdef int itercount = 0
    if len(elevation_heap.array) > 0:
        row, col = elevation_heap.get_row_col_at_index(0)
        while row >= 0:
            count = 0
            itercount += 1
            points_list = []
            while row >= 0:
                count += 1
                points_list = check_for_interior_points(interior_grid, row, col, points_list)
                local_grid = grid[row - 1: row + 2, col - 1: col + 2]
                dir_count = get_counts(local_grid)
                if dir_count == 3:
                    grid[row, col] = 0
                    itergrid[row, col] = itercount
                elif dir_count == 2:
                    if local_grid[1, 0] + local_grid[1, 2] != 2 and local_grid[0, 1] + local_grid[2, 1] != 2:
                        if not is_merge_type_1(local_grid):
                            grid[row, col] = 0
                            itergrid[row, col] = itercount
                elif dir_count == 1:
                    if local_grid[0, 0] + local_grid[0, 2] + local_grid[2, 0] + local_grid[2, 2] != 0:
                        if not is_merge_type_2(local_grid):
                            grid[row, col] = 0
                            itergrid[row, col] = itercount
                elevation_heap.remove_top()
                row, col = elevation_heap.get_row_col_at_index(0)
            elevation_heap.reset_free_index()
            elevation_heap, interior_grid = label_points(
                points_list, grid, interior_grid, False, elevation_heap
            )
            row, col = elevation_heap.get_row_col_at_index(0)
    return grid, itergrid


def thinner(cnp.ndarray[npint8, ndim=2] grid, cnp.ndarray[npint16, ndim=2] elevation_grid):
    grid = embed_in_larger_grid8(grid)
    elevation_grid = embed_in_larger_grid16(elevation_grid)
    itergrid = grid.copy()
    points_list = []
    cdef npint row
    cdef npint col
    for (row, col) in zip(*np.where(grid == 1)):
        local_grid = grid[row - 1: row + 2, col - 1: col + 2]
        dir_count = get_counts(local_grid)
        if dir_count < 4:
            points_list.append((row, col))
    if len(points_list) > 1:
        heap_array = np.zeros((max(len(points_list), 3), 2), dtype=np.int32) - 1
        heap = ElevationHeap(array=heap_array, elevation_grid=elevation_grid)
        grid, itergrid = evaluate_simple(points_list, grid, heap)
    return grid[1:-1, 1:-1], itergrid[1:-1, 1:-1]

