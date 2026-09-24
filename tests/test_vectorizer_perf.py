import unittest
import numpy as np
from collections import defaultdict
from wwvec.basin_vectorization.vectorizer import Vectorizer
from wwvec.basin_vectorization.fast_thinning import zhang_suen_thinning


class TestVectorizerPerformance(unittest.TestCase):
    def test_make_count_8_grid_numerical_parity(self):
        """Tests that vectorized make_count_8_grid matches legacy 8-neighbor sum exactly."""
        np.random.seed(42)
        grid = (np.random.rand(200, 200) > 0.85).astype(np.int16)
        # Embedded zero border
        grid[0, :] = 0; grid[-1, :] = 0; grid[:, 0] = 0; grid[:, -1] = 0

        # Legacy implementation
        legacy_count = np.zeros(grid.shape, dtype=np.int16)
        rows, cols = np.where(grid == 1)
        for r, c in zip(rows, cols):
            legacy_count[r, c] = grid[r - 1:r + 2, c - 1:c + 2].sum() - 1

        # Vectorized implementation
        vec_count = Vectorizer.make_count_8_grid(grid)

        np.testing.assert_array_equal(legacy_count, vec_count)

    def test_iterative_investigate_row_col(self):
        """Tests that iterative investigate_row_col traces lines without recursion errors."""
        grid = np.zeros((50, 500), dtype=np.int16)
        # Create a line with endpoints and inner cells
        grid[25, 10:450] = 2
        grid[25, 10] = 1
        grid[25, 449] = 1

        # Mock vectorizer attributes
        class MockVectorizer:
            def __init__(self, g):
                self.count_grid = g.copy()
                self.init_count_grid = g.copy()
                self.connections_seen = defaultdict(set)
            def add_to_connections_seen(self, n1, n2):
                self.connections_seen[n1].add(n2)
                self.connections_seen[n2].add(n1)

        v = MockVectorizer(grid)
        cell_list = [(25, 10)]
        Vectorizer.investigate_row_col(v, 25, 10, cell_list)

        self.assertEqual(len(cell_list), 440)
        self.assertEqual(cell_list[0], (25, 10))
        self.assertEqual(cell_list[-1], (25, 449))

    def test_zhang_suen_thinning(self):
        """Tests that fast_thinning produces a 1-pixel wide skeleton."""
        # Create a 7-pixel wide horizontal bar
        mask = np.zeros((40, 100), dtype=np.uint8)
        mask[15:22, 20:80] = 1

        skeleton = zhang_suen_thinning(mask)
        # Verify skeleton is non-empty and 1-pixel wide vertically
        self.assertTrue(np.any(skeleton == 1))
        col_sums = skeleton[:, 25:75].sum(axis=0)
        self.assertTrue(np.all(col_sums == 1))


if __name__ == "__main__":
    unittest.main()
