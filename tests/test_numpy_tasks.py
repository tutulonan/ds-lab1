from grader_contracts.numpy_tasks import (MatrixVectorBatchInput, BinarizeInput, MatrixInput, RandomMatrixInput, ChessInput, RectangleInput, EllipseInput, TimeSeriesInput, OneHotInput)
"""Проверки результатов, форм массивов и граничных случаев второй части."""
import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy_tasks import sum_prod, binarize, unique_rows, unique_columns, matrix_statistics, chess, draw_rectangle, draw_ellipse, analyze_time_series, one_hot

class NumpyTasksTests(unittest.TestCase):

    def test_sum_prod(self):
        a = np.array([[[1, 2], [3, 4]], [[2, 0], [0, 2]]])
        x = np.array([[[1], [2]], [[3], [4]]])
        assert_array_equal(sum_prod(MatrixVectorBatchInput(a, x)), [[11], [19]])
        assert_array_equal(sum_prod(MatrixVectorBatchInput([[[3]]], [[[4]]])), [[12]])
        assert_array_equal(sum_prod(MatrixVectorBatchInput(np.empty((0, 2, 2)), np.empty((0, 2, 1)))), np.zeros((2, 1)))
        rng = np.random.default_rng(42)
        a, x = (rng.normal(size=(5, 3, 3)), rng.normal(size=(5, 3, 1)))
        assert_allclose(sum_prod(MatrixVectorBatchInput(a, x)), sum((a[i] @ x[i] for i in range(5))))
        with self.assertRaises(ValueError):
            sum_prod(MatrixVectorBatchInput(np.zeros((2, 2, 3)), np.zeros((2, 3, 1))))

    def test_binarize(self):
        a = np.array([[-1, 0, 1], [2, 3, 4]])
        original = a.copy()
        result = binarize(BinarizeInput(a, 1))
        assert_array_equal(result, [[0, 0, 0], [1, 1, 1]])
        assert_array_equal(a, original)
        self.assertFalse(np.shares_memory(a, result))
        assert_array_equal(binarize(BinarizeInput(a, 10)), np.zeros_like(a))
        assert_array_equal(binarize(BinarizeInput(a, -2)), np.ones_like(a))
        self.assertEqual(binarize(BinarizeInput(np.empty((0, 3)), 0)).shape, (0, 3))

    def test_unique_rows(self):
        self.assertEqual(unique_rows(MatrixInput([[3, 1, 3], [2, 2, 2]])), [[1, 3], [2]])
        self.assertEqual(unique_rows(MatrixInput([[5]])), [[5]])
        self.assertEqual(unique_rows(MatrixInput(np.empty((0, 3)))), [])
        self.assertEqual(unique_rows(MatrixInput(np.empty((2, 0)))), [[], []])
        with self.assertRaises(ValueError):
            unique_rows(MatrixInput([1, 2]))

    def test_unique_columns(self):
        self.assertEqual(unique_columns(MatrixInput([[3, 1, 3], [2, 1, 2]])), [[2, 3], [1], [2, 3]])
        self.assertEqual(unique_columns(MatrixInput([[5]])), [[5]])
        self.assertEqual(unique_columns(MatrixInput(np.empty((0, 3)))), [[], [], []])
        self.assertEqual(unique_columns(MatrixInput(np.empty((2, 0)))), [])
        with self.assertRaises(ValueError):
            unique_columns(MatrixInput([1, 2]))

    def test_matrix_statistics(self):
        stats = matrix_statistics(RandomMatrixInput(2, 3, mean=5, std=2, seed=17))
        expected = np.random.default_rng(17).normal(5, 2, size=(2, 3))
        assert_array_equal(stats.matrix, expected)
        for data, means, variances in [(expected, stats.row_means, stats.row_variances), (expected.T, stats.column_means, stats.column_variances)]:
            manual_mean = [sum(row) / len(row) for row in data]
            manual_var = [sum(((v - mu) ** 2 for v in row)) / len(row) for row, mu in zip(data, manual_mean)]
            assert_allclose(means, manual_mean)
            assert_allclose(variances, manual_var)
        assert_array_equal(matrix_statistics(RandomMatrixInput(2, 3, 5, 2, 17)).matrix, stats.matrix)
        constant = matrix_statistics(RandomMatrixInput(1, 1, 7, 0, 1))
        assert_array_equal(constant.matrix, [[7]])
        assert_array_equal(constant.row_variances, [0])
        assert_array_equal(constant.column_variances, [0])
        for args in [(0, 3), (2, 0), (2, 3, 0, -1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                matrix_statistics(RandomMatrixInput(*args))

    def test_chess(self):
        assert_array_equal(chess(ChessInput(rows=3, columns=4, first=0, second=1)), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        assert_array_equal(chess(ChessInput(rows=1, columns=1, first=7, second=9)), [[7]])
        assert_array_equal(chess(ChessInput(rows=2, columns=3, first=0.5, second=-1.5)), [[0.5, -1.5, 0.5], [-1.5, 0.5, -1.5]])
        self.assertEqual(chess(ChessInput(rows=0, columns=3, first=0, second=1)).shape, (0, 3))
        assert_array_equal(chess(ChessInput(rows=2, columns=2, first=4, second=4)), np.full((2, 2), 4))
        with self.assertRaises(ValueError):
            chess(ChessInput(rows=-1, columns=3, first=0, second=1))

    def test_rectangle(self):
        image = draw_rectangle(RectangleInput(image_height=5, image_width=7, width=3, height=3, background_color=(255, 255, 255), shape_color=(10, 20, 30)))
        expected = np.full((5, 7, 3), 255, dtype=np.uint8)
        expected[1:4, 2:5] = [10, 20, 30]
        assert_array_equal(image, expected)
        self.assertEqual(image.dtype, np.uint8)
        even = draw_rectangle(RectangleInput(image_height=4, image_width=6, width=2, height=2, background_color=(255, 255, 255), shape_color=(0, 0, 0)))
        self.assertEqual(np.count_nonzero(np.all(even == 0, axis=2)), 4)
        assert_array_equal(draw_rectangle(RectangleInput(image_height=2, image_width=3, width=0, height=1, background_color=(255, 255, 255), shape_color=(0, 0, 0))), np.full((2, 3, 3), 255))
        assert_array_equal(draw_rectangle(RectangleInput(image_height=2, image_width=3, width=10, height=10, background_color=(255, 255, 255), shape_color=(0, 0, 0))), np.zeros((2, 3, 3)))
        with self.assertRaises(ValueError):
            draw_rectangle(RectangleInput(image_height=3, image_width=3, width=-1, height=2, background_color=(255, 255, 255), shape_color=(0, 0, 0)))
        with self.assertRaises(ValueError):
            draw_rectangle(RectangleInput(image_height=3, image_width=3, width=1, height=1, shape_color=(256, 0, 0), background_color=(255, 255, 255)))

    def test_ellipse(self):
        image = draw_ellipse(EllipseInput(image_height=5, image_width=5, semi_axis_x=1, semi_axis_y=1, background_color=(255, 255, 255), shape_color=(0, 0, 0)))
        expected = np.full((5, 5, 3), 255, dtype=np.uint8)
        for y, x in [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]:
            expected[y, x] = 0
        assert_array_equal(image, expected)
        self.assertEqual(image.dtype, np.uint8)
        for m, n, a, b in [(6, 8, 2, 1), (7, 9, 3, 2)]:
            with self.subTest(m=m, n=n):
                result = draw_ellipse(EllipseInput(image_height=m, image_width=n, semi_axis_x=a, semi_axis_y=b, background_color=(255, 255, 255), shape_color=(0, 0, 0)))
                assert_array_equal(result, result[::-1])
                assert_array_equal(result, result[:, ::-1])
        assert_array_equal(draw_ellipse(EllipseInput(image_height=1, image_width=1, semi_axis_x=1, semi_axis_y=1, background_color=(255, 255, 255), shape_color=(0, 0, 0))), np.zeros((1, 1, 3)))
        for a, b in [(0, 1), (1, -1), (float('nan'), 1)]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                draw_ellipse(EllipseInput(image_height=3, image_width=3, semi_axis_x=a, semi_axis_y=b, background_color=(255, 255, 255), shape_color=(0, 0, 0)))

    def test_time_series(self):
        stats = analyze_time_series(TimeSeriesInput([1, 3, 2, 4, 1], 3))
        self.assertAlmostEqual(stats.mean, 2.2)
        self.assertAlmostEqual(stats.variance, 1.36)
        self.assertAlmostEqual(stats.std, 1.36 ** 0.5)
        assert_array_equal(stats.local_maxima_indices, [1, 3])
        assert_array_equal(stats.local_minima_indices, [2])
        assert_allclose(stats.moving_average, [2, 3, 7 / 3])
        for series in [[5], [1, 2], [1, 2, 3], [3, 2, 1], [2, 2, 2], [1, 3, 3, 1]]:
            with self.subTest(series=series):
                result = analyze_time_series(TimeSeriesInput(series, 1))
                assert_array_equal(result.local_maxima_indices, [])
                assert_array_equal(result.local_minima_indices, [])
                assert_allclose(result.moving_average, series)
                assert_allclose(analyze_time_series(TimeSeriesInput(series, len(series))).moving_average, [np.mean(series)])
        for series, p in [([], 1), ([1, 2], 0), ([1, 2], 3), ([[1, 2]], 1)]:
            with self.subTest(series=series, p=p), self.assertRaises(ValueError):
                analyze_time_series(TimeSeriesInput(series, p))

    def test_one_hot(self):
        assert_array_equal(one_hot(OneHotInput([0, 2, 3, 0])), [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
        assert_array_equal(one_hot(OneHotInput([1, 0], 4)), [[0, 1, 0, 0], [1, 0, 0, 0]])
        assert_array_equal(one_hot(OneHotInput([0])), [[1]])
        self.assertEqual(one_hot(OneHotInput([])).shape, (0, 0))
        self.assertEqual(one_hot(OneHotInput([], 3)).shape, (0, 3))
        assert_array_equal(one_hot(OneHotInput([2, 2])).sum(axis=1), [1, 1])
        for labels, count in [([-1], None), ([1.5], None), ([2], 2), ([], -1), ([[0]], None)]:
            with self.subTest(labels=labels, count=count), self.assertRaises(ValueError):
                one_hot(OneHotInput(labels, count))
