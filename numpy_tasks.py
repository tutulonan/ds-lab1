"""Заготовки задач на NumPy."""

import operator
import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices, vectors = (np.asarray(matrices), np.asarray(vectors))
    if matrices.ndim != 3 or vectors.ndim != 3 or matrices.shape[1] != matrices.shape[2] or (vectors.shape != (matrices.shape[0], matrices.shape[1], 1)):
        raise ValueError('Ожидаются формы (p, n, n) и (p, n, 1)')
    return (matrices @ vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError('Ожидается двумерная матрица')
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError('Ожидается двумерная матрица')
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    if rows <= 0 or columns <= 0 or std < 0:
        raise ValueError('Размеры должны быть положительными, std — неотрицательным')
    matrix = np.random.default_rng(seed).normal(mean, std, size=(rows, columns))
    return MatrixStatistics(matrix, matrix.mean(axis=1), matrix.mean(axis=0), matrix.var(axis=1), matrix.var(axis=0))


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    if rows < 0 or columns < 0:
        raise ValueError('Размеры не должны быть отрицательными')
    rows, columns = np.ogrid[:rows, :columns]
    return np.where((rows + columns) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    width, height = (operator.index(width), operator.index(height))
    if width < 0 or height < 0:
        raise ValueError('Размеры прямоугольника не должны быть отрицательными')
    image, color = _rgb_image(image_height, image_width, background_color, shape_color)
    left, top = ((image_width - width) // 2, (image_height - height) // 2)
    image[max(0, top):min(image_height, top + height), max(0, left):min(image_width, left + width)] = color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if not np.isfinite(semi_axis_x) or not np.isfinite(semi_axis_y) or semi_axis_x <= 0 or (semi_axis_y <= 0):
        raise ValueError('Полуоси должны быть положительными конечными числами')
    image, color = _rgb_image(image_height, image_width, background_color, shape_color)
    y, x = np.ogrid[:image_height, :image_width]
    x0, y0 = ((image_width - 1) / 2, (image_height - 1) / 2)
    mask = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    image[mask] = color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)
    window = operator.index(window)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('Ожидается непустой одномерный ряд')
    if not 1 <= window <= values.size:
        raise ValueError('Окно p должно лежать в диапазоне от 1 до длины ряда')
    middle = values[1:-1]
    maxima = np.flatnonzero((middle > values[:-2]) & (middle > values[2:])) + 1
    minima = np.flatnonzero((middle < values[:-2]) & (middle < values[2:])) + 1
    moving_average = np.convolve(values, np.ones(window) / window, mode='valid')
    return TimeSeriesStatistics(float(values.mean()), float(values.var()), float(values.std()), maxima, minima, moving_average)


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError('Метки должны быть одномерным вектором')
    if labels.size and (not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError('Метки должны быть неотрицательными целыми числами')
    required = int(labels.max()) + 1 if labels.size else 0
    if class_count is None:
        class_count = required
    else:
        class_count = operator.index(class_count)
    if class_count < required or class_count < 0:
        raise ValueError('class_count не покрывает все метки классов')
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels.astype(np.intp)] = 1
    return result


def _rgb_image(m, n, background_color, shape_color):
    if m <= 0 or n <= 0:
        raise ValueError('Размеры изображения должны быть положительными')
    colors = np.asarray([background_color, shape_color])
    if colors.shape != (2, 3) or not np.issubdtype(colors.dtype, np.integer) or np.any(colors < 0) or np.any(colors > 255):
        raise ValueError('Цвета должны состоять из трёх целых значений от 0 до 255')
    image = np.empty((m, n, 3), dtype=np.uint8)
    image[:] = colors[0]
    return (image, colors[1].astype(np.uint8))
