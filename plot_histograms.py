from grader_contracts.numpy_tasks import (MatrixVectorBatchInput, BinarizeInput, MatrixInput, RandomMatrixInput, ChessInput, RectangleInput, EllipseInput, TimeSeriesInput, OneHotInput)
"""Гистограммы всех строк и столбцов"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy_tasks import matrix_statistics, draw_rectangle, draw_ellipse

def plot_histograms(matrix, output_dir='results'):
    """Каждая строка и каждый столбец представлены отдельной гистограммой."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError('Нужна непустая двумерная матрица')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for data, name, label in [(matrix, 'rows', 'Row'), (matrix.T, 'columns', 'Column')]:
        columns = min(3, len(data))
        rows = (len(data) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.8 * rows), squeeze=False)
        for i, values in enumerate(data):
            plt.sca(axes.flat[i])
            plt.hist(values, bins='auto', color='#4472c4', edgecolor='white')
            plt.title(f'{label} {i}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        for ax in axes.flat[len(data):]:
            ax.set_visible(False)
        fig.tight_layout()
        path = output_dir / f'histograms_{name}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths

def main():
    output_dir = Path(__file__).resolve().parent / 'results'
    stats = matrix_statistics(RandomMatrixInput(6, 8, mean=2, std=1.5, seed=42))
    for path in plot_histograms(stats.matrix, output_dir):
        print(f'Сохранён график: {path.name}')
    np.savez(output_dir / 'matrix_statistics.npz', matrix=stats.matrix, row_means=stats.row_means, column_means=stats.column_means, row_variances=stats.row_variances, column_variances=stats.column_variances)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(draw_rectangle(RectangleInput(image_height=120, image_width=180, width=100, height=60, background_color=(245, 247, 250), shape_color=(68, 114, 196))))
    axes[0].set_title('Rectangle')
    axes[1].imshow(draw_ellipse(EllipseInput(image_height=120, image_width=180, semi_axis_x=55, semi_axis_y=35, background_color=(245, 247, 250), shape_color=(220, 100, 80))))
    axes[1].set_title('Ellipse')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_dir / 'shapes.png', dpi=140)
    plt.close(fig)
if __name__ == '__main__':
    main()
