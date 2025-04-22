# Matrix Visualizer

A Python utility for visualizing 2D numeric matrices using Matplotlib, with support for binary or color representation, tiling large matrices into a grid, and dynamic updates in interactive mode.

## Features

* **Matrix Visualization:** Renders any 2D NumPy array as an image.
* **Display Modes:** Supports `binary` mode (values > 0 are white, <= 0 are black) and `color` mode (using a specified or default colormap).
* **Tiling:** Automatically divides large matrices into a grid of smaller plots for easier viewing.
* **Dynamic Updates:** Designed to work seamlessly with Matplotlib's interactive mode (`plt.ion()`) for updating the matrix display in real-time.
* **Configurable:** Allows setting figure size, colormap (for color mode), value range (vmin/vmax), and a main title.

## Prerequisites

* Python 3.6+
* `matplotlib`
* `numpy`

## Installation

1.  Make sure you have Python installed.
2.  Install the required libraries using pip:

    ```bash
    pip install matplotlib numpy
    ```
3.  Clone this repository or download the `draw_matrix.py` file.

## Usage

The main functionality is provided by the `draw_matrix()` function.

```python
draw_matrix(
    matrix: np.ndarray,
    mode: Literal['binary', 'color'] = 'binary',
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    figsize: Tuple[int, int] = (6, 6),
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None
) -> None
