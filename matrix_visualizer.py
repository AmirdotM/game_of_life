import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Sequence, Union, Tuple, Optional, Literal
import time # Used for pausing in the dynamic update example
import weakref # To hold weak references to figures

# Enable interactive mode for dynamic updates
plt.ion()

# Cache to store figure, axes, and image objects for dynamic updates
# Using weakref to allow figures to be garbage collected if the user closes them
_figure_cache = weakref.WeakValueDictionary()

def draw_matrix(
    matrix: np.ndarray,
    mode: Literal['binary', 'color'] = 'binary',
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    figsize: Tuple[int, int] = (6, 6),
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None
) -> None:
    """
    Render the given 2D matrix in a pop-up window using matplotlib,
    optionally tiling large matrices.

    Parameters
    ----------
    matrix : numpy.ndarray
        Numeric 2D numpy array to display.
    mode : 'binary' or 'color', default 'binary'
        - 'binary': map values <=0 to black and >0 to white per tile.
        - 'color': use a continuous colormap (default: 'viridis' unless `cmap` is set)
                   scaled across the entire matrix.
    grid_rows : int or None, optional
        Number of rows in the tile grid. If None, defaults to 1 (no tiling).
    grid_cols : int or None, optional
        Number of columns in the tile grid. If None, defaults to 1 (no tiling).
    figsize : tuple, default (6, 6)
        Figure size in inches. This is the total size for the grid of tiles.
    cmap : str or None, optional
        Colormap name for ‘color’ mode; ignored in ‘binary’ mode.
        Defaults to 'viridis' for 'color' mode if None.
    vmin, vmax : float or None, optional
        Data range for color scaling in 'color' mode. If None, computed from
        the *entire* matrix. Ignored in 'binary' mode.
    title : str or None, optional
        Optional title displayed above the entire figure.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise TypeError("Input 'matrix' must be a 2D numpy array.")

    if mode not in ['binary', 'color']:
        raise ValueError("Parameter 'mode' must be 'binary' or 'color'.")

    rows, cols = matrix.shape

    # Determine grid dimensions, default to 1x1 (no tiling)
    effective_grid_rows = grid_rows if grid_rows is not None else 1
    effective_grid_cols = grid_cols if grid_cols is not None else 1

    if not isinstance(effective_grid_rows, int) or effective_grid_rows <= 0:
        raise ValueError("'grid_rows' must be a positive integer.")
    if not isinstance(effective_grid_cols, int) or effective_grid_cols <= 0:
        raise ValueError("'grid_cols' must be a positive integer.")

    if rows % effective_grid_rows != 0 or cols % effective_grid_cols != 0:
         # For simplicity, let's require exact divisibility for now.
         # Padding or handling partial tiles would add complexity.
        raise ValueError(
            f"Matrix dimensions ({rows}x{cols}) must be divisible by "
            f"grid dimensions ({effective_grid_rows}x{effective_cols}). "
            "Padding or handling partial tiles is not supported in this version."
        )

    tile_rows = rows // effective_grid_rows
    tile_cols = cols // effective_grid_cols

    # --- Manage Figure and Axes for Dynamic Updates ---
    # Generate a unique key for this visualization configuration
    # Key includes title, grid size, and figure size to handle different layouts
    figure_key = (title, effective_grid_rows, effective_grid_cols, figsize)

    fig, axes, image_objects = _figure_cache.get(figure_key, (None, None, None))

    # If figure/axes don't exist or were closed, create them
    if fig is None or not plt.fignum_exists(fig.number):
        fig, axes = plt.subplots(effective_grid_rows, effective_grid_cols, figsize=figsize)
        # Ensure axes is a 2D array even for 1xN or Nx1 grids
        if effective_grid_rows == 1 and effective_grid_cols == 1:
            axes = np.array([[axes]])
        elif effective_grid_rows == 1:
            axes = axes[np.newaxis, :]
        elif effective_grid_cols == 1:
            axes = axes[:, np.newaxis]

        image_objects = np.empty((effective_grid_rows, effective_grid_cols), dtype=object)

        # Store in cache using a weak reference to the figure
        _figure_cache[figure_key] = (fig, axes, image_objects)

        # Initial setup of axes properties
        for i in range(effective_grid_rows):
            for j in range(effective_grid_cols):
                ax = axes[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                # Optional: Add tile title/label
                # ax.set_title(f'Tile ({i},{j})', fontsize=8)

    # --- Update Image Data and Colormaps ---
    if mode == 'binary':
        # Create a binary representation (True for >0, False for <=0)
        binary_matrix = matrix > 0
        # Use a discrete colormap: index 0 (False/0) -> black, index 1 (True/1) -> white
        binary_cmap = mcolors.ListedColormap(['black', 'white'])
        plot_vmin, plot_vmax = 0, 1 # Ensure mapping is correct

        for i in range(effective_grid_rows):
            for j in range(effective_grid_cols):
                ax = axes[i, j]
                tile_data = binary_matrix[i * tile_rows:(i + 1) * tile_rows, j * tile_cols:(j + 1) * tile_cols]

                if image_objects[i, j] is None:
                    im = ax.imshow(tile_data, cmap=binary_cmap, vmin=plot_vmin, vmax=plot_vmax, aspect='auto')
                    image_objects[i, j] = im
                else:
                    im = image_objects[i, j]
                    im.set_data(tile_data)
                    im.set_cmap(binary_cmap)
                    im.set_clim(plot_vmin, plot_vmax)

    elif mode == 'color':
        # Determine global vmin, vmax if not provided
        plot_vmin = vmin if vmin is not None else np.min(matrix)
        plot_vmax = vmax if vmax is not None else np.max(matrix)
        plot_cmap = cmap if cmap is not None else 'viridis'

        for i in range(effective_grid_rows):
            for j in range(effective_grid_cols):
                ax = axes[i, j]
                tile_data = matrix[i * tile_rows:(i + 1) * tile_rows, j * tile_cols:(j + 1) * tile_cols]

                if image_objects[i, j] is None:
                    im = ax.imshow(tile_data, cmap=plot_cmap, vmin=plot_vmin, vmax=plot_vmax, aspect='auto')
                    image_objects[i, j] = im
                else:
                    im = image_objects[i, j]
                    im.set_data(tile_data)
                    im.set_cmap(plot_cmap)
                    im.set_clim(plot_vmin, plot_vmax)

    # --- Final Figure Adjustments ---
    if title:
        fig.suptitle(title)
    else:
        # Clear previous title if any
        if fig._suptitle:
             fig._suptitle.set_text('')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle

    # Draw and flush events to update the plot in interactive mode
    fig.canvas.draw()
    fig.canvas.flush_events()

    # A small pause is necessary in interactive mode to allow the GUI event loop to process
    plt.pause(0.001)

    # Note: plt.show() is NOT called here because we are in interactive mode (plt.ion())
    # The window will remain open until explicitly closed by the user or script.

if __name__ == '__main__':
    # Example Usage (Demonstrates dynamic update with tiling)

    # --- Binary Mode Example ---
    print("Displaying random binary matrix (100x100) in a 2x2 grid...")
    matrix_size = (100, 100)
    grid_size = (2, 2)

    # Initial matrix
    data = np.random.rand(*matrix_size) - 0.5 # values around 0

    draw_matrix(data, mode='binary', grid_rows=grid_size[0], grid_cols=grid_size[1], title="Dynamic Binary Matrix (2x2 grid)")

    # Dynamic update loop
    print("Updating matrix dynamically...")
    for i in range(100):
        # Update data (e.g., simulate some change)
        data = np.random.rand(*matrix_size) - 0.5 + 0.1 * np.sin(i * 0.1)
        draw_matrix(data, mode='binary', grid_rows=grid_size[0], grid_cols=grid_size[1], title=f"Dynamic Binary Matrix (Update {i+1})")
        time.sleep(0.1) # Pause briefly

    print("Binary mode example finished.")
    time.sleep(2) # Keep the last frame visible for a moment

    # --- Color Mode Example ---
    print("\nDisplaying random color matrix (120x120) in a 3x3 grid...")
    matrix_size_color = (120, 120)
    grid_size_color = (3, 3)

    # Initial matrix
    data_color = np.random.rand(*matrix_size_color) * 100 # values 0-100

    draw_matrix(data_color, mode='color', grid_rows=grid_size_color[0], grid_cols=grid_size_color[1], title="Dynamic Color Matrix (3x3 grid)")

    # Dynamic update loop
    print("Updating color matrix dynamically...")
    for i in range(100):
        # Update data
        data_color = np.random.rand(*matrix_size_color) * 100 + 20 * np.sin(i * 0.05)
        draw_matrix(data_color, mode='color', grid_rows=grid_size_color[0], grid_cols=grid_size_color[1], title=f"Dynamic Color Matrix (Update {i+1})")
        time.sleep(0.1) # Pause briefly

    print("Color mode example finished.")

    # Keep plots open until user closes them manually
    print("\nExamples finished. Close the matplotlib windows to exit.")
    plt.ioff() # Turn off interactive mode
    # Note: Since windows were opened in interactive mode, they might persist.
    # A final plt.show() is not typically needed here, as ion() keeps them responsive.
    # If the script were to exit immediately, the windows might close.
    # Keeping the script running allows you to interact with the windows.
    # A simple way to keep it running until windows are closed is a loop:
    # while plt.get_fignums():
    #    plt.pause(0.1)
    # For this example, let's just finish and let the user close.