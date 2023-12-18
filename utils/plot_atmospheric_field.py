import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.patches as patches


def plot_atmospheric_field(
    x, save=False, save_path=None, projection=ccrs.Orthographic(-10, 62), show=False
):
    """
    Plots atmospheric field data on a global map using Cartopy.

    Args:
        x (numpy.ndarray): 2D array of atmospheric field data.
        save (bool): Whether to save the plot to a file. Default is False.
        save_path (str): Path to save the plot. Required if save is True.
        projection (cartopy.crs.Projection): Projection to use for the map. Default is Orthographic(-10, 62).
        show (bool): Whether to display the plot. Default is False.

    Returns:
        None

    Raises:
        ValueError: If save is True but save_path is not provided.

    Possible projections:
        - AlbersEqualArea
        - AzimuthalEquidistant
        - EquidistantConic
        - LambertConformal
        - LambertCylindrical
        - Mercator
        - Miller
        - Mollweide
        - NorthPolarStereo
        - OSGB
        - Orthographic
        - PlateCarree
        - Robinson
        - RotatedPole
        - Sinusoidal
        - SouthPolarStereo
        - Stereographic
        - TransverseMercator
        - UTM
        - WinkelTripel
    """
    # add 0 degree longitude to the end of the array
    x = np.hstack((x, x[:, 0].reshape(-1, 1)))
    # add average of first row to the beginning of the array
    x = np.vstack((np.mean(x[0]) * np.ones((1, x.shape[1])), x))

    # add average of last row to the end of the array
    x = np.vstack((x, np.mean(x[-1]) * np.ones((1, x.shape[1]))))
    phi = np.rad2deg(np.arange(0, 2 * np.pi + np.deg2rad(3), np.deg2rad(3)))
    theta = 88.5 - np.arange(0, 180, 3)
    theta = np.hstack((90, theta, -90))

    lons, lats = np.meshgrid(phi, theta)

    # Create a new figure and add a subplot with the specified projection
    fig = plt.figure(figsize=(5.9, 5))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # Create a filled contour plot of the data with specified levels and colormap
    filled_c = ax.contourf(
        lons,
        lats,
        x,
        np.arange(45, 60.5, 0.5) * 10**3,
        transform=ccrs.PlateCarree(),
        cmap="nipy_spectral",
    )

    # Create a contour plot of the data with the same levels as the filled contour plot
    line_c = ax.contour(
        lons,
        lats,
        x,
        levels=filled_c.levels,
        linewidths=0.4,
        colors=["black"],
        transform=ccrs.PlateCarree(),
    )

    # Create a contour plot of the data with a single level of 5500 gpm
    line_c2 = ax.contour(
        lons,
        lats,
        x,
        levels=[55 * 10**3],
        linewidths=2,
        colors=["black"],
        transform=ccrs.PlateCarree(),
    )

    # Add coastlines to the plot and set the global extent
    ax.coastlines()
    ax.set_global()

    # Add a colorbar to the plot and adjust the layout
    fig.colorbar(filled_c, ax=ax, fraction=0.045)
    fig.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save is True.")
        plt.savefig(save_path, dpi=300)

    # Show the plot
    if show:
        plt.show()


def plot_true_and_predicted_atomspheric_field(
    left_x,
    right_x,
    save=False,
    save_path=None,
    projection=ccrs.Orthographic(-10, 62),
    show=False,
    show_colorbar=False,
    left_title="True",
    right_title="Predicted",
    title="",
    height_title=0.95,
    cmap="nipy_spectral"
):
    """
    Plots two atmospheric field data on a global map using Cartopy.

    Args:
        x (numpy.ndarray): 2D array of atmospheric field data.
        y (numpy.ndarray): 2D array of atmospheric field data.
        save (bool): Whether to save the plot to a file. Default is False.
        save_path (str): Path to save the plot. Required if save is True.
        projection (cartopy.crs.Projection): Projection to use for the map. Default is Orthographic(-10, 62).
        show (bool): Whether to display the plot. Default is False.
        show_colorbar (bool): Whether to display the colorbar. Default is False.
        x_title (str): Title of the x plot. Default is "".
        y_title (str): Title of the y plot. Default is "".
        title (str): Title of the plot. Default is "".
        height_title (float): Height of the title. Default is 0.95.
        cmap (str): Colormap to use for the plot. Default is "nipy_spectral".

    Returns:
        None

    Raises:
        ValueError: If save is True but save_path is not provided.

    Possible projections:
        - AlbersEqualArea
        - AzimuthalEquidistant
        - EquidistantConic
        - LambertConformal
        - LambertCylindrical
        - Mercator
        - Miller
        - Mollweide
        - NorthPolarStereo
        - OSGB
        - Orthographic
        - PlateCarree
        - Robinson
        - RotatedPole
        - Sinusoidal
        - SouthPolarStereo
        - Stereographic
        - TransverseMercator
        - UTM
        - WinkelTripel
    """
    # add 0 degree longitude to the end of the array
    left_x = np.hstack((left_x, left_x[:, 0].reshape(-1, 1)))
    right_x = np.hstack((right_x, right_x[:, 0].reshape(-1, 1)))
    # add average of first row to the beginning of the array
    left_x = np.vstack((np.mean(left_x[0]) * np.ones((1, left_x.shape[1])), left_x))
    right_x = np.vstack((np.mean(right_x[0]) * np.ones((1, right_x.shape[1])), right_x))
    # add average of last row to the end of the array
    left_x = np.vstack((left_x, np.mean(left_x[-1]) * np.ones((1, left_x.shape[1]))))
    right_x = np.vstack((right_x, np.mean(right_x[-1]) * np.ones((1, right_x.shape[1]))))
    phi = np.rad2deg(np.arange(0, 2 * np.pi + np.deg2rad(3), np.deg2rad(3)))
    theta = 88.5 - np.arange(0, 180, 3)
    theta = np.hstack((90, theta, -90))

    lons, lats = np.meshgrid(phi, theta)

    # Create a new figure and add a subplot with the specified projection
    fig, axs = plt.subplots(
        1, 2, figsize=(11.8, 5), subplot_kw={"projection": projection} #,facecolor='black'}
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y = height_title)
    # Create a filled contour plot of the data with specified levels and colormap
    filled_c1 = axs[0].contourf(
        lons,
        lats,
        left_x,
        np.linspace(left_x.min()*0.98, left_x.max()*1.02, 30),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )
    filled_c2 = axs[1].contourf(
        lons,
        lats,
        right_x,
        np.linspace(left_x.min()*0.98, left_x.max()*1.02, 30),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )

    # Create a contour plot of the data with the same levels as the filled contour plot
    line_c1 = axs[0].contour(
        lons,
        lats,
        left_x,
        levels=filled_c1.levels,
        linewidths=0.3,
        colors=["black"],
        transform=ccrs.PlateCarree(),
    )
    line_c2 = axs[1].contour(
        lons,
        lats,
        right_x,
        levels=filled_c2.levels,
        linewidths=0.3,
        colors=["black"],
        transform=ccrs.PlateCarree(),
    )

    line_c3 = axs[0].contour(
        lons,
        lats,
        left_x,
        levels=[left_x.mean()],
        linewidths=2,
        colors=["black"],
        transform=ccrs.PlateCarree(),
    )
    line_c4 = axs[1].contour(
        lons,
        lats,
        right_x,
        levels=[right_x.mean()],
        linewidths=2,
        colors=["black"],
        transform=ccrs.PlateCarree(),
    )

    axs[0].set_title(left_title)
    axs[1].set_title(right_title)

    # Add coastlines to the plot and set the global extent
    axs[0].coastlines()
    axs[0].set_global()
    axs[1].coastlines()
    axs[1].set_global()

    # Add a colorbar to the plot and adjust the layout
    if show_colorbar:
        fig.colorbar(filled_c1, ax=axs[1], fraction=0.045)

    fig.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save is True.")
        plt.savefig(save_path, dpi=300)

    # Show the plot
    if show:
        plt.show()
