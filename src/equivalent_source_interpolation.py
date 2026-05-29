'''
Created on Feb 19, 2025

@author: Admin
'''
import ensaio
import pandas as pd
import pygmt
import pyproj
import verde as vd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import harmonica as hm

# Fetch the sample total-field magnetic anomaly data from Great Britain
fname = ensaio.fetch_britain_magnetic(version=1)
data = pd.read_csv(fname)

# Slice a smaller portion of the survey data to speed-up calculations for this
# example
region = [-5.5, -4.7, 57.8, 58.5]
inside = vd.inside((data.longitude, data.latitude), region)
data = data[inside]
print("Number of data points:", data.shape[0])
print("Mean height of observations:", data.height_m.mean())

# Since this is a small area, we'll project our data and use Cartesian
# coordinates
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
easting, northing = projection(data.longitude.values, data.latitude.values)
coordinates = (easting, northing, data.height_m)
xy_region = vd.get_region((easting, northing))

# Create the equivalent sources.
# We'll use block-averaged sources at given depth beneath the observation
# points. We will interpolate on a grid with a resolution of 500m, so we will
# use blocks of the same size. The damping parameter helps smooth the predicted
# data and ensure stability.
eqs = hm.EquivalentSources(depth=1000, damping=1, block_size=500)

# Fit the sources coefficients to the observed magnetic anomaly.
eqs.fit(coordinates, data.total_field_anomaly_nt)

# Evaluate the data fit by calculating an R² score against the observed data.
# This is a measure of how well the sources fit the data, NOT how good the
# interpolation will be.
print("R² score:", eqs.score(coordinates, data.total_field_anomaly_nt))

# Interpolate data on a regular grid with 500 m spacing. The interpolation
# requires the height of the grid points (upward coordinate). By passing in
# 1500 m, we're effectively upward-continuing the data (mean flight height is
# 500 m).

grid_coords = vd.grid_coordinates(region=xy_region, spacing=500, extra_coords=1500)

grid = eqs.grid(coordinates=grid_coords, data_names=["magnetic_anomaly"])

# The grid is a xarray.Dataset with values, coordinates, and metadata
print("\nGenerated grid:\n", grid)

# Set figure properties
w, e, s, n = xy_region
fig_height = 10
fig_width = fig_height * (e - w) / (n - s)
fig_ratio = (n - s) / (fig_height / 100)
fig_proj = f"x1:{fig_ratio}"

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), 
                             gridspec_kw={'wspace': 0.4})

# Calculate maximum absolute value for consistent color scaling
maxabs = vd.maxabs(data.total_field_anomaly_nt, grid.magnetic_anomaly.values) * 0.95

# Create colormap centered at 0 (similar to PyGMT's vik cmap)
cmap = plt.get_cmap('RdBu_r')  # RdBu_r is similar to vik
norm = TwoSlopeNorm(vmin=-maxabs, vcenter=0, vmax=maxabs)

# First plot - Observed magnetic anomaly data
scatter = ax1.scatter(easting, northing, 
                     c=data.total_field_anomaly_nt,
                     s=10,  # size similar to 0.1c
                     cmap=cmap,
                     norm=norm)
ax1.set_title("Observed magnetic anomaly data", fontsize=12)
ax1.set_xlabel('Easting')
ax1.set_ylabel('Northing')
# Set tick marks similar to xa10000, ya10000
ax1.set_xticks(np.arange(min(easting), max(easting), 10000))
ax1.set_yticks(np.arange(min(northing), max(northing), 10000))
# Add colorbar
plt.colorbar(scatter, ax=ax1, label='nT')

# Second plot - Gridded and upward-continued
# Convert grid to arrays for plotting
x = grid.easting.values
y = grid.northing.values
X, Y = np.meshgrid(x, y)
Z = grid.magnetic_anomaly.values

im = ax2.pcolormesh(X, Y, Z, 
                   cmap=cmap,
                   norm=norm)
ax2.set_title("Gridded and upward-continued", fontsize=12)
ax2.set_xlabel('Easting')
ax2.set_ylabel('Northing')
# Set tick marks
ax2.set_xticks(np.arange(min(x), max(x), 10000))
ax2.set_yticks(np.arange(min(y), max(y), 10000))
# Add colorbar
plt.colorbar(im, ax=ax2, label='nT')

# Adjust layout and display
plt.tight_layout()
plt.show()

# # Plot original magnetic anomaly and the gridded and upward-continued version
# fig = pygmt.Figure()

# title = "Observed magnetic anomaly data"

# # Make colormap of data
# # Get the 95 percentile of the maximum absolute value between the original and
# # gridded data so we can use the same color scale for both plots and have 0
# # centered at the white color.
# maxabs = vd.maxabs(data.total_field_anomaly_nt, grid.magnetic_anomaly.values) * 0.95
# pygmt.makecpt(
#     cmap="vik",
#     series=(-maxabs, maxabs),
#     background=True,
# )

# with pygmt.config(FONT_TITLE="12p"):
#     fig.plot(
#         projection=fig_proj,
#         region=xy_region,
#         frame=[f"WSne+t{title}", "xa10000", "ya10000"],
#         x=easting,
#         y=northing,
#         fill=data.total_field_anomaly_nt,
#         style="c0.1c",
#         cmap=True,
#     )

# fig.colorbar(cmap=True, frame=["a400f100", "x+lnT"])

# fig.shift_origin(xshift=fig_width + 1)

# title = "Gridded and upward-continued"

# with pygmt.config(FONT_TITLE="12p"):
#     fig.grdimage(
#         frame=[f"ESnw+t{title}", "xa10000", "ya10000"],
#         grid=grid.magnetic_anomaly,
#         cmap=True,
#     )

# fig.colorbar(cmap=True, frame=["a400f100", "x+lnT"])

# fig.show()