"""
Author: Chihiro Matsumoto
Date: 03/05/2022
Description: GEOM90042 Assignment3
    - Working with Rasters and Digital Elevation Models
"""

import os
import sys
import math
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import show
from rasterio.enums import Resampling
import pyproj
from pyproj import CRS
import numpy as np
import reproducible_report.ProjectModule.PrettyTable as pt
import matplotlib.pyplot as plt


def summary_dem(filename):
    '''Import a raster file and store the data in a dictionary
    '''
    try:
        # Read a raster file
        with rasterio.open(filename, 'r') as dataset:

            summary_dict = {}

            # filename
            summary_dict['filename'] = filename

            # coordinate system
            src_crs = dataset.crs
            summary_dict['CRS'] = src_crs
            src_units = (
                CRS(src_crs.to_epsg())
                .coordinate_system.axis_list[0].unit_name)
            dst_crs = 32755
            dst_units = (
                CRS.from_epsg(dst_crs)
                .coordinate_system.axis_list[0].unit_name)

            if src_units == 'degree':
                # degree
                summary_dict['min_lon'] = {
                    'value': dataset.bounds.left, 'units': src_units}
                summary_dict['max_lon'] = {
                    'value': dataset.bounds.right, 'units': src_units}
                summary_dict['min_lat'] = {
                    'value': dataset.bounds.bottom, 'units': src_units}
                summary_dict['max_lat'] = {
                    'value': dataset.bounds.top, 'units': src_units}

                # convert to metre
                dst_bounds = rasterio.warp.transform_bounds(
                    src_crs, dst_crs,
                    dataset.bounds.left, dataset.bounds.bottom,
                    dataset.bounds.right, dataset.bounds.top)
                summary_dict['min_x'] = {
                    'value': dst_bounds[0], 'units': dst_units}
                summary_dict['max_x'] = {
                    'value': dst_bounds[2], 'units': dst_units}
                summary_dict['min_y'] = {
                    'value': dst_bounds[1], 'units': dst_units}
                summary_dict['max_y'] = {
                    'value': dst_bounds[3], 'units': dst_units}

            if src_units == 'metre':
                # metre
                summary_dict['min_x'] = {
                    'value': dataset.bounds.left, 'units': src_units}
                summary_dict['max_x'] = {
                    'value': dataset.bounds.right, 'units': src_units}
                summary_dict['min_y'] = {
                    'value': dataset.bounds.bottom, 'units': src_units}
                summary_dict['max_y'] = {
                    'value': dataset.bounds.top, 'units': src_units}

                # convert to degree
                dst_bounds = rasterio.warp.transform_bounds(
                    src_crs, 4326, dataset.bounds.left, dataset.bounds.bottom,
                    dataset.bounds.right, dataset.bounds.top)
                dst_units = 'degree'
                summary_dict['min_lon'] = {
                    'value': dst_bounds[0], 'units': dst_units}
                summary_dict['max_lon'] = {
                    'value': dst_bounds[2], 'units': dst_units}
                summary_dict['min_lat'] = {
                    'value': dst_bounds[1], 'units': dst_units}
                summary_dict['max_lat'] = {
                    'value': dst_bounds[3], 'units': dst_units}

            # other
            summary_dict['width'] = {
                'value': dataset.width, 'units': 'columns'}
            summary_dict['height'] = {'value': dataset.height, 'units': 'rows'}
            summary_dict['cell_size'] = {
                'value_x': dataset.res[0],
                'value_y': dataset.res[1], 'units': src_units}
            summary_dict['NoData'] = dataset.nodata

            bands = dataset.read(1, masked=True)
            summary_dict['min_value'] = {
                'value': bands.min(), 'units': 'metre'}
            summary_dict['max_value'] = {
                'value': bands.max(), 'units': 'metre'}

    except rasterio.errors.RasterioIOError:
        sys.exit('The dataset cannot be opened.')

    except pyproj.exceptions.CRSError:
        sys.exit('CRS error occurs.')

    except rasterio.errors.WarpOperationError:
        sys.exit('Warp operations fail.')

    except rasterio.errors.DatasetAttributeError:
        sys.exit('Dataset attributes are misused.')

    return summary_dict


def dict_to_nested_list(dict):
    '''This function converts datatype of summary information
    from a dictionary to a list.
    It returns a list of summary information.
    Each row has a set of parameter names and the values.
    '''

    # Create a string for each value
    filename = dict['filename'].split('/')
    val_file = filename[-1]

    val_crs = dict['CRS']
    val_minX = (
        '%.4f' % dict['min_x']['value'] + ' [' + dict['min_x']['units'] + '], '
        + '%.4f' % dict['min_lon']['value'] + ' ['
        + dict['min_lon']['units'] + ']')
    val_maxX = (
        '%.4f' % dict['max_x']['value'] + ' [' + dict['max_x']['units'] + '], '
        + '%.4f' % dict['max_lon']['value']
        + ' [' + dict['max_lon']['units'] + ']')
    val_minY = (
        '%.4f' % dict['min_y']['value'] + ' [' + dict['min_y']['units'] + '], '
        + '%.4f' % dict['min_lat']['value']
        + ' [' + dict['min_lat']['units'] + ']')
    val_maxY = (
        '%.4f' % dict['max_y']['value'] + ' [' + dict['max_y']['units'] + '], '
        + '%.4f' % dict['max_lat']['value']
        + ' [' + dict['max_lat']['units'] + ']')
    val_resolution = (
        str(dict['width']['value']) + ' [' + dict['width']['units'] + '], '
        + str(dict['height']['value']) + ' [' + dict['height']['units']
        + '], (' + '%.6f' % dict['cell_size']['value_x'] + ', '
        + '%.6f' % dict['cell_size']['value_y'] + ') ['
        + dict['cell_size']['units'] + ']')
    val_noData = dict['NoData']
    val_values = (
        '%.4f' % dict['min_value']['value'] + ' [' + dict['min_value']['units']
        + '], ' + '%.4f' % dict['max_value']['value']
        + ' [' + dict['max_value']['units'] + ']')

    # Compose a list
    list = [
        ['Filename', val_file],
        ['Coordinate system', val_crs],
        ['Min x, Min Lon', val_minX], ['Max x, Max Lon', val_maxX],
        ['Min y, Min Lon', val_minY], ['Max Lat, Max Lat', val_maxY],
        ['Width, Height, Cell size', val_resolution],
        ['NoData', val_noData],
        ['Min value, max value', val_values]]

    return list


def display_summary(summary_dict):
    '''This function outputs summary of the raster data stored in a distionary.
    '''

    nested_list = dict_to_nested_list(summary_dict)
    header = ['Parameter', 'Value']

    return pt.PrettyTable(nested_list, extra_header=header)


def plot_raster(filename):
    '''This function visualizes the raster data
    and plots the cell with th highest value
    '''

    # Call coordinates of cells with max height
    max_cells = find_highest_cell(filename)

    # Retrieve file name from entire file path
    filename_list = filename.split('/')
    title = filename_list[-1]

    try:
        with rasterio.open(filename, 'r') as dataset:

            bands = dataset.read(1, masked=True)
            max_val = bands.max()

            fig, ax = plt.subplots(1, figsize=(10, 12))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(title)
            show(dataset, ax=ax)

            # Highlight the highest cell
            for cell in max_cells:
                plt.plot(cell[0], cell[1], color='red', marker='s')
                str_coordinates = (
                    '%.4f' % max_val + ' m\n' + '[' + '%.4f' % cell[0] + ', '
                    + '%.4f' % cell[1] + ']')
                plt.annotate(
                    str_coordinates, xy=cell,
                    xytext=(cell[0]-0.2, cell[1]+0.04))

    except rasterio.errors.RasterioIOError:
        sys.exit('The dataset cannot be opened.')

    except rasterio.errors.DatasetAttributeError:
        sys.exit('Dataset attributes are misused.')


def find_highest_cell(filename):
    '''This function finds the cell with the highest value
    in the imported raster data
    '''
    try:
        with rasterio.open(filename, 'r') as dataset:

            bands = dataset.read(1)
            max_val = bands.max()
            cells = []  # coordinates of the cells with highest value

            for row in range(dataset.height):
                for col in range(dataset.width):
                    if bands[row, col] == max_val:
                        cells.append(dataset.xy(row, col, offset='center'))

    except rasterio.errors.RasterioIOError:
        sys.exit('The dataset cannot be opened.')

    except rasterio.errors.DatasetAttributeError:
        sys.exit('Dataset attributes are misused.')

    return cells


def project(src_file, dst_file):
    '''This function converts the dataset to a projected coordinate system
    and saves the projecte raster in a new file
    '''

    dst_crs = 'EPSG:32755'   # WGS84/UTM zone 55S

    try:
        with rasterio.open(src_file) as src:
            # Create the transform
            transform, width, height = (
                rasterio.warp.calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds))

            # Build the metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Create a new raster with reprojected data
            with rasterio.open(dst_file, 'w', **kwargs) as dst:
                for i in range(1, src.count+1):
                    rasterio.warp.reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=rasterio.warp.Resampling.bilinear
                    )

    except rasterio.errors.RasterioIOError:
        sys.exit('The dataset cannot be opened.')

    except rasterio.errors.WarpOperationError:
        sys.exit('Warp operations fail.')

    except rasterio.errors.DatasetAttributeError:
        sys.exit('Dataset attributes are misused.')

    except rasterio.errors.ResamplingAlgorithmError:
        sys.exit('Resampling algorithm is invalid.')


def resample_bilinear(src_file, dst_file, scale):
    '''This function resapmles the dataset using bilinear interpolation
    and saves the data in a new file
    '''

    try:
        upscale_factor = 1/scale
    except ZeroDivisionError:
        sys.exit('Scale variable can not be zero.')

    try:
        with rasterio.open(src_file) as src_dataset:

            # Resample data to target shape
            data = src_dataset.read(
                out_shape=(
                    src_dataset.count,
                    int(src_dataset.height * upscale_factor),
                    int(src_dataset.width * upscale_factor)
                ),
                resampling=Resampling.bilinear
            )

            # Scale image transform
            transform = src_dataset.transform * src_dataset.transform.scale(
                (src_dataset.width / data.shape[-1]),
                (src_dataset.height / data.shape[-2])
            )

            # Save resampling raster data
            kwargs = src_dataset.profile
            kwargs.update(
                crs=src_dataset.crs,
                transform=transform,
                width=int(src_dataset.width * upscale_factor),
                height=int(src_dataset.height * upscale_factor),
            )

            with rasterio.open(dst_file, 'w', **kwargs) as dst:
                dst.write(data)

    except rasterio.errors.RasterioIOError:
        sys.exit('The dataset cannot be opened.')

    except rasterio.errors.DatasetAttributeError:
        sys.exit('Dataset attributes are misused.')

    except rasterio.errors.ResamplingAlgorithmError:
        sys.exit('Resampling algorithm is invalid.')

    except rasterio.errors.TransformError:
        sys.exit('Transform arguments are invalid.')


def resample_nearest_neighbour(src_file, dst_file, scale):
    '''This function resapmles the dataset using nearest neighbour interpolation
    and saves the data in a new file
    '''

    upscale_factor = 1/scale

    try:
        with rasterio.open(src_file) as src_dataset:

            # Resample data to target shape
            data = src_dataset.read(
                out_shape=(
                    src_dataset.count,
                    int(src_dataset.height * upscale_factor),
                    int(src_dataset.width * upscale_factor)
                ),
                resampling=Resampling.nearest
            )

            # Scale image transform
            transform = src_dataset.transform * src_dataset.transform.scale(
                (src_dataset.width / data.shape[-1]),
                (src_dataset.height / data.shape[-2])
            )

            # Save resampling raster data
            kwargs = src_dataset.profile
            kwargs.update(
                crs=src_dataset.crs,
                transform=transform,
                width=int(src_dataset.width * upscale_factor),
                height=int(src_dataset.height * upscale_factor),
            )

            with rasterio.open(dst_file, 'w', **kwargs) as dst:
                dst.write(data)

    except rasterio.errors.RasterioIOError:
        sys.exit('The dataset cannot be opened.')

    except rasterio.errors.DatasetAttributeError:
        sys.exit('Dataset attributes are misused.')

    except rasterio.errors.ResamplingAlgorithmError:
        sys.exit('Resampling algorithm is invalid.')

    except rasterio.errors.TransformError:
        sys.exit('Transform arguments are invalid.')


def plot_histogram_height(filename):
    '''This function plots the histogram from the data in imported file
    '''

    try:
        with rasterio.open(filename, 'r') as dataset:

            # Retrieve title from the file path
            file_list = filename.split('/')
            title = file_list[-1]

            # Create 1D-list and remove noData
            height_list = []
            bands = dataset.read(1)
            for row in range(dataset.height):
                for col in range(dataset.width):
                    if bands[row][col] > 0:
                        height_list.append(bands[row][col])

            n, bins, patches = plt.hist(height_list, 50, color='blue')
            plt.xlabel('Height [metre]')
            plt.ylabel('Frequency')
            plt.title(title)

            plt.show()

    except rasterio.errors.RasterioIOError:
        sys.exit('The dataset cannot be opened.')

    except rasterio.errors.DatasetAttributeError:
        sys.exit('Dataset attributes are misused.')


def compute_slope_2FD(filename):
    '''This function computes slop using 2FD
    '''

    # 3*3 window
    # z9 z8 z7
    # z6 z5 z4
    # z3 z2 z1

    try:
        with rasterio.open(filename, 'r') as dataset:
            values = dataset.read(1)
            gx = dataset.res[0]      # cell size in x axis
            gy = dataset.res[1]      # cell size in y axis
            slope = [
                [0 for i in range(dataset.width)]
                for j in range(dataset.height)]

            for row in range(dataset.height):
                for col in range(dataset.width):
                    # Assign raster value into 3*3 window
                    z2 = values[row][col+1] if col+1 < dataset.width else 0
                    z4 = (
                        values[row+1][col+2]
                        if (row+1 < dataset.height and col+2 < dataset.width)
                        else 0)
                    z6 = values[row+1][col] if row+1 < dataset.height else 0
                    z8 = (
                        values[row+2][col+1]
                        if (row+2 < dataset.height and col+1 < dataset.width)
                        else 0)

                    # Compute fx and fy
                    fx = (z6-z4) / (2*gx)
                    fy = (z8-z2) / (2*gy)

                    # Compute slope
                    slope[row][col] = (
                        math.atan(math.sqrt(fx*fx + fy*fy))*(180/math.pi))

    except rasterio.errors.RasterioIOError:
        sys.exit('The dataset cannot be opened.')

    except rasterio.errors.DatasetAttributeError:
        sys.exit('Dataset attributes are misused.')

    return slope


def compute_slope_maximum_max(filename):
    '''This function computes slop using Maximum Max
    '''

    # 3*3 window
    # z9 z8 z7
    # z6 z5 z4
    # z3 z2 z1

    try:
        with rasterio.open(filename, 'r') as dataset:

            values = dataset.read(1)
            gx = dataset.res[0]      # cell size in x axis
            gy = dataset.res[1]      # cell size in y axis
            gz = math.sqrt(gx*gx + gy*gy)  # length of diadonal
            slope = [
                [0 for i in range(dataset.width)]
                for j in range(dataset.height)]

            for row in range(dataset.height):
                for col in range(dataset.width):
                    # Assign raster value into 3*3 window
                    z1 = values[row][col+2] if col+2 < dataset.width else 0
                    z2 = values[row][col+1] if col+1 < dataset.width else 0
                    z3 = values[row][col]
                    z4 = (
                        values[row+1][col+2]
                        if (row+1 < dataset.height and col+2 < dataset.width)
                        else 0)
                    z5 = (
                        values[row+1][col+1]
                        if (row+1 < dataset.height and col+1 < dataset.width)
                        else 0)
                    z6 = values[row+1][col] if row+1 < dataset.height else 0
                    z7 = (
                        values[row+2][col+2]
                        if (row+2 < dataset.height and col+2 < dataset.width)
                        else 0)
                    z8 = (
                        values[row+2][col+1]
                        if (row+2 < dataset.height and col+1 < dataset.width)
                        else 0)
                    z9 = values[row+2][col] if row+2 < dataset.height else 0

                    # Find the maximum tangent
                    max_tangent = max(
                        abs(z5-z1)/gz, abs(z5-z2)/gy,
                        abs(z5-z3)/gz, abs(z5-z9)/gz,
                        abs(z5-z7)/gz, abs(z5-z6)/gx,
                        abs(z5-z8)/gy, abs(z5-z4)/gx)

                    # Compute slop
                    slope[row][col] = math.atan(max_tangent)*(180/math.pi)

    except rasterio.errors.RasterioIOError:
        sys.exit('The dataset cannot be opened.')

    except rasterio.errors.DatasetAttributeError:
        sys.exit('Dataset attributes are misused.')

    return slope


def plot_histogram_slope(nested_list, filename, interpolation):
    '''This function plots a histogram from an imported slope list.
    '''

    # Convert 2D-list to 1D-list
    slope_list = list(np.concatenate(nested_list).flat)

    # Define title
    file_list = filename.split('/')
    title = file_list[-1] + ' (' + interpolation + ')'

    n, bins, patches = plt.hist(slope_list, 50, color='blue')
    plt.xlim(xmin=0, xmax=90)
    plt.xlabel('Slope [degree]')
    plt.ylabel('Frequency')
    plt.title(title)

    plt.show()


def compare_slope_distribution(data1, data2, data3, data4):
    '''This function compares distribution of the slope values.
    Returns summary table.
    Datasets in the arguments have different resampling method
                                    and slope calculation.
    '''

    header = ['Resampling', 'Slope', 'Mean', 'Median', 'Variance', 'SD']

    nested_list = [
        ['Bilinear', '2FD',
            np.mean(data1), np.median(data1), np.var(data1), np.std(data1)],
        ['Bilinear', 'Max',
            np.mean(data2), np.median(data2), np.var(data2), np.std(data2)],
        ['Nearest', '2FD',
            np.mean(data3), np.median(data3), np.var(data3), np.std(data3)],
        ['Nearest', 'Max',
            np.mean(data4), np.median(data4), np.var(data4), np.std(data4)]]

    for i in range(len(nested_list)):
        for j in range(len(header)):
            if isinstance(nested_list[i][j], float):
                nested_list[i][j] = '%.3f' % nested_list[i][j]

    return pt.PrettyTable(nested_list, extra_header=header)


def main():

    # Constant variable for files
    raster_file = 'CLIP.tif'
    filename = os.path.join(os.getcwd(), raster_file)

    # Task 1
    # Run functions in Task1 for unittest
    summary_dict = summary_dem(filename)
    table = display_summary(summary_dict)
    print(table)
    plot_raster(filename)

    print('Program complete')


if __name__ == '__main__':
    main()
