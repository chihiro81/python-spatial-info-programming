"""
Author: Benjamin Lee (1259510), Chihiro Matsumoto (1147341)
Date: 08/06/2022
Description: GEOM90042 Assignment4

This is a module caontaining functions for data processing.
"""

# Import libraries
import os
import geopandas as gpd
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import fiona


# Task 1
def import_shp(file_shp):
    """This function reads a shapefile containing the accidents data in Victoria

    argument:
    file_shp (yyyy.shp)
    """

    # Define filepath
    data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    if (file_shp[0] == '2'):
        year = file_shp.split('.')[0]
        filepath = os.path.join(
            data_path, "data", "VicRoadsAccidents", year, file_shp)
    elif (file_shp[0] == 'L'):
        filepath = os.path.join(
            data_path, "data", "RegionsLGA_2017", file_shp)
    elif (file_shp[0] == 'S'):
        filepath = os.path.join(
            data_path, "data", "RegionsSA2_2016", file_shp)
    else:
        print(IOError)

    # Read the file
    try:
        data = gpd.read_file(filepath)
    except IOError as e:
        print(e)

    return data


def combine_dataframes(data_list):
    """This function combines all dataframes in a imported list."""

    return pd.concat(data_list)


def count_frequency(list):
    """This function counts the frequency of elements in a list
    using Collections modeule.
    Return the first and second common types and the percentages
    """

    freq = collections.Counter(list)
    return freq.most_common()


def calculate_accidents_by_vehicle_type(dataframe):
    """This function calculates the number of accidents by vehicle type
    The vehicle types include:
        - pedal vehicle
        - heavy vehicle
        - passenger vehicle
        - motorcycle
        - public transport vehicle
    Return a dictionary

    Argument: dataframe containing crash data for a specific year
    """

    col_list = [
        'BICYCLIST', 'HEAVYVEHIC', 'PASSENGERV', 'MOTORCYCLE', 'PUBLICVEHI']

    # Sum all columns
    sum = dataframe[col_list].sum(axis=0)

    return sum


def show_accident_type_stats(
            data2013, data2014, data2015, data2016, data2017, data2018):
    """This function shows the number of accidents by vehicle type and by year.
    Return a dataframe sortedy by the number in 2013
    """

    # Calculate accidents number by year
    # Combine all dataframes
    data = {
        '2013': calculate_accidents_by_vehicle_type(data2013).values,
        '2014': calculate_accidents_by_vehicle_type(data2014).values,
        '2015': calculate_accidents_by_vehicle_type(data2015).values,
        '2016': calculate_accidents_by_vehicle_type(data2016).values,
        '2017': calculate_accidents_by_vehicle_type(data2017).values,
        '2018': calculate_accidents_by_vehicle_type(data2018).values
    }

    # Define index of the new dataframe
    index_labels = [
        'Pedal Vehicle', 'Heavy Vehicle', 'Passenger Vehicle',
        'Motorcycle', 'Public Transport']
    df = pd.DataFrame(data, index=index_labels)

    # Sort by the number in 2013
    sorted_df = df.sort_values('2013', axis=0, ascending=False)

    return sorted_df


def get_accident_count(data, column):
    """This function returns a count of the number of instances based on
    the assigned column to group by in the dataset
    """

    return data[column].value_counts()


def show_accident_LGA_count_stats(
            data2013, data2014, data2015, data2016,
            data2017, data2018, number_of_LGAs=10):
    """This function shows the number of accidents by LGA per year as well
    as the differences and percent change between subsequent years.
    Returns a limited dataframe sorted by the data in 2013.
    """

    column = 'LGA_NAME'

    # Calculate accidents per LGA by year
    count2013 = get_accident_count(data2013, column)
    count2014 = get_accident_count(data2014, column)
    count2015 = get_accident_count(data2015, column)
    count2016 = get_accident_count(data2016, column)
    count2017 = get_accident_count(data2017, column)
    count2018 = get_accident_count(data2018, column)

    # Calculate secondary column data
    diff2014 = count2014 - count2013
    change2014 = diff2014/count2013
    diff2015 = count2015 - count2014
    change2015 = diff2015/count2014
    diff2016 = count2016 - count2015
    change2016 = diff2016/count2015
    diff2017 = count2017 - count2016
    change2017 = diff2017/count2016
    diff2018 = count2018 - count2017
    change2018 = diff2018/count2017

    # Compile the dataframes into a dictionary
    data = {
        tuple(['2013', 'No.']): count2013,
        tuple(['2014', 'No.']): count2014,
        tuple(['2014', 'Diff.']): diff2014,
        tuple(['2014', 'Change']): change2014,
        tuple(['2015', 'No.']): count2015,
        tuple(['2015', 'Diff.']): diff2015,
        tuple(['2015', 'Change']): change2015,
        tuple(['2016', 'No.']): count2016,
        tuple(['2016', 'Diff.']): diff2016,
        tuple(['2016', 'Change']): change2016,
        tuple(['2017', 'No.']): count2017,
        tuple(['2017', 'Diff.']): diff2017,
        tuple(['2017', 'Change']): change2017,
        tuple(['2018', 'No.']): count2018,
        tuple(['2018', 'Diff.']): diff2018,
        tuple(['2018', 'Change']): change2018
    }

    # Define the Multi-Header for the dataframe
    header1 = ['2013', '2014', '2014', '2014', '2015', '2015', '2015',
               '2016', '2016', '2016', '2017', '2017', '2017',
               '2018', '2018', '2018']
    header2 = ['No.', 'No.',  'Diff.', 'Change', 'No.', 'Diff.', 'Change',
               'No.', 'Diff.', 'Change', 'No.', 'Diff.', 'Change',
               'No.', 'Diff.', 'Change']
    head_tuple = list(zip(header1, header2))

    mux = pd.MultiIndex.from_tuples(head_tuple)

    # Iterative process to create a dataframe,
    # even if some frames are of differing length.
    # This is specifically relevant
    # because of UNINCORPORATED VIC data.
    df = pd.DataFrame(
            dict([(k, pd.Series(v)) for k, v in data.items()]), columns=mux)
    df.index.name = tuple(['', 'LGA'])

    # Sort by the number in 2013
    if (number_of_LGAs > 0):
        sorted_df = df.sort_values(('2013', 'No.'),
                                   axis=0, ascending=False)[:number_of_LGAs]
    elif (number_of_LGAs < -1):
        sorted_df = df.sort_values(('2013', 'No.'),
                                   axis=0, ascending=False)[:number_of_LGAs+1]
    else:
        sorted_df = df.sort_values(('2013', 'No.'),
                                   axis=0, ascending=False)

    # The LGA is currently the index,
    # so the dataframe returned has that adjusted.
    return sorted_df.reset_index()


def show_accident_severity_stats(
            data2013, data2014, data2015, data2016, data2017, data2018):
    """This function returns a dataframe of the count of each level of severity
    """

    # Title to be used for the dataframe
    t = 'Yearly change of the total number of accidents between 2013 and 2018'

    # Get aggregated severity data per year
    # Combine all dataframes
    column = 'SEVERITY'
    data = {
        '2013': get_accident_count(data2013, column),
        '2014': get_accident_count(data2014, column),
        '2015': get_accident_count(data2015, column),
        '2016': get_accident_count(data2016, column),
        '2017': get_accident_count(data2017, column),
        '2018': get_accident_count(data2018, column)
    }

    # Define index of the new dataframe
    index_labels = ['Other injury accident',
                    'Serious injury accident',
                    'Fatal accident',
                    'Non injury accident']
    df = pd.DataFrame(data, index=index_labels).T
    df.fillna(0)

    ax = df.plot(
        kind='line',
        grid=True)
    ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=7)

    plt.ylabel('Number of accidents', fontsize=8)
    plt.title(t, fontsize=8)
    plt.tick_params(axis='x', labelsize=7)
    plt.tick_params(axis='y', labelsize=7)


def add_labels(x, y, width):
    """This function adds labels on the chart
    """

    for i in range(len(x)):
        plt.text(i+width, y[i]+20, y[i], ha='center', size='x-small')


def plot_bars_accidents(df1, df2):
    """This function show a bar shart comparing accident numbers between two years.

    Arguments:
    df1, df2: dataframes
    """

    # Calculate accident numbers by day of the week
    list1 = calculate_accidents_by_dayOfWeek(df1)
    year1 = 2013
    list2 = calculate_accidents_by_dayOfWeek(df2)
    year2 = 2018
    days_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Set variables for plotting  a bar chart
    X = np.arange(len(list1))
    width = 0.25

    # Draw a bar chart
    plt.figure(figsize=(5, 4))
    plt.bar(X, list1, width, label=year1)
    add_labels(X, list1, 0)
    plt.bar(X + width, list2, width, label=year2)
    add_labels(X, list2, width)

    # Set chart properties
    plt.tick_params(axis='x', labelsize=7)
    plt.tick_params(axis='y', labelsize=7)
    plt.ylabel('The Number of Accidents', fontsize=8)
    plt.title(
        'Accident Numbers in %s and %s by days of the week' % (year1, year2),
        fontsize=8)
    plt.xticks(X+width/2, days_list)
    plt.legend(fontsize=6)

    plt.show()


def calculate_accidents_by_dayOfWeek(dataframe):
    """This function groups items by day of the week.
    Returns a new list of accidents numbbers sorted by day of the week.
    (Mon, Tues, wed, Thus, Fri, Sat, Sun)
    Excluding "None"
    """

    # Extract the column containing days of the week
    dist = count_frequency(dataframe["DAY_OF_WEE"])

    # Create a list sorted by the defined order
    # Exclude "None" value
    acc_list = []
    days_list = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
        'Saturday', 'Sunday']
    for i in range(len(days_list)):
        for j in range(len(dist)):
            if days_list[i] == dist[j][0]:
                acc_list.append(dist[j][1])

    return acc_list


def adjust_lga_names(lga_data):
    """This function adds a column to the LGA data to standardize
    the LGA names between the accident data and the LGA data.
    """
    lga_data['LGA_NAME_adj'] = [lga.split(' (')[0].upper()
                                for lga in lga_data.LGA_NAME17.values]

    return


def adjust_acc_lga_names(accident_data, lga_data,
                         LGA_NAME_column='LGA_NAME', inplace=False):
    """This function adds a column to the accident data to standardize
    the LGA names between the accident data and the LGA data.
    """

    # K:V Pairs for some of the discrepancies in nomenclature.
    lga_correct = {
                'BENDIGO': 'GREATER BENDIGO',
                'DANDENONG': 'GREATER DANDENONG',
                'GEELONG': 'GREATER GEELONG',
                'SHEPPARTON': 'GREATER SHEPPARTON',
                'COLAC OTWAY': 'COLAC-OTWAY'}

    LGA_NAME_adj = []
    for accident in accident_data[LGA_NAME_column]:
        if (accident not in lga_data.LGA_NAME_adj.values):
            if (accident in lga_correct):
                accident = lga_correct.get(accident)
            else:
                # According to the metadata, regions in the accident data
                # encapsulated in parentheses should be treated as being
                # part of the UINCORPORATED VIC data.
                accident = 'UNINCORPORATED VIC'
        LGA_NAME_adj.append(accident)

    if inplace:
        accident_data[LGA_NAME_column] = LGA_NAME_adj
    else:
        accident_data['LGA_NAME_adj'] = LGA_NAME_adj

    return


def most_common_accident_type(accident_data, column='ACCIDENT_1'):
    """This function returns the most common category of accidents
    in a dataset. By default, this is identified in the ACCIDENT_1 column"""

    freq_table = accident_data.groupby(column).size()\
        .sort_values(ascending=False).reset_index(name='FREQUENCY')

    return freq_table.loc[freq_table.groupby(column)
                          .size().sort_values(ascending=False)[0]][column]


def create_choropleth_df(accident_data, lga_data,
                         common_type, column='ACCIDENT_1'):
    """This function returns a dataframe based on the most common type
    of accident in a chosen column per LGA. Plotting the dataframe according
    to the column will produce the relevant choropleth visualization.
    """

    adjust_lga_names(lga_data)
    adjust_acc_lga_names(accident_data, lga_data)

    freq_table = accident_data.groupby(['LGA_NAME_adj', column])\
        .size().reset_index(name='FREQUENCY')
    total_table = accident_data.groupby('LGA_NAME_adj').size()\
        .reset_index(name='TOTAL')
    common_table = freq_table.loc[freq_table[column] == common_type]\
        .reset_index().drop(columns=['index', column])
    common_table = common_table.set_index('LGA_NAME_adj')\
        .join(total_table.set_index('LGA_NAME_adj'))
    common_table['FRACTION'] = common_table['FREQUENCY']/common_table['TOTAL']

    return lga_data.set_index('LGA_NAME_adj').join(common_table)


def create_choropleths(dataset1, dataset2, lga_data, common_type,
                       column='ACCIDENT_1',
                       title='Fraction of Common Accident Types per LGA'):

    df1 = create_choropleth_df(accident_data=dataset1,
                               lga_data=lga_data,
                               common_type=common_type,
                               column=column)
    df2 = create_choropleth_df(accident_data=dataset2,
                               lga_data=lga_data,
                               common_type=common_type,
                               column=column)

    # Create normalized color gradient
    vmin = min(df1.FRACTION.min(), df2.FRACTION.min())
    vmax = max(df1.FRACTION.max(), df2.FRACTION.max())
    vcenter = (vmax + vmin)/2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = 'Reds'
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    fig.suptitle(title, fontsize=9)
    ax1.set_title("2013", fontsize=7)
    ax1.tick_params(axis='both', labelsize=7)
    ax2.set_title("2018", fontsize=7)
    ax2.tick_params(axis='both', labelsize=7)

    df1.plot(column='FRACTION', ax=ax1, cmap=cmap, norm=norm, legend=False)
    df2.plot(column='FRACTION', ax=ax2, cmap=cmap, norm=norm, legend=False)

    cbar = fig.colorbar(cbar, ax=(ax1, ax2), location='bottom', shrink=0.5)
    cbar.ax.tick_params(labelsize=7)

    return


def get_single_lga_data(accident_data, lga_name, LGA_NAME_column='LGA_NAME_A'):

    return accident_data.loc[accident_data[LGA_NAME_column] == lga_name]


def get_state_geom(lga_data):
    """Returns a geometry based on various LGA geometries in a state"""

    state = lga_data[lga_data.geometry.notnull()]

    return state.dissolve().geometry[0]


def add_state_geom(df, lga_data, transpose=True):
    """Returns a GeoDataFrame with a uniform state geometry"""

    state_geom = get_state_geom(lga_data)

    if (transpose):
        geom = [state_geom]*len(df.T)
        return gpd.GeoDataFrame(df.T.assign(geometry=geom))

    else:
        geom = [state_geom]*len(df)
        return gpd.GeoDataFrame(df.assign(geometry=geom))


def create_accident_per_LGA_gdf(data2013, data2014, data2015, data2016,
                                data2017, data2018, lga_data,
                                number_of_LGAs=0):
    """Returns a GeoDataFrame detailing accidents per LGA from 2013-2018"""

    geo_acc_LGA = show_accident_LGA_count_stats(data2013, data2014,
                                                data2015, data2016, data2017,
                                                data2018, number_of_LGAs)
    geo_acc_LGA.columns = [c[0] + '_' + c[1] for c in geo_acc_LGA.columns]
    adjust_acc_lga_names(geo_acc_LGA, lga_data,
                         LGA_NAME_column='_LGA', inplace=False)

    return gpd.GeoDataFrame(geo_acc_LGA.merge(lga_data, on='LGA_NAME_adj'))


# Task2-1
def integrate_vehicles(df):
    """This function creates a string including all involved vehicle types.
    The types are ordered alphabetically.
    """

    type = ''
    # Order alphabetically
    if df["BICYCLIST"] > 0:
        type = type + 'Bicycle '
    if df["HEAVYVEHIC"] > 0:
        type = type + 'HeavyVehicle '
    if df["MOTORCYCLE"] > 0:
        type = type + 'Motorcycle '
    if df["PASSENGERV"] > 0:
        type = type + 'PassengerVehicle '
    if df["PUBLICVEHI"] > 0:
        type = type + 'PublicVehicle'

    return type


def create_accident_location(dataframe):
    # Extract accident data with at least 4 people involved
    tmp = dataframe.loc[dataframe["TOTAL_PERS"] >= 3]

    tmp.loc[:, "VehicleType"] = tmp.apply(integrate_vehicles, axis=1)

    # Extract required columns
    df_acc = tmp.loc[:, [
        'OBJECTID', 'ACCIDENT_N', 'VehicleType', 'DAY_OF_WEE',
        'TOTAL_PERS', 'SEVERITY', 'LONGITUDE', 'LATITUDE']]

    # Rename specific columns
    df_acc = df_acc.rename(columns={
        'ACCIDENT_N': 'AccidentNumber', 'DAY_OF_WEE': 'DayOfWeek',
        'TOTAL_PERS': 'NumPeople', 'SEVERITY': 'Severity'})

    # Drop 'OBJECTID', 'Longitude' and 'Latitude' columns
    df_tmp = df_acc.drop(columns=['OBJECTID', 'LONGITUDE', 'LATITUDE'])

    # Create a geodataframe
    # Create points from latitude and longitude
    gdf = gpd.GeoDataFrame(
            df_tmp, geometry=gpd.points_from_xy(
                df_acc.loc[:, 'LONGITUDE'], df_acc.loc[:, 'LATITUDE']))

    # Set coordinate system
    gdf = gdf.set_crs('epsg:4283')  # GDA94

    return gdf


def export_gpkg(gdf, filename, layer_name, driver='GPKG'):
    """Exports a file based on a GeoDataFrame"""

    try:
        # Define filepath
        data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        output_path = os.path.join(data_path, 'outputs', filename)

        gdf.to_file(
            filename=output_path, layer=layer_name, driver=driver, mode='w')

    except OSError:
        print('File description may be wrong!')


# Task 2.3
def identify_SA2(fn_acc):
    """This function identifies SA2 in which the accident happened
    and add the SA2 field to 'AccidentLocations' layer.

    Args:
        fn_acc (String): Geopackage filename of accident data
    """

    try:
        # Read AccidentLocations layer
        data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        acc_path = os.path.join(data_path, 'outputs', fn_acc)
        acc = gpd.read_file(acc_path, layer='AccidentLocations')
        acc_columns = pd.Index(acc.columns.tolist())

        # Read SA2 shapefile
        sa2_path = os.path.join(
            data_path, 'data', 'RegionsSA2_2016', 'SA2_2016_AUST.shp')
        sa2 = gpd.read_file(sa2_path)

        acc_SA2 = gpd.sjoin(acc, sa2, how='left', predicate='within')

        # Redefine geodataframe (remove unnecessary columns)
        acc_SA2 = acc_SA2[acc_columns.append(pd.Index(['SA2_NAME16']))]
        acc_SA2 = acc_SA2.rename(columns={'SA2_NAME16': 'SA2'})

        # Overwrite a new goedataframe
        export_gpkg(acc_SA2, fn_acc, 'AccidentLocations', driver='GPKG')

    except OSError:
        print('File path may be wrong!!')

    except fiona.errors.DriverError:
        print('File path may be wrong!!')


# Task 2.3 (Using spatial index)
def identify_SA2_RTree(fn_acc):
    """This function identifies SA2 in which the accident happened
    and add the SA2 field to 'AccidentLocations' layer.
    Using RTree as a spatial index.

    Args:
        fn_acc (String): Geopackage filename of accident data
    """

    try:
        # Read AccidentLocations layer
        data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        acc_path = os.path.join(data_path, 'outputs', fn_acc)
        acc = gpd.read_file(acc_path, layer='AccidentLocations')

        # Read SA2 shapefile
        sa2_path = os.path.join(
            data_path, 'data', 'RegionsSA2_2016', 'SA2_2016_AUST.shp')
        sa2 = gpd.read_file(sa2_path)

        # Remove empty geometry
        acc_clean = acc.loc[acc.is_valid]
        sa2_clean = sa2.loc[sa2.is_valid]

        # Create an R-tree spatial index
        sindex = acc_clean.sindex

        # Identify SA2
        for i, sa2_poly in sa2_clean.iterrows():
            possible_matches_index = list(
                        sindex.intersection(sa2_poly.geometry.bounds))
            possible_matches = acc_clean.iloc[possible_matches_index]

            precise_matches_index = possible_matches\
                .intersects(sa2_poly.geometry)
            precise_matches_index = \
                precise_matches_index[precise_matches_index].index

            # Assign SA2 name
            acc_clean.loc[precise_matches_index,
                          'SA2'] = sa2_poly['SA2_NAME16']

        # Finalise
        acc_SA2 = acc_clean

        # Overwrite a new goedataframe
        export_gpkg(acc_SA2, fn_acc, 'AccidentLocations', driver='GPKG')

    except OSError:
        print('File path may be wrong!!')

    except fiona.errors.DriverError:
        print('File path may be wrong!!')


# Task 2.4
def split_accident_locations(fn_acc):
    """This function splits 'AccidentLocations' into
        'SevereAccidentsWeekday' and 'SevereAccidentsWeekend'.

    Args:
        fn_acc (String): Geopackage filename of accident data
    """

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    severity = ['Fatal accident', 'Serious injury']

    try:
        # Read AccidentLocations layer
        data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        acc_path = os.path.join(data_path, 'outputs', fn_acc)
        acc = gpd.read_file(acc_path, layer='AccidentLocations')

        # Create 'SevereAccidentsWeekday'
        acc_wdays = acc.loc[acc['DayOfWeek'].isin(weekdays)]
        acc_wdays_sev = acc_wdays.loc[acc_wdays['Severity'].isin(severity)]
        export_gpkg(
            acc_wdays_sev, fn_acc, 'SevereAccidentsWeekday', driver='GPKG')

        # Create 'SevereAccidentsWeekend'
        acc_wend = acc.loc[acc['DayOfWeek'].isin(weekends)]
        acc_wend_sev = acc_wend.loc[acc_wend['Severity'].isin(severity)]
        export_gpkg(
            acc_wend_sev, fn_acc, 'SeverityAccidentsWeekend', driver='GPKG')

    except OSError:
        print('File path may be wrong!!')

    except fiona.errors.DriverError:
        print('File path may be wrong!!')


def import_pop_gpkg():
    """This function reads a geopackage containing population data
    export combined data
    """

    try:
        # Define filepath
        data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        filepath = os.path.join(
                data_path, "data", "ERP_SA2_2020_gpkg",
                "SA2 ERP GeoPackage 2020.gpkg")

        # Read the file
        data_pop = gpd.read_file(filepath, layer="SA2_ERP_2020")

        # Retrieve data inside Victoria
        data_pop = data_pop[data_pop['State_name_2016'] == 'Victoria']

        # Retrive necessary columns
        data_pop = data_pop[[
            "State_name_2016", "SA2_maincode_2016", "SA2_name_2016",
            "ERP_2016", "geometry"]]

    except OSError:
        print('File path may be wrong!!')
    except KeyError:
        print("Unknow values")

    return data_pop
