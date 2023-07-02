"""
Author: Chihiro Matsumoto
Date: 03/04/2022
Description: GEOM90042 Assignment1 - Trajectory analysis
"""

# Imports
import csv
import os
import sys
import math
from datetime import datetime
from pyproj import Transformer
from pyproj import CRS


def import_csv(filename):
    '''Import CSV file
    Arguments:
    filename -- name of the input file
    '''
    with open(filename, 'r') as in_file:
        data = list(csv.reader(in_file))
        points = data[1:]

    # Check if the file contains empty cell
    for row in points:
        for column in row:
            if len(column) == 0:
                sys.exit("The input file is invalid.")

    return points


def project_coordinate(from_epsg, to_epsg, in_x, in_y):
    ''' Return a single tuple of (x, y) in the output projection

    Arguments:
    from_epsg -- int, EPSG code of the original reference system
    to_epsg -- int, EPSG code of the new reference system
    in_x -- longitude
    in_y -- latitude
    '''

    transformer = Transformer.from_crs(
                    CRS(from_epsg), CRS(to_epsg), always_xy=True)
    utm = transformer.transform(in_x, in_y)

    return utm


def output_csv(input, xy):
    ''' Create a new CSV file including two new columns in a UTM projection
        and return the data in output file

    Arguments:
    input -- data from input file
    xy -- tuple of UTM coordinates
    '''

    output_data = []
    with open('assessment2.csv', 'w') as out_file:
        writer = csv.writer(out_file)

        # header
        header = [
            'trajectory_id', 'node_id', 'latitude',
            'longitude', 'altitude', 'time', 'x', 'y']
        writer.writerow(header)

        for i in range(len(xy)):
            row = input[i]
            row.extend(list(xy[i]))
            writer.writerow(row)
            output_data.append(row)

    return output_data


def task_1(filename):
    '''This function does task1
    1. Read CSV
    2. Process projection
    3. Write new CSV

    Return reprojected data including UTM coordinates

    Argument:
    filename -- filename
    '''

    input_points = import_csv(filename)
    utm = []     # x and y coordinates in UTM

    for p in input_points:
        # Type casting
        p[0] = int(p[0])
        p[1] = int(p[1])
        p[2] = float(p[2])
        p[3] = float(p[3])

        # Project coordinates
        reproj = project_coordinate(4326, 4796, p[3], p[2])
        utm.append(reproj)

    return output_csv(input_points, utm)


def compute_distance(from_x, from_y, to_x, to_y):
    '''Compute a distance between two UTM coordinate observations in metres.'''

    return math.sqrt((to_x-from_x)**2 + (to_y-from_y)**2)


def compute_time_difference(start_time, end_time):
    '''Compute the time difference between two timestamped observations in seconds.

    Arguments:
    start_time -- string
    end_time -- string
    '''

    # Compute the diffrence and convert to seconds
    diff = (datetime.strptime(end_time, "%H:%M:%S")
            - datetime.strptime(start_time, "%H:%M:%S"))
    diff_seconds = diff.seconds

    return diff_seconds


def compute_speed(total_distance, total_time):
    '''Compute the speed in metres per second

    Arguments:
    total_distance -- double
    total_time -- double
    '''

    return total_distance / total_time


def compute_sampling_rate(sum_time, num_sampling):
    '''Compute the average sampling rate

    Arguments:
    sum_time -- total time of the trajectory
    num_sumpling -- number of sumpling
    '''

    return sum_time / num_sampling


def print_trajectory_summary(traj, filename):
    '''Write summary of each trajectory

    Arguments:
    traj -- dictionary, one trajectry
    filename -- output filename
    '''
    file1 = open(filename, "a")
    file1.write(
        'Trajectory %d\'s length is %.2fm.\n'
        % (traj["id"]+1, traj["total_length"]))
    file1.write(
        'The length of its longest segment is %.2fm, and the index is %d.\n'
        % (traj["longest_seg"], traj["longest_seg_id"]))
    file1.write(
        'The average sampling rate for the trajectory is %.2fs.\n'
        % (traj["sum_time"]/traj["num_segments"]))
    file1.write(
        'For the segment index %d, the minimal travel speed is reached.\n'
        % (traj["min_speed_id"]))
    file1.write(
        'For the segment index %d, the maximum travel speed is reached.\n'
        % (traj["max_speed_id"]))
    file1.write('----\n')
    file1.close()


def print_final_report(report, filename):
    '''Print the final report

    Argument:
    report -- dictionary
    filename -- output filename
    '''
    file1 = open(filename, "a")
    file1.write(
        'The total length of all trajectories is %.2fm.\n'
        % (report["total_trajs"]))
    file1.write((
        'The index of the longest trajectory is %d, '
        + 'and the average speed along the trajectory is %.2fm/s.\n')
        % (report["longest_traj_id"], report["ave_speed"]))
    file1.close()


def task_3(reproj_data, text_fname):
    """This function does Task3

    Argument:
    reproj_data -- Reprojected data
    """

    # dictionary for one trajectry
    trajectory = {
        "id": 0,
        "num_segments": 0,
        "total_length": 0,
        "longest_seg": 0,
        "longest_seg_id": 0,
        "sum_time": 0,
        "min_speed": 1000000,
        "min_speed_id": 0,
        "max_speed": 0,
        "max_speed_id": 0
    }

    report = {
        "total_trajs": 0,
        "longest_traj": 0,
        "longest_traj_id": 0,
        "ave_speed": 0
    }

    # Create text file
    text_file = open(text_fname, "w")
    text_file.close()

    # Calculate segments line by line
    row = 1
    while row < len(reproj_data):

        # if the node is on the same trajectory
        if reproj_data[row-1][1] < reproj_data[row][1]:

            # Compute of the trajectory length
            segment_length = compute_distance(
                        reproj_data[row-1][6], reproj_data[row-1][7],
                        reproj_data[row][6], reproj_data[row][7])
            trajectory["id"] = reproj_data[row][0]
            trajectory["total_length"] += segment_length

            # Find the longest segment
            if trajectory["longest_seg"] < segment_length:
                trajectory["longest_seg_id"] = reproj_data[row][1]
                trajectory["longest_seg"] = segment_length

            # Sum up of sampling time
            sampling_time = compute_time_difference(
                reproj_data[row-1][5], reproj_data[row][5])
            trajectory["sum_time"] += sampling_time

            # Compute speed and Find min and max
            speed = compute_speed(segment_length, sampling_time)
            if trajectory["min_speed"] > speed:
                trajectory["min_speed_id"] = reproj_data[row][1]
                trajectory["min_speed"] = speed
            if trajectory["max_speed"] < speed:
                trajectory["max_speed_id"] = reproj_data[row][1]
                trajectory["max_speed"] = speed

        # if the observation is the first one of the trajectory
        # or the very last line of the dataset
        if (reproj_data[row-1][1] > reproj_data[row][1]
                or row == len(reproj_data)-1):

            # Print summary of the trajectory
            if reproj_data[row-1][1] > reproj_data[row][1]:
                trajectory["num_segments"] = reproj_data[row-1][1]
            else:
                trajectory["num_segments"] = reproj_data[row][1]

            print_trajectory_summary(trajectory, text_fname)

            # Update
            report["total_trajs"] += trajectory["total_length"]

            # Find the longest trajectory
            if report["longest_traj"] < trajectory["total_length"]:
                report["longest_traj"] = trajectory["total_length"]
                report["longest_traj_id"] = trajectory["id"]
                report["ave_speed"] = (
                    trajectory["total_length"] / trajectory["sum_time"])

        # Initialize if the node is the first one of the trajectory
        if reproj_data[row-1][1] > reproj_data[row][1]:
            trajectory["id"] = 0
            trajectory["num_segments"] = 0
            trajectory["total_length"] = 0
            trajectory["longest_seg"] = 0
            trajectory["longest_seg_id"] = 0
            trajectory["sum_time"] = 0
            trajectory["min_speed"] = 1000000
            trajectory["min_speed_id"] = 0
            trajectory["max_speed"] = 0

        # Move next row
        row += 1

    # Print final report
    print_final_report(report, text_fname)


def main():

    # Set up constant
    csv_file = 'trajectory_data.csv'    # input file
    filename1 = os.path.join(os.getcwd(), csv_file)
    text_file = 'assessment2_out.txt'   # report file
    filename2 = os.path.join(os.getcwd(), text_file)

    # Run tasks
    reprojected_data = task_1(filename1)
    task_3(reprojected_data, filename2)

    # Finished
    out_text = open(filename2, "a")
    out_text.write("Program complete\n")
    out_text.close()


if __name__ == '__main__':
    main()
