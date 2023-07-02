#!/usr/bin/python3

"""
Author: Chihiro Matsumoto
Date: 03/05/2022
Description: GEOM90042 Assignment3
    - Working with Rasters and Digital Elevation Models

    This is a test code for rasters.py
"""

import unittest
import os
import rasters
import reproducible_report.ProjectModule.PrettyTable as pt


class TestTaskOne(unittest.TestCase):
    def setUp(self):
        ''' Set variable for this class '''

        self.filename = os.path.join(os.getcwd(), "CLIP.tif")
        self.summary_dict = rasters.summary_dem(self.filename)

        # Correct coordinates derived from QGIS
        self.min_x = 363305.67772037826
        self.max_x = 451368.01830525836
        self.min_y = 5795416.55507391523
        self.max_y = 5856525.67432560957
        self.min_lon = 145.4548611116239556
        self.max_lon = 146.4462500005288348
        self.min_lat = -37.9787500002590619
        self.max_lat = -37.4368055558058686

    def test_summary_dem(self):
        ''' Test the summary_dem function
            summary_dem must return a dictionary '''

        # Test datatype of the returnd value
        self.assertTrue(
            isinstance(self.summary_dict, dict), "Output nut a dictionary")

        # Test coordinates
        self.assertAlmostEqual(
            self.summary_dict['min_x']['value'],
            self.min_x, "Min x seems incorrect")
        self.assertAlmostEqual(
            self.summary_dict['max_x']['value'],
            self.max_x, "Max x seems incorrect")
        self.assertAlmostEqual(
            self.summary_dict['min_y']['value'],
            self.min_y, "Min y seems incorrect")
        self.assertAlmostEqual(
            self.summary_dict['max_y']['value'],
            self.max_y, "Max y seems incorrect")
        self.assertAlmostEqual(
            self.summary_dict['min_lat']['value'],
            self.min_lat, "Min Lat seems incorrect")
        self.assertAlmostEqual(
            self.summary_dict['max_lat']['value'],
            self.max_lat, "Max Lat seems incorrect")
        self.assertAlmostEqual(
            self.summary_dict['min_lon']['value'],
            self.min_lon, "Min Lon seems incorrect")
        self.assertAlmostEqual(
            self.summary_dict['max_lon']['value'],
            self.max_lon, "Max Lon seems incorrect")

        # Test units of the coordinates
        self.assertEqual(
            self.summary_dict['min_x']['units'],
            'metre', "The unit of Min x not metre.")
        self.assertEqual(
            self.summary_dict['max_x']['units'],
            'metre', "The unit of Max x not metre.")
        self.assertEqual(
            self.summary_dict['min_y']['units'],
            'metre', "The unit of Min y not metre.")
        self.assertEqual(
            self.summary_dict['max_y']['units'],
            'metre', "The unit of Max y not metre.")
        self.assertEqual(
            self.summary_dict['min_lon']['units'],
            'degree', "The unit of Min Lon not degree.")
        self.assertEqual(
            self.summary_dict['max_lon']['units'],
            'degree', "The unit of Max Lon not degree.")
        self.assertEqual(
            self.summary_dict['min_lon']['units'],
            'degree', "The unit of Min Lat not degree.")
        self.assertEqual(
            self.summary_dict['max_lon']['units'],
            'degree', "The unit of Max Lat not degree.")

    def test_dict_to_nested_list(self):
        ''' Test the dict_to_nested_list function
            dict_to_nested_list must return a nested list. '''
        summary_dict = rasters.summary_dem(self.filename)
        return_value = rasters.dict_to_nested_list(summary_dict)
        self.assertTrue(
            any(isinstance(sub, list) for sub in return_value),
            "Output not a nested list.")

    def test_display_summary(self):
        ''' Test the display_summary function
            display_summary must return PrettyTable. '''
        return_value = rasters.display_summary(self.summary_dict)
        self.assertTrue(
            isinstance(return_value, pt.PrettyTable),
            "Output not a table derived from PrettyTable")


if __name__ == '__main__':
    unittest.main()
