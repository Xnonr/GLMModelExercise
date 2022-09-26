#!/usr/bin/env python3
# Indicates to the terminal that this file is not a shell script and must be run as Python3

# Imports Required Libraries
import os
import unittest

from common import batch_and_test

# Global Variables
test_files_directory = os.path.dirname(os.path.abspath(__file__))


class TestGLMModel(unittest.TestCase):

    def test_raw_json_1_row_v1(self):
        res = batch_and_test(500, 0, test_files_directory,
                             'sample_raw_json_1_row_v1.json')
        self.assertEqual(res[0]['cnt_actl'], 0)

    def test_raw_json_1_row_v2(self):
        res = batch_and_test(500, 1, test_files_directory,
                             'sample_raw_json_1_row_v2.json')
        self.assertEqual(res[0]['cnt_actl'], 0)

    def test_raw_json_10_rows(self):
        res = batch_and_test(500, 2, test_files_directory,
                             'sample_raw_json_10_rows.json')
        self.assertEqual(res[0]['cnt_actl'], 2)

    def test_raw_json_100_rows(self):
        res = batch_and_test(500, 3, test_files_directory,
                             'sample_raw_json_100_rows.json')
        self.assertEqual(res[0]['cnt_actl'], 24)

    def test_raw_json_1000_rows(self):
        res = batch_and_test(500, 4, test_files_directory,
                             'sample_raw_json_1000_rows.json')
        self.assertEqual(res[0]['cnt_actl'], 215)

    # @unittest.skip('Work in progress.')
    def test_raw_json_10000_rows(self):
        res = batch_and_test(500, 5, test_files_directory,
                             'sample_raw_json_10000_rows.json')
        self.assertEqual(res[0]['cnt_actl'], 2013)


if __name__ == "__main__":
    unittest.main()
