#!/usr/bin/env python3
# Indicates to the terminal that this file is not a shell script and must be run as Python3

# Imports Required Libraries
import pandas as pd
import json
import requests


def retrieve_response(raw_json_data_file):
    '''
    Returns a list response after sending an HTTP POST request filled with raw JSON data

    Keyword Arguments:
    raw_json_data_file -- A file containing raw JSON data
    '''

    api_url = "http://127.0.0.1:1313/predict"

    post_json_data = open(raw_json_data_file, 'r')
    post_headers = {'content-type': 'application/json'}

    response = requests.post(
        api_url, data=post_json_data, headers=post_headers)
    response_json_data = response.json()

    return response_json_data


def test_prediction_results_count(test_num, test_preds):
    '''
    Returns a duo of String message of pertaining as to whether or not the given list's length 
        of predicted testing data based results match up with the pre-determined, true 
        and expected outcomes given the same said testing data

    Keyword Arguments:
    test_num - The test number for the given list of predictions being passed in
    test_preds - A list of containing dictionaries of prediction results converted from JSON
    '''

    test_res = 'FAILED'
    true_expected_prediction_results_counts = [0, 0, 2, 24, 215, 2013]

    test_preds_cnt = len(test_preds)

    if test_preds_cnt == true_expected_prediction_results_counts[test_num]:
        test_res = 'PASSED'

    test_res_dtls = f'\tPrediction Count: Expected = {true_expected_prediction_results_counts[test_num]}'
    test_res_dtls += f', Actual = {test_preds_cnt}\n\n'

    return test_res, test_res_dtls


def test_prediction_results_probabilities(test_preds):
    '''
    Returns a String message of 'PASSED' or 'FAILED' pertaining as to whether or not the given 
        list's predicted testing data based results' probabilities meet, equal or exceed the
        client established 75% probability cutoff point, as well as a list of those predictions
        which failed to meet said cutoff point

    Keyword Arguments:
    test_preds - A list of containing dictionaries of prediction results converted from JSON
    '''

    test_res = 'FAILED'
    failed_pred_tests_lst = []

    for pred in test_preds:
        pred_prob = float(pred.get('p_hat'))

        if pred_prob < 0.75:
            failed_pred_tests_lst.append(pred_prob)

    failed_pred_tests_cnt = len(failed_pred_tests_lst)

    if failed_pred_tests_cnt == 0:
        test_res = 'PASSED'

    return test_res, failed_pred_tests_lst


def test_prediction_results_business_outcomes(test_preds):
    '''
    Returns a String message of 'PASSED' or 'FAILED' pertaining as to whether or not the given 
        list's predicted testing data based results' business outcomes accurately reflect their
        associated probabilities, and are only those representing a sale to a customer, 
        as well as a list of those predictions which failed to meet requirements

    Keyword Arguments:
    test_preds - A list of containing dictionaries of prediction results converted from JSON
    '''

    test_res = 'FAILED'
    failed_pred_tests_lst = []

    for pred in test_preds:
        pred_prob = float(pred.get('p_hat'))
        busns_out = int(pred.get('business_outcome'))

        if busns_out != 1 or (busns_out == 1 and pred_prob < 0.75):
            failed_pred_tests_lst.append([busns_out, pred_prob])

    failed_pred_tests_cnt = len(failed_pred_tests_lst)

    if failed_pred_tests_cnt == 0:
        test_res = 'PASSED'

    return test_res, failed_pred_tests_lst


def test_prediction_results_proper_input_variables(test_preds):
    '''
    Returns a String message of 'PASSED' or 'FAILED' pertaining as to whether or not the given 
        list's predicted testing data based results' has all of the required input variables,
        that they are in proper alphanumerical order, have no missing blank, empty NaN or null values, 
        as well as a list of those predictions which failed to meet requirements

    Keyword Arguments:
    test_preds - A list of containing dictionaries of prediction results converted from JSON
    '''

    test_res = 'FAILED'
    failed_pred_tests_lst = []

    if len(test_preds) == 0:
        test_res = 'PASSED'

    else:
        true_expected_prediction_results_variables = sorted(
            ['x5_saturday', 'x81_July', 'x81_December', 'x31_japan', 'x81_October',
             'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May',
             'x5_monday', 'x81_September', 'x81_March', 'x53', 'x81_November',
             'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August',
             'x81_January', 'x62', 'x31_germany', 'x58', 'x56'])

        test_preds_df = pd.DataFrame(test_preds)

        test_preds_df_columns = list(test_preds_df.drop(['business_outcome', 'p_hat'],
                                                        axis=1).columns)

        test_preds_rows_missing_values = test_preds_df.isnull().any(axis=1)

        test_preds_rows_missing_values = (
            list(test_preds_rows_missing_values[test_preds_rows_missing_values == True].index))

        if test_preds_df_columns == true_expected_prediction_results_variables:
            if True not in set(test_preds_rows_missing_values):
                test_res = 'PASSED'
            else:
                for index in test_preds_rows_missing_values:
                    failed_pred_tests_lst.append(
                        test_preds_df.loc[index].to_dict())

    return test_res, failed_pred_tests_lst


def generate_prediction_messages():
    '''
    Returns a list of String test result messages describing each of their outcomes
    '''

    test_results = []

    sample_raw_json_data_files = ['sample_raw_json_1_row_v1.json',
                                  'sample_raw_json_1_row_v2.json',
                                  'sample_raw_json_10_rows.json',
                                  'sample_raw_json_100_rows.json',
                                  'sample_raw_json_1000_rows.json',
                                  'sample_raw_json_10000_rows.json']

    for index1 in range(len(sample_raw_json_data_files)):

        test_res_msg = '------------------------------------------------------------\n'
        test_res_msg += f'Test #{index1}: Data = {sample_raw_json_data_files[index1]}\n\n'

        pred_list = retrieve_response(sample_raw_json_data_files[index1])

        test_res1, test_res_dtls1 = test_prediction_results_count(
            index1, pred_list)
        test_res_msg += f'Results List Length Test: {test_res1}\n'
        test_res_msg += test_res_dtls1

        test_res2, failed_tests_lst1 = test_prediction_results_probabilities(
            pred_list)

        test_res_msg += f'Results List Probability Test: {test_res2}'

        if len(failed_tests_lst1) > 0:

            for index2 in range(len(failed_tests_lst1)):
                test_res_msg += (
                    f'\t Prediction #{index2}: p_hat: Expected >= 0.75, Actual = {failed_tests_lst1[index2]}\n')

        test_res3, failed_tests_lst2 = test_prediction_results_probabilities(
            pred_list)

        test_res_msg += f'\n\nResults List Business Outcome Test: {test_res3}'

        if len(failed_tests_lst2) > 0:

            for index3 in range(len(failed_tests_lst2)):
                test_res_msg += f'\t Prediction #{index3}: (business_outcome, p_hat): Expected (1, >= 0.75)'
                test_res_msg += f', Actual = ({failed_tests_lst1[index3][0]}, {failed_tests_lst1[index3][1]})\n'

        test_res4, failed_tests_lst3 = test_prediction_results_proper_input_variables(
            pred_list)

        test_res_msg += f'\n\nResults List Input Variables Test: {test_res4}'

        if len(failed_tests_lst3) > 0:

            for index4 in range(len(failed_tests_lst3)):
                test_res_msg += f'\t Prediction #{index4}: Missing Value(s) {failed_tests_lst3[index4]}\n'

        test_res_msg += '\n------------------------------------------------------------\n'

        test_results.append(test_res_msg)

    return test_results


def main():
    '''
    Runs all tests and prints out the resulting string messages of their outsomes
    '''

    test_results = generate_prediction_messages()

    for outcome in test_results:
        print(outcome)


if __name__ == "__main__":
    main()

