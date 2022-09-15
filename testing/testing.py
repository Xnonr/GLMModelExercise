# Imports Required Libraries
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
    
    response = requests.post(api_url, data = post_json_data, headers = post_headers)
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

def test_prediction_results_probabilities(test_num, test_preds):
    '''
    Returns a String message of 'PASSED' or 'FAILED' pertaining as to whether or not the given 
        list's predicted testing data based results' probabilities meet, equal or exceed the
        client established 75% probability cutoff point, as well as a list of those predictions
        which failed to meet said cutoff point
    
    Keyword Arguments:
    test_num - The test number for the given list of predictions being passed in
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

def test_predictions():
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
        
        test_res1, test_res_dtls1 = test_prediction_results_count(index1, pred_list)
        test_res_msg += f'Results List Length Test: {test_res1}\n'
        test_res_msg += test_res_dtls1

        test_res2, failed_tests_lst = test_prediction_results_probabilities(index1, pred_list)
        
        test_res_msg += f'Results List Probability Test: {test_res2}'
        
        if len(failed_tests_lst) > 0:
            
            for index2 in range(len(failed_tests_lst)):
                test_res_msg += (
                    f'\t Prediction #{index2}: p_hat: Expected >= 0.75, Actual = {failed_tests_lst[index2]}\n')
        
        test_res_msg += '\n------------------------------------------------------------\n'
            
        test_results.append(test_res_msg)
           
    return test_results

def main():
    
    test_results = test_predictions()
    
    for outcome in test_results:
        print(outcome)

main()