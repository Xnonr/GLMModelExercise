# Imports Required Libraries
import aiohttp
import asyncio
import json
import pandas as pd
import os
import requests
import time

from math import ceil


def batch_by_row_count(max_batch_size, test_file_directory, raw_json_data_test_file):
    '''
    Returns 

    Keyword Arguments:
    max_batch_size --
    raw_json_data_test_file -- 
    '''

    batches_of_content = 1
    batched_contents = []

    with open(os.path.join(test_file_directory, raw_json_data_test_file), 'rb') as file:
        content = json.load(file)

        if isinstance(content, list) == False:
            content = [content]

        length_of_content = len(content)

        #print(f'Total Rows of Data: {length_of_content}')

        if length_of_content > max_batch_size:
            batches_of_content = ceil(length_of_content / max_batch_size)

            #print(f'Total Batches of Data: {batches_of_content}')

            for batch_number in range(batches_of_content):

                batch_of_content = []

                if batch_number != batches_of_content - 1:
                    batch_of_content = content[
                        (batch_number * max_batch_size):(((batch_number + 1) * max_batch_size))]
                else:
                    batch_of_content = content[
                        (batch_number * max_batch_size):]

                batched_contents.append(batch_of_content)

        else:
            batched_contents.append(content)

    return batched_contents


def retrieve_http_response(raw_json_data):
    '''
    Returns a List formatted response after sending an HTTP POST request filled with raw JSON data

    Keyword Arguments:
    raw_json_data -- Raw JSON data used for testing purposes
    '''

    api_url = "http://127.0.0.1:1313/predict"

    post_headers = {'content-type': 'application/json'}

    response = requests.post(api_url,
                             data=json.dumps(raw_json_data),
                             headers=post_headers)

    if (response.status_code != 200):
        print('----- Server returned error -----')
        print(raw_json_data)
        print(response)
        print('---------------------------------')
        response_json_data = []
    else:
        response_json_data = response.json()

    # Prints the size of the HTTP response in Bytes
    #print(f'Size of HTTP Response: {len(response.content)} Bytes')

    return response_json_data


async def async_retrieve_http_response(session, raw_json_data):
    '''
    Returns a List formatted response after sending an HTTP POST request filled with raw JSON data

    Keyword Arguments:
    raw_json_data -- Raw JSON data used for testing purposes
    '''

    api_url = "http://127.0.0.1:1313/predict"

    post_headers = {'content-type': 'application/json'}

    async with session.post(api_url, data=json.dumps(raw_json_data), headers=post_headers) as response:

        response_json_data = await response.json()

        return response_json_data


def prediction_results_verify_counts(test_num, test_preds):
    '''
    Returns 

    Keyword Arguments:
    test_num - The test number for the given list of predicition dictionaries being passed in
    test_preds - A list containing dictionaries of prediction results converted from JSON
    '''

    test_res = False
    true_preds_cnt = [0, 0, 2, 24, 215, 2013]

    test_preds_cnt = len(test_preds)

    if test_preds_cnt == true_preds_cnt[test_num]:
        test_res = True

    test_res_dtls = {
        # 'passed': test_res,
        'expctd': true_preds_cnt[test_num],
        'actl': test_preds_cnt
    }

    return test_res_dtls


def prediction_results_verify_probabilities(test_preds):
    '''
    Returns 

    Keyword Arguments:
    test_preds - A list of containing dictionaries of prediction results converted from JSON
    '''

    test_res = False
    invalid_test_preds = []

    for pred in test_preds:
        pred_prob = float(pred.get('p_hat'))

        if pred_prob < 0.75:
            invalid_test_preds.append(pred_prob)

    if len(invalid_test_preds) == 0:
        test_res = True

    test_res_dtls = {
        'passed': test_res,
        'invld_preds': invalid_test_preds
    }

    return test_res_dtls


def prediction_results_verify_business_outcomes(test_preds):
    '''
    Returns 

    Keyword Arguments:
    test_preds - A list of containing dictionaries of prediction results converted from JSON
    '''

    test_res = False
    invalid_test_preds = []

    for pred in test_preds:
        pred_prob = float(pred.get('p_hat'))
        pred_busns_out = int(pred.get('business_outcome'))

        if pred_busns_out != 1 or (pred_busns_out == 1 and pred_prob < 0.75):
            invalid_test_preds.append([pred_busns_out, pred_prob])

    if len(invalid_test_preds) == 0:
        test_res = True

    test_res_dtls = {
        'passed': test_res,
        'invld_preds': invalid_test_preds
    }

    return test_res_dtls


def prediction_results_verify_proper_input_variables(test_preds):
    '''
    Returns 

    Keyword Arguments:
    test_preds - A list of containing dictionaries of prediction results converted from JSON
    '''

    test_res = False
    invalid_test_preds = []

    if len(test_preds) == 0:
        test_res = True

    else:
        true_pred_vars = sorted(
            ['x5_saturday', 'x81_July', 'x81_December', 'x31_japan', 'x81_October',
             'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May',
             'x5_monday', 'x81_September', 'x81_March', 'x53', 'x81_November',
             'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August',
             'x81_January', 'x62', 'x31_germany', 'x58', 'x56'])

        test_preds_df = pd.DataFrame(test_preds)

        test_preds_df_clmns = list(test_preds_df.drop(['business_outcome', 'p_hat'],
                                                      axis=1).columns)

        test_preds_rows_missing_values = test_preds_df.isnull().any(axis=1)

        test_preds_rows_missing_values = (
            list(test_preds_rows_missing_values[test_preds_rows_missing_values == True].index))

        if test_preds_df_clmns == true_pred_vars:
            if True not in set(test_preds_rows_missing_values):
                test_res = True
            else:
                for index in test_preds_rows_missing_values:
                    invalid_test_preds.append(
                        test_preds_df.loc[index].to_dict())

    test_res_dtls = {
        'passed': test_res,
        'invld_preds': invalid_test_preds
    }

    return test_res_dtls


def prediction_results_agglomeration(cnt_dtls, prob_dtls, busns_out_dtls, in_var_dtls):
    '''
    Returns

    Keyword Arguments:
    cnt_dtls -- 
    prob_dtls -- 
    busns_out_dtls -- 
    in_var_dtls -- 
    '''
    tests_res_dtls = {
        # 'cnt_passed': cnt_dtls.get('passed'),
        'cnt_expctd': cnt_dtls.get('expctd'),
        'cnt_actl': cnt_dtls.get('actl'),
        'prob_passed': prob_dtls.get('passed'),
        'prob_invld_preds': prob_dtls.get('invld_preds'),
        'busns_out_passed': busns_out_dtls.get('passed'),
        'busns_out_invld_preds': busns_out_dtls.get('invld_preds'),
        'in_var_passed': in_var_dtls.get('passed'),
        'in_var_invld_preds': in_var_dtls.get('invld_preds')
    }

    return tests_res_dtls


def collect_batch_prediction_results(test_num, raw_json_data):
    '''
    Returns


    Keyword Arguments:
    test_num --
    raw_json_data -- 
    '''

    batch_preds = retrieve_http_response(raw_json_data)

    cnt_res_dtls = prediction_results_verify_counts(test_num, batch_preds)
    prob_res_dtls = prediction_results_verify_probabilities(batch_preds)
    busns_out_res_dtls = prediction_results_verify_business_outcomes(
        batch_preds)
    in_var_res_dtls = prediction_results_verify_proper_input_variables(
        batch_preds)

    batch_res_dtls = prediction_results_agglomeration(cnt_res_dtls,
                                                      prob_res_dtls,
                                                      busns_out_res_dtls,
                                                      in_var_res_dtls)

    return batch_res_dtls


async def async_collect_batch_prediction_results(session, test_num, raw_json_data):
    '''
    Returns


    Keyword Arguments:
    test_num --
    raw_json_data -- 
    '''

    batch_preds = await async_retrieve_http_response(session, raw_json_data)

    cnt_res_dtls = prediction_results_verify_counts(test_num, batch_preds)
    prob_res_dtls = prediction_results_verify_probabilities(batch_preds)
    busns_out_res_dtls = prediction_results_verify_business_outcomes(
        batch_preds)
    in_var_res_dtls = prediction_results_verify_proper_input_variables(
        batch_preds)

    batch_res_dtls = prediction_results_agglomeration(cnt_res_dtls,
                                                      prob_res_dtls,
                                                      busns_out_res_dtls,
                                                      in_var_res_dtls)

    return batch_res_dtls


def merge_batch_agglomerated_prediction_results(aglom_batch_preds_res1, aglom_batch_preds_res2):
    '''
    Returns

    Keyword Arguments:
    aglom_batch_preds_res1 -- 
    aglom_batch_preds_res1 -- 
    '''

    merged_tests_res_dtls = {
        # 'cnt_passed':
        'cnt_expctd': aglom_batch_preds_res1.get('cnt_expctd'),

        'cnt_actl': aglom_batch_preds_res1.get('cnt_actl') + aglom_batch_preds_res2.get('cnt_actl'),

        'prob_passed':  (
            False
            if (aglom_batch_preds_res1.get('prob_passed') == False or
                aglom_batch_preds_res2.get('prob_passed') == False)
            else True),

        'prob_invld_preds': aglom_batch_preds_res1.get('prob_invld_preds') + aglom_batch_preds_res2.get('prob_invld_preds'),

        'busns_out_passed': (
            False
            if (aglom_batch_preds_res1.get('busns_out_passed') == False or
                aglom_batch_preds_res2.get('busns_out_passed') == False)
            else True),

        'busns_out_invld_preds': aglom_batch_preds_res1.get('busns_out_invld_preds') + aglom_batch_preds_res2.get('busns_out_invld_preds'),

        'in_var_passed': (
            False
            if (aglom_batch_preds_res1.get('in_var_passed') == False or
                aglom_batch_preds_res2.get('in_var_passed') == False)
            else True),

        'in_var_invld_preds': aglom_batch_preds_res1.get('in_var_invld_preds') + aglom_batch_preds_res2.get('in_var_invld_preds')
    }

    return merged_tests_res_dtls


def collect_merged_batch_prediction_results(test_num, sample_raw_json_data_batches):
    '''
    Returns

    Keyword Arguments:
    test_num -- 
    sample_raw_json_data_batches --
    '''

    fnl_test_res_dtls = {}

    for batch in sample_raw_json_data_batches:
        if not fnl_test_res_dtls:
            fnl_test_res_dtls.update(
                collect_batch_prediction_results(test_num,
                                                 batch))

        else:
            fnl_test_res_dtls = merge_batch_agglomerated_prediction_results(
                fnl_test_res_dtls, collect_batch_prediction_results(test_num,
                                                                    batch))

    return fnl_test_res_dtls


async def async_collect_merged_batch_prediction_results(test_num, sample_raw_json_data_batches):
    '''
    Returns

    Keyword Arguments:
    test_num -- 
    sample_raw_json_data_batches --
    '''

    async with aiohttp.ClientSession() as session:

        tasks = []
        for batch in sample_raw_json_data_batches:
            tasks.append(asyncio.ensure_future(
                async_collect_batch_prediction_results(session, test_num, batch)))

        results = await asyncio.gather(*tasks)

        fnl_test_res_dtls = {}
        for result in results:
            if not fnl_test_res_dtls:
                fnl_test_res_dtls.update(result)
            else:
                fnl_test_res_dtls = merge_batch_agglomerated_prediction_results(
                    fnl_test_res_dtls, result)

        return fnl_test_res_dtls


def batch_and_test(batch_size, test_num, test_file_dir, sample_raw_json_data_file):
    '''
    Returns

    Keyword Arguments:
    batch_size --
    sample_raw_json_data_file --
    '''

    starting_time = time.time()

    batched_sample_raw_json_data = batch_by_row_count(batch_size,
                                                      test_file_dir,
                                                      sample_raw_json_data_file)

    tests_res = collect_merged_batch_prediction_results(
        test_num, batched_sample_raw_json_data)

    sync_time_to_completion = {time.time() - starting_time}

    return tests_res, sync_time_to_completion


async def async_batch_and_test(batch_size, test_num, test_file_dir, sample_raw_json_data_file):
    '''
    Returns

    Keyword Arguments:
    batch_size --
    sample_raw_json_data_file --
    '''

    starting_time = time.time()

    batched_sample_raw_json_data = batch_by_row_count(batch_size,
                                                      test_file_dir,
                                                      sample_raw_json_data_file)

    tests_res = await async_collect_merged_batch_prediction_results(test_num, batched_sample_raw_json_data)

    async_time_to_completion = {time.time() - starting_time}

    return tests_res, async_time_to_completion
