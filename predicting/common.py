# Imports Required Libraries
import asyncio
import numpy as np
import pandas as pd

from math import ceil

def transform_json_to_df(obj):
    '''
    Returns a DataFrame containing raw JSON data

    Keyword Arguments:
    obj -- Raw JSON data in the data type form of a String, List or Dictionary
    '''

    if isinstance(obj, str):
        if obj[0] == '[' and obj[len(obj) - 1] == ']':
            df = pd.read_json(obj, orient='records')
        elif obj[0] == '{' and obj[len(obj) - 1] == '}':
            obj = '[' + obj + ']'
            df = pd.read_json(obj, orient='records')
        else:
            df = pd.DataFrame()
    elif isinstance(obj, list):
        df = pd.DataFrame.from_dict(obj)
    elif isinstance(obj, dict):
        obj = [obj]
        df = pd.DataFrame.from_dict(obj)
    else:
        df = pd.DataFrame()

    return df


def batch_df(max_batch_size, df):
    '''
    Returns

    Keyword Arguments:
    max_batch_size -- 
    df -- 
    '''

    df_batches = []

    num_rows = len(df.index)

    if num_rows > max_batch_size:

        num_batches = ceil(num_rows / max_batch_size)

        for batch_num in range(num_batches):
            starting_index = batch_num * max_batch_size
            #print(f'Starting Index: {starting_index}')

            if batch_num != num_batches - 1:
                ending_index = (batch_num + 1) * max_batch_size
                #print(f'Ending Index: {ending_index}')
                batch_df = df.iloc[starting_index:ending_index].copy(deep=True)

            else:
                batch_df = df.iloc[starting_index:].copy(deep=True)

            df_batches.append(batch_df)

    else:
        df_batches.append(df.copy(deep=True))

    return df_batches


def format_df_column_variables(df):
    '''
    Returns a DataFrame after having transformed those columns, whose values 
        consisted of those of the String data type and representing quantitative 
        data, into values being that of the Float data type after removing any
        non mathematically interpretable symbols 

    Keyword Arguments:
    df -- A DataFrame containing raw JSON data
    '''

    # Formats the 'x12' and 'x63' columns', consisting of the String data type,
    #    and respectively representing first monetary then percentage values,
    #    into the Float data type so as to be able to later on apply mathematical
    #    work upon said columns' values

    columns_to_format = ['x12', 'x63']

    unwanted_symbols = ['$', ',', '(', ')', '%']

    for clmn in columns_to_format:
        for usymb in unwanted_symbols:
            if usymb != '(':
                df[clmn] = df[clmn].str.replace(usymb, '', regex=True)
            else:
                df[clmn] = df[clmn].str.replace(usymb, '-', regex=True)
        df[clmn] = df[clmn].astype(float)

    return df


def impute_missing_df_data(si, df):
    '''
    Returns a DataFrame with no column variables whose data is of a qualitative nature, 
        as well as having filled in any remaining blank, NaN, NULL or otherwise missing 
        values via the usage of an imported, pre-trained SimpleImputer using a mean based 
        strategy

    Keyword Arguments:
    si -- A pre-trained, imported SimpleImputer
    df -- A formatted DataFrame
    '''

    # Columns containing qualitative data that are dropped ahead of the imputation phase
    #    as the SimpleImputer cannot properly function upon them
    qual_clmns = ['x5', 'x31', 'x81', 'x82']

    df = pd.DataFrame(si.transform(df.drop(columns=qual_clmns)),
                      columns=df.drop(columns=qual_clmns).columns)

    return df


def scale_df_data(ss, df):
    '''
    Returns a DataFrame whose column variable values have all been scaled via a 
        standardization method for the purpose of feature scaling, utilizing an
        imported, pre-trained StandardScalar

    Keyword Arguments:
    ss -- A pre-trained, imported StandardScalar
    df -- A formatted DataFrame without any blank, NaN, NULL or otherwise missing values
    '''

    # Of particular interest and focus is that of the 'x12' column representing
    #    monetary values which tend to outscale all other column variable values
    #    by some orders of magnitude

    df = pd.DataFrame(ss.transform(df),
                      columns=df.columns)

    return df


def create_df_dummy_column_variables_new(df1, df2):
    '''
    Returns a DataFrame with dummy variables for those column variables consisting of
        qualitative data, as well as whose column variables of numeric quantitative data
        have no missing values and are scaled

    Keyword Arguments:
    df1 -- A DataFrame containing the original raw JSON data in order to retrieve those column
           variable values of a qualitative nature previously dropped and must now be 
           dummified
    df2 -- A DataFrame with no missing column variable values and whose said values have already been scaled
    '''

    # A list of column variable names of a quantitative nature which require dummification
    vars_to_dummify = ['x5', 'x31', 'x81', 'x82']
    dummy_dfs = [df2]

    for var in vars_to_dummify:

        var_dummy_vars_df = pd.get_dummies(df1[var],
                                           drop_first=True,
                                           prefix=var,
                                           prefix_sep='_',
                                           dummy_na=True)

        dummy_dfs.append(var_dummy_vars_df)

    df2 = pd.concat(dummy_dfs, axis=1, sort=False)

    return df2


def filter_df_column_variables(ordr_clmn_names_lst, df):
    '''
    Returns a DataFrame containing only those column variables required by the pre-trained model
        for predictions, filtering out from the given DataFrame only said columns

    Keyword Arguments:
    ordr_clmn_names_lst -- A list of pre-ordered DataFrame column variable names required by the pre-trained model
    df -- The DataFrame with dummy variables, and whose quantitative values have been scaled and 
          have none missing
    '''

    necessary_clmn_vars_set = set(ordr_clmn_names_lst)
    avlbl_clmn_vars_set = set(df.columns)

    # Depending upon the type and amount of JSON data originally having been passed in,
    #    not all of the desired dummy variables will always be successfully generated,
    #    necessitating their inclusion afterwards via the code below
    if necessary_clmn_vars_set.issubset(avlbl_clmn_vars_set) == False:
        nan_df = pd.DataFrame(np.nan, index=range(
            df.shape[0]), columns=ordr_clmn_names_lst)
        df = df.combine_first(nan_df)
        df = df.fillna(0)

    #df = df[ordr_clmn_names_lst].copy()
    df = df[ordr_clmn_names_lst].copy(deep=True)

    return df


def predict_outcomes(df, mdl, alphanum_ord_clmn_var_names_lst):
    '''
    Returns a JSON message containing either the model's predicted outcomes, 
        marked as 'business_outcome', predicted probability, marked as 'p_hat',
        and the inputs in the alphanumerical order of their variables' names or,
        should the predicted probability be under that of 75%, a message indicating
        as such; if no valid JSON data as far as the application can tell is passed in, 
        then a JSON encoded error message will be returned instead

    Keyword Arguments:
    df -- A DataFrame containing only those 25 ordered column variables required by the pre-trained model
    mdl -- A pre-trained, imported prediction model
    alphanum_ord_clmn_var_names_lst -- A list of alphanumerically ordered DataFrame column variable names 
                                       required by the pre-trained model
    '''

    num_rows_df = df.shape[0]

    if num_rows_df == 0:

        return {'message': 'ERROR - No valid JSON data available for prediction.'}

    else:

        output_msgs_lst = []

        for row in range(num_rows_df):
            predicted_outcome = 0
            mdl_inputs = {}

            predicted_probability = mdl.predict(df.iloc[row])[0]

            if predicted_probability >= 0.75:
                predicted_outcome = 1

                for var in alphanum_ord_clmn_var_names_lst:
                    mdl_inputs[var] = df.iloc[row][var]

                mdl_predictions = {'business_outcome': str(predicted_outcome),
                                   'p_hat': str(predicted_probability)}

                prediction_msg = mdl_predictions | mdl_inputs

                output_msgs_lst.append(prediction_msg)

            else:
                #output_msgs_lst.append({'message': 'Business outcome probability too low.'})
                pass

        return output_msgs_lst


def extract_transform_predict_df(df, si, ss, ordr_clmn_names_lst,
                                 mdl, alphanum_ord_clmn_var_names_lst):
    '''
    Returns 

    Keyword Arguments:
    df -- A DataFrame containing raw JSON data
    si -- A pre-trained, imported SimpleImputer
    ss -- A pre-trained, imported StandardScalar
    ordr_clmn_names_lst -- A list of pre-ordered DataFrame column variable names required by the pre-trained model
    mdl -- A pre-trained, imported prediction model
    alphanum_ord_clmn_var_names_lst -- A list of alphanumerically ordered DataFrame column variable names 
                                       required by the pre-trained model
    '''

    if df.shape[0] < 1:
        filtered_dummy_scaled_imputed_df = pd.DataFrame()

    else:

        # Resets the index to avoid concatenation issues with dummy variable DataFrames
        df.reset_index(drop=True, inplace=True)

        batch_df = df.copy(deep=True)

        formatted_df = format_df_column_variables(batch_df)

        imputed_df = impute_missing_df_data(si, formatted_df)

        scaled_imputed_df = scale_df_data(ss, imputed_df)

        dummy_scaled_imputed_df = create_df_dummy_column_variables_new(
            df, scaled_imputed_df)

        filtered_dummy_scaled_imputed_df = filter_df_column_variables(ordr_clmn_names_lst,
                                                                      dummy_scaled_imputed_df)

    predictions = predict_outcomes(filtered_dummy_scaled_imputed_df, mdl,
                                   alphanum_ord_clmn_var_names_lst)

    return predictions


async def async_extract_transform_predict_df(df, si, ss, ordr_clmn_names_lst,
                                             mdl, alphanum_ord_clmn_var_names_lst):
    '''
    Returns 

    Keyword Arguments:
    df -- A DataFrame containing raw JSON data
    si -- A pre-trained, imported SimpleImputer
    ss -- A pre-trained, imported StandardScalar
    ordr_clmn_names_lst -- A list of pre-ordered DataFrame column variable names required by the pre-trained model
    mdl -- A pre-trained, imported prediction model
    alphanum_ord_clmn_var_names_lst -- A list of alphanumerically ordered DataFrame column variable names 
                                       required by the pre-trained model
    '''

    # Triggers the asynchronous utility by informing Python that other functions of this kind can
    #    be started up in parallel once the program reaches this line of code's depth
    await asyncio.sleep(0)

    if df.shape[0] < 1:
        filtered_dummy_scaled_imputed_df = pd.DataFrame()

    else:

        # Resets the index to avoid concatenation issues with dummy variable DataFrames
        df.reset_index(drop=True, inplace=True)

        batch_df = df.copy(deep=True)

        formatted_df = format_df_column_variables(batch_df)

        imputed_df = impute_missing_df_data(si, formatted_df)

        scaled_imputed_df = scale_df_data(ss, imputed_df)

        dummy_scaled_imputed_df = create_df_dummy_column_variables_new(
            df, scaled_imputed_df)

        filtered_dummy_scaled_imputed_df = filter_df_column_variables(ordr_clmn_names_lst,
                                                                      dummy_scaled_imputed_df)

    predictions = predict_outcomes(filtered_dummy_scaled_imputed_df, mdl,
                                   alphanum_ord_clmn_var_names_lst)

    return predictions


def extract_transform_predict_df_batches(df_batches, si, ss, ordr_clmn_names_lst,
                                         mdl, alphanum_ord_clmn_var_names_lst):
    '''
    Returns

    Keyword Arguments:
    df_batches -- 
    si -- A pre-trained, imported SimpleImputer
    ss -- A pre-trained, imported StandardScalar
    ordr_clmn_names_lst -- A list of DataFrame column variable names required by the pre-trained model
    mdl -- A pre-trained, imported prediction model
    alphanum_ord_clmn_var_names_lst -- A list of alphanumerically ordered DataFrame column variable names 
                                       required by the pre-trained model
    '''

    df_batches_predictions = []

    for batch in df_batches:
        batch_predictions = extract_transform_predict_df(batch, si, ss, ordr_clmn_names_lst,
                                                         mdl, alphanum_ord_clmn_var_names_lst)

        df_batches_predictions += batch_predictions

    return df_batches_predictions


async def async_extract_transform_predict_df_batches(df_batches, si, ss, ordr_clmn_names_lst,
                                                     mdl, alphanum_ord_clmn_var_names_lst):
    '''
    Returns

    Keyword Arguments:
    df_batches -- 
    si -- A pre-trained, imported SimpleImputer
    ss -- A pre-trained, imported StandardScalar
    ordr_clmn_names_lst -- A list of DataFrame column variable names required by the pre-trained model
    mdl -- A pre-trained, imported prediction model
    alphanum_ord_clmn_var_names_lst -- A list of alphanumerically ordered DataFrame column variable names 
                                       required by the pre-trained model
    '''

    df_batches_predictions = []

    tasks = []

    for batch in df_batches:
        tasks.append(asyncio.ensure_future(async_extract_transform_predict_df(
            batch, si, ss, ordr_clmn_names_lst, mdl, alphanum_ord_clmn_var_names_lst)))

    results = await asyncio.gather(*tasks)

    for batch_predictions in results:
        df_batches_predictions += batch_predictions

    return df_batches_predictions
