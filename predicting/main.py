# Imports required libraries
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

from fastapi import FastAPI
from fastapi import Request

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

mdl = None

final_df_column_variable_names_order = [
    'x5_saturday', 'x81_July', 'x81_December', 'x31_japan', 'x81_October',
    'x5_sunday', 'x31_asia', 'x81_February', 'x91', 'x81_May',
    'x5_monday', 'x81_September', 'x81_March', 'x53', 'x81_November',
    'x44', 'x81_June', 'x12', 'x5_tuesday', 'x81_August',
    'x81_January', 'x62', 'x31_germany', 'x58', 'x56']

alphanumerically_sorted_df_column_variable_names = sorted(
    final_df_column_variable_names_order)

app = FastAPI(title='GLM', description='GLM API', version='1.0')


@app.on_event('startup')
def load_model():
    global mdl
    mdl = sm.load('models/glm_final_model.pickle')


@app.post('/predict')
async def get_prediction(info: Request):

    req_info = await info.json()

    df = extract_transform_input_data_pipeline(
        req_info, final_df_column_variable_names_order)

    json_output_message = predict_outcome(
        df, mdl, alphanumerically_sorted_df_column_variable_names)

    return json_output_message


def transform_json_to_df(json):

    df = pd.read_json(json, orient='records')

    return df


def format_df_column_variables(df):

    # Formats the 'x12' column's String monetary value into a Float in order to apply maths upon it
    df['x12'] = df['x12'].str.replace('$', '')
    df['x12'] = df['x12'].str.replace(',', '')
    df['x12'] = df['x12'].str.replace(')', '')
    df['x12'] = df['x12'].str.replace('(', '-')
    df['x12'] = df['x12'].astype(float)

    df['x63'] = df['x63'].str.replace('%', '')
    df['x63'] = df['x63'].astype(float)

    return df


def impute_missing_df_data(df):

    if df.shape[0] <= 1:
        df = pd.DataFrame(df.drop(columns=['x5', 'x31', 'x81', 'x82']),
                          columns=df.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
        df = df.fillna(0)

    else:
        # Creates and instantiates a simple imputer
        si = SimpleImputer(missing_values=np.nan, strategy='mean')

        # Imputes via a simple mean stategy those column values which are missing
        df = pd.DataFrame(si.fit_transform(df.drop(columns=['x5', 'x31', 'x81', 'x82'])),
                          columns=df.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)

    return df


def scale_df_data(df):

    # Creates and instantiates a standard scaler
    ss = StandardScaler()

    '''
    Scales all column values via a standardization method for feature scaling, 
        of particular interest and focus being being that of the monetary value column
    '''
    df = pd.DataFrame(ss.fit_transform(df),
                      columns=df.columns)

    return df


def create_df_dummy_column_variables_new(df1, df2):
    '''
    Creates the dummy variables for the non-numeric, qualitative data type columns and then reconcatenates
        them back into the now imputed and standardized scaled dataframe
    '''
    vars_to_dummify = ['x5', 'x31', 'x81', 'x82']

    for var in vars_to_dummify:

        var_dummy_vars = pd.get_dummies(df1[var],
                                        drop_first=True,
                                        prefix=var,
                                        prefix_sep='_',
                                        dummy_na=True)

        df2 = pd.concat([df2, var_dummy_vars],
                        axis=1,
                        sort=False)

    return df2


def create_df_dummy_column_variables_old(df1, df2):

    x5_dummy_variables = pd.get_dummies(df1['x5'],
                                        drop_first=True,
                                        prefix='x5',
                                        prefix_sep='_',
                                        dummy_na=True)

    df2 = pd.concat([df2, x5_dummy_variables], axis=1, sort=False)

    x31_dummy_variables = pd.get_dummies(df1['x31'],
                                         drop_first=True,
                                         prefix='x31',
                                         prefix_sep='_',
                                         dummy_na=True)

    df2 = pd.concat([df2, x31_dummy_variables], axis=1, sort=False)

    x81_dummy_variables = pd.get_dummies(df1['x81'],
                                         drop_first=True,
                                         prefix='x81',
                                         prefix_sep='_',
                                         dummy_na=True)

    df2 = pd.concat([df2, x81_dummy_variables], axis=1, sort=False)

    x82_dummy_variables = pd.get_dummies(df1['x82'],
                                         drop_first=True,
                                         prefix='x82',
                                         prefix_sep='_',
                                         dummy_na=True)

    df2 = pd.concat([df2, x82_dummy_variables], axis=1, sort=False)

    return df2


'''
Filters out and retrives only those columns previously determined to be the most useful during the
    creation of the prediction model
'''


def filter_df_column_variables(df, ordr_clmn_names_lst):

    necessary_clmn_vars_set = set(ordr_clmn_names_lst)
    avlbl_clmn_vars_set = set(df.columns)

    '''
    Depending upon the type and amount of data passed in, not all dummy variables will always be
        successfully generated, necessitating their inclusion afterwards
    '''
    if necessary_clmn_vars_set.issubset(avlbl_clmn_vars_set) == False:
        nan_df = pd.DataFrame(np.nan, index=range(
            df.shape[0]), columns=ordr_clmn_names_lst)
        df = df.combine_first(nan_df)
        df = df.fillna(0)

    df = df[ordr_clmn_names_lst].copy(deep=True)

    return df


def extract_transform_input_data_pipeline(json_data, ordr_clmn_names_lst):

    df = transform_json_to_df(json_data)

    if df.shape[0] < 1:
        df = pd.DataFrame()

    else:
        df = format_df_column_variables(df)

        imputed_df = impute_missing_df_data(df)

        scaled_imputed_df = scale_df_data(imputed_df)

        df = create_df_dummy_column_variables_old(df, scaled_imputed_df)

        df = filter_df_column_variables(df, ordr_clmn_names_lst)

    return df


def predict_outcome(df, model, alphanum_ord_clmn_var_names_lst):

    num_rows_df = df.shape[0]

    if num_rows_df == 0:
        return json.dumps({'message': 'Error'})

    else:

        for row in range(num_rows_df):
            predicted_outcome = 0
            model_inputs = {}

            for var in alphanum_ord_clmn_var_names_lst:
                model_inputs[var] = df.iloc[row][var]

            predicted_probability = model.predict(df.iloc[row])[0]

            if predicted_probability >= 0.75:
                predicted_outcome = 1

            model_predictions = {'business_outcome': str(predicted_outcome),
                                 'p_hat': str(predicted_probability)}

            output = model_predictions | model_inputs

            return json.dumps(output)
