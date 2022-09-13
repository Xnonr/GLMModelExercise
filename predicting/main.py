# Imports Required Libraries
import json
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm

from fastapi import FastAPI
from fastapi import Request

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Global Variables

# Properly filled in upon the application's startup
mdl = None
si = None
ss = None

# List of the already properly ordered column variables required by the pre-trained
#    model in order for it to carry out accurate predictions
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
def load_models():
    '''
    Loads in pre-trained Prediction, SimpleImputer and StandardScalar models
    '''
    global mdl
    global si
    global ss

    mdl = sm.load('models/glm_final_model.pickle')
    si = pickle.load(open('models/glm_simple_imputer.pickle', 'rb'))
    ss = pickle.load(open('models/glm_standard_scalar.pickle', 'rb'))


@app.post('/predict')
async def get_prediction(info: Request):
    '''
    Returns a single event or a list of events as a JSON message containing the business outcome, 
        probability of said outcome, along with the input variables which led to said outcome 
        in alphanumerical order for all those predictions which met the minimum standard of 75% 
        chance of a successful sale to a potential buying customer

    Keyword Arguments:
    info.json() -- Raw JSON data
    '''

    req_info = await info.json()

    df = extract_transform_input_data_pipeline(
        req_info,
        si,
        ss,
        final_df_column_variable_names_order)

    json_output_message = predict_outcome(
        df,
        mdl,
        alphanumerically_sorted_df_column_variable_names)

    return json_output_message


def transform_json_to_df(json_str):
    '''
    Returns a DataFrame containing raw JSON data

    Keyword Arguments:
    json_str -- Raw JSON data
    '''

    obj = json.loads(json_str)

    if isinstance(obj, list):
        df = pd.read_json(json_str, orient='records')

    else:
        json_str = '[' + json_str + ']'
        df = pd.read_json(json_str, orient='records')

    return df


def format_df_column_variables(df):
    '''
    Returns a DataFrame after having transformed those columns, whose values 
        consisted of those of data types String and represented quantitative 
        data, into values being that of the Float data type after removing any
        non mathematical symbols 

    Keyword Arguments:
    df -- A DataFrame containing raw JSON data
    '''

    # Formats the 'x12' and 'x63' columns', consisting of the data type String,
    #    and respectively representing first monetary then percentage values,
    #    into the data type Float so as to be able to latter on apply mathermatical
    #    work upon said columns' values later on

    df['x12'] = df['x12'].str.replace('$', '')
    df['x12'] = df['x12'].str.replace(',', '')
    df['x12'] = df['x12'].str.replace(')', '')
    df['x12'] = df['x12'].str.replace('(', '-')
    df['x12'] = df['x12'].astype(float)

    df['x63'] = df['x63'].str.replace('%', '')
    df['x63'] = df['x63'].astype(float)

    return df


def impute_missing_df_data(si, df):
    '''
    Returns a DataFrame with no column variables whose data is of a qualitative nature, 
        as well as having filled in any remaining blank, NaN or NULL or otherwise missing 
        values via the usage of an imported, pre-trained SimpleImputer using a mean based 
        strategy

    Keyword Arguments:
    si -- A pre-trained, imported SimpleImputer
    df -- A formatted DataFrame
    '''

    df = pd.DataFrame(si.transform(df.drop(columns=['x5', 'x31', 'x81', 'x82'])),
                      columns=df.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)

    return df


def scale_df_data(ss, df):
    '''
    Returns a DataFrame whose column variable values have all been scaled via a 
        standardization method for the purpose of feature scaling, utilizing an
        imported. pre-trained StandardScalar

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


def filter_df_column_variables(ordr_clmn_names_lst, df):
    '''
    Returns a DataFrame containing only those column variables required by the pre-trained model
        for predictions, filtering out drom the given DataFrame only said columns

    Keyword Arguments:
    ordr_clmn_names_lst -- A list of DataFrame column variable names required by the pre-trained model
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

    df = df[ordr_clmn_names_lst].copy(deep=True)

    return df


def extract_transform_input_data_pipeline(json, si, ss, ordr_clmn_names_lst):
    '''
    Returns a DataFrame created from the passed in, raw JSON data, transforming said data via
        imputation, to fill in any and all missing values, scaling, and the creation of dummy
        variables for those qualitative column variables which require such action

    Keyword Arguments:
    json -- Raw JSON data
    si -- A pre-trained, imported SimpleImputer
    ss -- A pre-trained, imported StandardScalar
    ordr_clmn_names_lst -- A list of DataFrame column variable names required by the pre-trained model
    '''

    df = transform_json_to_df(json)

    if df.shape[0] < 1:
        df = pd.DataFrame()

    else:
        df = format_df_column_variables(df)

        imputed_df = impute_missing_df_data(si, df)

        scaled_imputed_df = scale_df_data(ss, imputed_df)

        df = create_df_dummy_column_variables_old(df, scaled_imputed_df)

        df = filter_df_column_variables(ordr_clmn_names_lst, df)

    return df


def predict_outcome(df, mdl, alphanum_ord_clmn_var_names_lst):
    '''
    Returns a JSON message containing either the model's predicted outcomes, 
        marked as 'business_outcome', predicted probability, marked as 'p_hat',
        and the inputs in the alphanumerical order of their variables' names or,
        should the predicted probability be under that of 75%, a message indicating
        as such; if no valid JSON data as far as the application can tell is passed in, 
        then a JSON encoded error message will be returned instead

    Keyword Arguments:
    df -- A DataFrame containing only those 25 column variables required by the pre-trained model
    mdl -- A pre-trained, imported prediction model
    alphanum_ord_clmn_var_names_lst -- A list of DataFrame column variable names required by 
                                       the pre-trained model order alphanumerically
    '''

    num_rows_df = df.shape[0]

    if num_rows_df == 0:

        return json.dumps({'message': 'ERROR - No valid JSON data available for prediction.'})

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
                pass

        return json.dumps(output_msgs_lst)
