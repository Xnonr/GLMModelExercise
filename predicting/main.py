# Imports Required Libraries
import pickle
import statsmodels.api as sm

from fastapi import FastAPI
from fastapi import Request

# Imported in order of function sequence
from predicting.common import transform_json_to_df
from predicting.common import batch_df

# Synchronous Method Version
from predicting.common import extract_transform_predict_df_batches

# Asynchronous Method Version
from predicting.common import async_extract_transform_predict_df_batches

# May aid with the Pickle file loading, functions however without, better safe than sorry
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
async def get_parallel_prediction(info: Request):
    '''
    Returns a single event or a list of events as a JSON message containing the business outcome, 
        probability of said outcome, along with the input variables which led to said outcome 
        in alphanumerical order for all those predictions which met the minimum standard of 75% 
        chance of a successful sale to a potential buying customer

    Keyword Arguments:
    info.json() -- Raw JSON data
    '''

    req_info = await info.json()

    df = transform_json_to_df(req_info)

    df_batches = batch_df(500, df)

    json_output_message = await async_extract_transform_predict_df_batches(
        df_batches, si, ss, final_df_column_variable_names_order,
        mdl, alphanumerically_sorted_df_column_variable_names)

    return json_output_message


@app.post('/sequential/predict')
async def get_sequential_prediction(info: Request):
    '''
    Returns a single event or a list of events as a JSON message containing the business outcome, 
        probability of said outcome, along with the input variables which led to said outcome 
        in alphanumerical order for all those predictions which met the minimum standard of 75% 
        chance of a successful sale to a potential buying customer

    Keyword Arguments:
    info.json() -- Raw JSON data
    '''

    req_info = await info.json()

    df = transform_json_to_df(req_info)

    df_batches = batch_df(500, df)

    json_output_message = extract_transform_predict_df_batches(
        df_batches, si, ss, final_df_column_variable_names_order,
        mdl, alphanumerically_sorted_df_column_variable_names)

    return json_output_message
