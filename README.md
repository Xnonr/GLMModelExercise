# README

## Prediction Model Training

The pre-trained prediction model, along with the associated SimpleImputer and StandardScalar all originate from the revised
version of the originally presented Python JupyterNotebook

```
jupyter-notebook
```

## Building Docker Containers

### Base Python Scikit-Learn Docker Image

Builds the base Docker Image contaning Python as well as all of the required libraries needed to run the prediction application.

On M1 Mac please execute the following command:

```
docker build -t xnonr/sklearn:0.1.0 -f ./build/DockerFile.sklearn .
```

On PC or other Linux based machine please execute the following command:

```
docker build -t xnonr/sklearn-amd64:0.1.0 -f ./build/DockerFile.sklearn-amd64 .
```

### Prediction Application Docker Image

Builds the Docker Image that actually contains the prediction application along with the API server.

On M1 Mac please execute the following command:

```
docker build -t xnonr/predictionapp:0.4.0 -f ./build/DockerFile.app .
```

On PC or other Linux based machine please execute the following command:

```
docker build -t xnonr/predictionapp-amd64:0.4.0 -f ./build/DockerFile.app-amd64 .
```

## Predicting

### Manually Runs The Server

```
uvicorn predicting.main:app --port 1313
```

### Automatically Runs The Server Within A Docker Container

Builds and runs the Docker Container of the prediction application API.

On M1 Mac please execute the following command:

```
docker run -d -p 1313:1313 --name prediction-api xnonr/predictionapp:0.4.0
```

On PC or other Linux based machine please execute the following command:

```
docker run -d -p 1313:1313 --name prediction-api xnonr/predictionapp-amd64:0.4.0
```

Alternatively you can run the ultilty script, but please note you must edit the script via comments in order to run the script on Mac M1.

``` 
run_api.sh
```

### Invokes the API

Tests the prediction application API on variously sized raw JSON data files taken from the originally presented testing data '.csv' file.

Tests 1 Row of JSON in an unlisted format
```
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' -d @testing/sample_raw_json_1_row_v1.json
```

Tests 1 Row of JSON in a listed format
```
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' -d @testing/sample_raw_json_1_row_v2.json
```

Tests 10 Rows of JSON in a listed format
```
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' -d @testing/sample_raw_json_10_rows.json
```

Tests 100 Rows of JSON in a listed format
```
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' -d @testing/sample_raw_json_100_rows.json
```

Tests 1000 Rows of JSON in a listed format
```
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' -d @testing/sample_raw_json_1000_rows.json
```

Tests 10000 Rows of JSON in a listed format
```
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' -d @testing/sample_raw_json_10000_rows.json
```
