# README

## Prediction Model Training

```
jupyternotebook
```

## Building Docker Containers

### Base Python Scikit-Learn Docker Image
```
docker build -t xnonr/sklearn:0.1.0 -f ./build/DockerFile.sklearn .
```

### Prediction Application Docker Image
```
docker build -t xnonr/predictionapp:0.1.0 -f ./build/DockerFile.app .
```

## Predicting

### Manually Runs The Server

```
uvicorn predicting.main:app --port 1313
```

### Automatically Runs The Server Within A Docker Container

```
docker run -d -p 1313:1313 --name prediction-api xnonr/predictionapp:0.1.0
```

### Invokes the API
```
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' -d @testing/sample_json1.json
```
