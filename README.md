# README

## training

```
jupyternotebook
```

## building

### sklearn
```
docker build -t xnonr/sklearn:0.1.0 -f ./build/DockerFile.sklearn .
```

### predicting docker image
```
docker build -t xnonr/predictionapp:0.1.0 -f ./build/DockerFile.app .
```

## predicting

### run the server manually

```
uvicorn predicting.main:app --port 1313
```

### run the server automatically in docker

```
docker run -d -p 1313:1313 --name prediction-api xnonr/predictionapp:0.1.0
```

### Invokes the API
```
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' -d @testing/sample_json1.json
```
