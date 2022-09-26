# Sets the path so that 'kind' may be found by the terminal
export PATH=$PATH:$HOME/go/bin

# Initalizes the variables
appimagename="predictionapp-unknown"
appimageversion="0.6.0"

# EXTREMELY SENSITIVE
# Switch statement determines the current machine's name and the operating system it is making use of
case $(uname -m) in
    x86_64) appimagename="predictionapp-amd64" ;;
    arm64) appimagename="predictionapp" ;;
esac

# Creates a kind cluster in the form of a Docker container
#kind create cluster --name predictingcluster

# Creates a kind cluster according to a specific configuration in the form of a Docker container
kind create cluster --config deploy/kind_cluster_ingress.yaml

# Retrieves details about the kind Docker container
kubectl get all -A

# Prepares the Kubernetes cluster by activating Ingress nginx within the Kind cluster
kubectl apply --wait=true -f deploy/ingress_impementation.yaml

# Brute force stopgap waiting for the Ingress to properly set itself up
sleep 60

kubectl get all -A

# Retrieves a Docker Image and loads it into the kind Docker Container, a Docker image within a Docker container
docker pull xnonr/${appimagename}:${appimageversion}
kind load docker-image xnonr/${appimagename}:${appimageversion} --name predictingcluster

# Enters the Docker container itself and opens up a shell command line within
#docker exec -ti predictingcluster-control-plane bash

# Shows those images loaded within the Docker container
#crictl images

# Leaves and exits out of the Docker container
#exit

# Creates and loads the deployment object within kubernetes kind, which itself then deploys the kubernetes pods with intent basis,
# as well as the service object used to resolved deployed pods shifting IP addresses
# Shell script waits that deployment is complete before proceeding to avoid loading errors when going to fast
kubectl apply --wait=true -f deploy/predictionapp_ingress.yaml
kubectl wait pod -l app=predictionapp --for condition=Ready --timeout=30s
kubectl wait deployment predictionapp-deployment --for condition=Available=True --timeout=30s

# Play it safe, as it is difficult to check when the kuberentes service is fully up, ready and available
# Otherwise the script tries to move forward before then and fails
echo "Waiting for kubernetes service to become available."
sleep 10

# Displays the status of the kubernetes deployment object
#kubectl get deployment predictionapp-deployment

# Displays and describes the details of the deployment and service object
#kubectl get deployment predictionapp-deployment -o yaml
#kubectl describe deployment predictionapp-deployment

# Pay attention to the endpoints, which indicates all relevant pods IP addresses handled by the service object
#kubectl describe service predictionapp-service

# Displays the status of the kubernetes pod created by the kubernetes deployment object previously
# kubectl get pod predictionapp-deployment-5c4497bf75-nrzql

# Displays all of those pods with Docker containers running a specific Docker image, similar to an SQL select statement
#kubectl get pod -l app=predictionapp

# Runs the Python test script for the Python prediction application that should now be running within a Docker container within one of the pods
./testing/testing.py

# As long as the kubernetes deploymnet object exists, pods will keep being created as indicated if any halt or are deleted
# The only way to shut down all pods is to delete the kubernetes deployment object itself to stop it from notifying the scheduling process
kubectl delete --wait=true -f deploy/predictionapp_ingress.yaml

# Deletes the kind Docker container properly, avoid doing so via the Docker Desktop GUI
kind delete cluster --name predictingcluster

