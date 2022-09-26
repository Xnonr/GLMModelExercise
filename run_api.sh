# Initalizes the variables
appimagename="predictionapp-unknown"
appimageversion="0.6.0"

# EXTREMELY SENSITIVE
# Switch statement determines the current machine's name and the operating system it is making use of
case $(uname -m) in
    x86_64) appimagename="predictionapp-amd64" ;;
    arm64) appimagename="predictionapp" ;;
esac

# Variables are plugged in automatically for greater adaptability
docker run -d -p 1313:1313 --name prediction-api xnonr/${appimagename}:${appimageversion}
