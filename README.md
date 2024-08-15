# First, clone this repository:

# Clone this repository
git clone https://github.com/devvc/AI_Solution_Dev.git
cd AI_Solution_Dev

# Secondly, prepare to run:

# Open command prompt on your computer. (Best to run as administratr.)
# Type in the command below: (Remember to have your docker running in the background)
minikube start 
minikube status # Check for status ensure minikube is running correctly.

# Mount volume, keep the cmd running
minikube mount "path/to/volume/directory":/mnt/data #(Ensure the volume directory contains the customer_dataset.csv)

# Open a new cmd, make sure to cd to the directory same as initial step, make sure you give time for each deployments to run.
run_pipeline.bat

# Thirdly, to check for the application: (Keep this running so that you can access to the website)
minikube service model-inference-service

# Lastly, make sure to clean up so it doesn't make your computer laggy.
kubectl delete -f kubernetes_manifests/
minikube stop
