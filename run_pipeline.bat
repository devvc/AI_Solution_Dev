@echo off

REM Apply all Kubernetes manifests (including volume, services, and deployments)
kubectl apply -f kubernetes_manifests/volume.yaml
kubectl apply -f kubernetes_manifests/service.yaml

REM Run Data Preprocessing
echo Running Data Preprocessing...
kubectl apply -f kubernetes_manifests/preprocessing-deployment.yaml
timeout /t 60
echo Data Preprocessing Completed.

REM Run Model Training
echo Running Model Training...
kubectl apply -f kubernetes_manifests/training-deployment.yaml
timeout /t 120
echo Model Training Completed.

REM Run Model Inference
echo Running Model Inference...
kubectl apply -f kubernetes_manifests/inference-deployment.yaml
timeout /t 60
echo Model Inference Completed.
