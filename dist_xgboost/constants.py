import os

if os.path.exists("/mnt/cluster_storage/"):  # running on Anyscale
    local_storage_path = "/mnt/cluster_storage/"
else:
    local_storage_path = "/tmp/"

preprocessor_fname = "preprocessor.pkl"
preprocessor_path = os.path.join(local_storage_path, preprocessor_fname)
model_fname = "model.ubj"  # name used by XGBoost
home_dir = os.path.expanduser("~")
model_registry = os.path.join(home_dir, "mlflow")
experiment_name = "breast_cancer_all_features"
