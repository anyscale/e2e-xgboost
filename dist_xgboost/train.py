import os
import pickle

# Enable Ray Train v2. This will be the default in an upcoming release.
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
# It is now safe to import Ray Train.

import ray
import xgboost
from ray.data.preprocessors import StandardScaler
from ray.train import Result, RunConfig, ScalingConfig
from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer

from dist_xgboost.constants import local_storage_path, preprocessor_path
from dist_xgboost.data import log_run_to_mlflow, prepare_data

NUM_WORKERS = 4
USE_GPU = True


def train_preprocessor(train_dataset: ray.data.Dataset) -> StandardScaler:
    # pick some dataset columns to scale
    columns_to_scale = [c for c in train_dataset.columns() if c != "target"]

    # Initialize the preprocessor
    preprocessor = StandardScaler(columns=columns_to_scale)
    # train the preprocessor on the training set
    preprocessor.fit(train_dataset)

    return preprocessor


def save_preprocessor(preprocessor: StandardScaler):
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)


def train_fn_per_worker(config: dict):
    """Training function that runs on each worker.

    This function:
    1. Gets the dataset shard for this worker
    2. Converts to pandas for XGBoost
    3. Separates features and labels
    4. Creates DMatrix objects
    5. Trains the model using distributed communication
    """
    # Get this worker's dataset shard convert
    train_ds, val_ds = (
        ray.train.get_dataset_shard("train"),
        ray.train.get_dataset_shard("validation"),
    )

    # Materialize the data and convert to pandas
    train_ds = train_ds.materialize().to_pandas()
    val_ds = val_ds.materialize().to_pandas()

    # Separate the labels from the features
    train_X, train_y = train_ds.drop("target", axis=1), train_ds["target"]
    eval_X, eval_y = val_ds.drop("target", axis=1), val_ds["target"]

    # Convert the data into a DMatrix
    dtrain = xgboost.DMatrix(train_X, label=train_y)
    deval = xgboost.DMatrix(eval_X, label=eval_y)

    # Do distributed data-parallel training.
    # Ray Train sets up the necessary coordinator processes and
    # environment variables for your workers to communicate with each other.
    # it also handles checkpointing via the `RayTrainReportCallback`
    _booster = xgboost.train(
        config["xgboost_params"],
        dtrain=dtrain,
        evals=[(dtrain, "train"), (deval, "validation")],
        num_boost_round=10,
        callbacks=[
            RayTrainReportCallback(
                frequency=config["checkpoint_frequency"],
                checkpoint_at_end=True,
                metrics=config["xgboost_params"]["eval_metric"],
            )
        ],
    )


def main():
    # Load and split the dataset
    train_dataset, valid_dataset, _test_dataset = prepare_data()

    # Train the preprocessor
    preprocessor = train_preprocessor(train_dataset)

    # Save the preprocessor
    save_preprocessor(preprocessor)

    train_dataset = preprocessor.transform(train_dataset)
    valid_dataset = preprocessor.transform(valid_dataset)

    run_config = RunConfig(
        ## If running in a multi-node cluster, this is where you
        ## should configure the run's persistent storage that is accessible
        ## across all worker nodes with `storage_path="s3://..."`
        storage_path=local_storage_path,
    )

    # Define the scaling config
    scaling_config = ScalingConfig(
        # Number of workers to use for data parallelism.
        num_workers=NUM_WORKERS,
        # Whether to use GPU acceleration. Set to True to schedule GPU workers.
        use_gpu=USE_GPU,
    )

    # Params that will be passed to the base XGBoost model.
    config = {
        "model_config": {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
        },
        "checkpoint_frequency": 10,
    }
    if USE_GPU:
        config["model_config"]["xgboost_params"]["device"] = "cuda"

    trainer = XGBoostTrainer(
        train_fn_per_worker,
        train_loop_config=config,
        # Register the data subsets.
        datasets={"train": train_dataset, "validation": valid_dataset},
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result: Result = trainer.fit()
    print(f"Training metrics: {result.metrics}")

    log_run_to_mlflow(config["model_config"], result, preprocessor_path)
    print("Training complete")


if __name__ == "__main__":
    main()
