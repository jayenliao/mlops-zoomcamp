import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    data_path_train = os.path.join(data_path, "train.pkl")
    data_path_val = os.path.join(data_path, "val.pkl")

    X_train, y_train = load_pickle(data_path_train)
    X_val, y_val = load_pickle(data_path_val)

    print("Setting the tracking URI ...", end='\t')
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    print("Done!")
    print("Setting up the experiment...", end='\t')
    mlflow.set_experiment("nyc-taxi-experiment")
    print("Done!")

    with mlflow.start_run():
        mlflow.set_tag("developer", "jay")
        mlflow.log_param("train-data-path", data_path_train)
        mlflow.log_param("val-data-path", data_path_val)

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        min_samples_split = rf.get_params()["min_samples_split"]
        mlflow.log_param("min_samples_split", min_samples_split)

        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred)**.5
        mlflow.log_metric("RMSE", rmse)

if __name__ == '__main__':
    run_train()
