import os
import pandas as pd
import geopandas as gpd
import rasterio as rio

import collections
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import clf_utils
import eval_utils
import data_utils

import logging
logging.basicConfig(level=logging.INFO)
SEED = 42


def _get_scalers(scalers):
    """Returns a list of scalers for hyperparameter optimization.

    Args:
        scalers (list): A list of strings indicating the scalers
            to include in the hyperparameter search space.

    Returns:
        list: A list of sclaer instances.
    """

    scalers_list = [None]

    for scaler in scalers:
        scalers_list.append(clf_utils.get_scaler(scaler))
    
    return scalers_list


def _get_pipeline(model, selector):
    """Instantiates and returns a pipeline based on
    the input configuration.

    Args:
        model (object): The model instance to include in the pipeline.
        selector (object): The selector instance to include in the pipeline.

    Returns:
        sklearn pipeline instance.
    """

    if model in clf_utils.MODELS:
        model = clf_utils.get_model(model)

    if selector in clf_utils.SELECTORS:
        selector = clf_utils.get_selector(selector)

    return Pipeline(
        [
            ("scaler", "passthrough"),
            ("selector", selector),
            ("model", model),
        ]
    )


def _get_params(scalers, model_params, selector_params):
    """Instantiates the parameter grid for hyperparameter optimization.

    Args:
        scalers (dict): A dictionary indicating the the list of scalers.
        model_params (dict): A dictionary containing the model parameters.
        selector_params (dict): A dictionary containing the feature
            selector parameters.

    Returns
        dict: Contains the parameter grid, combined into a single dictionary.
    """

    def _get_range(param):
        if param[0] == "np.linspace":
            return list(np.linspace(*param[1:]).astype(int))
        elif param[0] == "range":
            return list(range(*param[1:]))
        return param

    scalers = {"scaler": _get_scalers(scalers)}

    if model_params:
        model_params = {
            "model__" + name: _get_range(param) for name, param in model_params.items()
        }
    else:
        model_params = {}

    if selector_params:
        selector_params = {
            "selector__" + name: _get_range(param)
            for name, param in selector_params.items()
        }
    else:
        selector_params = {}

    params = [model_params, selector_params, scalers]

    return dict(collections.ChainMap(*params))


def get_cv(c):
    """Returns a model selection instance.

    Args:
        c (dict): The config dictionary indicating the model,
            selector, scalers, parameters, and model selection
            instance.

    Returns:
        object: The model selector instance.
    """

    pipe = _get_pipeline(c["model"], c["selector"])
    params = _get_params(c["scalers"], c["model_params"], c["selector_params"])
    cv, cv_params = c["cv"], c["cv_params"]

    assert cv in [
        "RandomizedSearchCV",
        "GridSearchCV",
    ]

    scoring = eval_utils.get_scoring(c["pos_class"])
    if cv == "RandomizedSearchCV":
        return RandomizedSearchCV(
            pipe, params, scoring=scoring, random_state=SEED, **cv_params
        )
    elif cv == "GridSearchCV":
        return GridSearchCV(pipe, params, scoring=scoring, **cv_params)


def model_trainer(c, data, features, target):
    """
    Trains a machine learning model using the provided data and specified features 
    to predict the target variable.

    Args:
    - c (dict): Configuration parameters for the model training.
    - data (Pandas DataFrame): Input data containing the features and target variable.
    - features (list): List of feature columns used for training the model.
    - target (str): The target variable to be predicted.

    Returns:
    - CV estimator: Trained model using cross-validation.
    """
    
    logging.info("Features: {}, Target: {}".format(features, target))

    X = data[features]
    y = data[target].values

    cv = get_cv(c)
    logging.info(cv)
    cv.fit(X, y)

    logging.info("Best estimator: {}".format(cv.best_estimator_))
    return cv


def load_data(
    config, 
    name=None,
    attributes=["rurban"],
    in_dir="clean", 
    out_dir="train",
    verbose=True
):
    cwd = os.path.dirname(os.getcwd())
    vector_dir = os.path.join(cwd, config["vectors_dir"])
    iso_codes = config["iso_codes"]
    name = iso_codes[0] if not name else name
    test_size = config["test_size"]

    filename = f"{name}_{out_dir}.geojson"
    out_file = os.path.join(cwd, vector_dir, out_dir, filename)
    if os.path.exists(out_file):
        data = gpd.read_file(out_file)
        if verbose:
            logging.info(f"Reading file {out_file}")
            print(_print_stats(data, attributes, test_size))
        return data
        
    data = []
    data_utils._makedir(os.path.dirname(out_file))
    for iso_code in iso_codes:
        in_file = f"{iso_code}_{in_dir}.geojson"
        pos_file = os.path.join(vector_dir, config["pos_class"], in_dir, in_file)
        neg_file = os.path.join(vector_dir, config["neg_class"], in_dir, in_file)
        
        pos = gpd.read_file(pos_file)
        pos["class"] = config["pos_class"]
        if "validated" in pos.columns:
            pos = pos[pos["validated"] == 0]
            
        neg = gpd.read_file(neg_file)
        neg["class"] = config["neg_class"]
        if "validated" in neg.columns:
            neg = neg[neg["validated"] == 0]
        data.append(pd.concat([pos, neg]))
    
    data = gpd.GeoDataFrame(pd.concat(data))
    data = data[(data["clean"] == 0)]
    data = _get_rurban_classification(config, data)
    data = _train_test_split(
        data, test_size=test_size, attributes=attributes, verbose=verbose
    )
    data.to_file(out_file, driver="GeoJSON")
    if verbose:
        _print_stats(data, attributes, test_size)
    
    return data


def _print_stats(data, attributes, test_size):
    total_size = len(data)
    test_size = int(total_size * test_size)
    attributes = attributes + ["class"]
    value_counts = data.groupby(attributes)[attributes[-1]].value_counts()
    value_counts = pd.DataFrame(value_counts).reset_index()
    value_counts["percentage"] = value_counts["count"]/total_size
    logging.info(f'\n{value_counts}')
        
    subcounts = pd.DataFrame(
        data.groupby(attributes + ["dataset"]).size().reset_index()
    )
    subcounts.columns = attributes + ["dataset", "count"]
    subcounts["percentage"] = (
        subcounts[subcounts.dataset == "test"]["count"] / test_size
    )
    subcounts = subcounts.set_index(attributes + ["dataset"])
    logging.info(f'\n{subcounts.to_string()}')

    subcounts = pd.DataFrame(
        data.groupby(["dataset", "class"]).size().reset_index()
    )
    subcounts.columns = ["dataset", "class", "count"]
    subcounts = subcounts.set_index(["dataset", "class"])
    
    logging.info(f'\n{subcounts.to_string()}')
    logging.info(f'\n{data.dataset.value_counts()}')
    for attribute in attributes:
        logging.info(f"\n{data[attribute].value_counts()}") 


def _get_rurban_classification(config, data):
    """
    Classifies geographical data points as 'rural' or 'urban' based on provided rasters.

    Parameters:
    - config (dict): Configuration settings including directories and file paths.
    - data (GeoDataFrame): Geographical data containing 'geometry' column with point coordinates.

    Returns:
    - GeoDataFrame: Updated geographical data with an additional 'rurban' column indicating the classification.
    """
    
    data = data.to_crs("ESRI:54009")
    coord_list = [(x, y) for x, y in zip(data["geometry"].x, data["geometry"].y)]

    cwd = os.path.dirname(os.getcwd())
    raster_dir = os.path.join(cwd, config["rasters_dir"])
    ghsl_path = os.path.join(raster_dir, "ghsl", config["ghsl_smod_file"])
    with rio.open(ghsl_path) as src:
        data["ghsl_smod"]  = [x[0] for x in src.sample(coord_list)]

    rural = [10, 11, 12, 13]
    data["rurban"] = "urban"
    data.loc[data["ghsl_smod"].isin(rural), "rurban"] = "rural"

    return data


def _train_test_split(data, test_size=0.8, attributes=["rurban"], verbose=True):
    """
    Splits the input data into training and test sets based on specified attributes.

    Args:
    - data (DataFrame): Input data to be split.
    - test_size (float): Proportion of the data to be allocated for the test set (default=0.8).
    - attributes (list): List of column names to use as attributes for splitting (default=["rurban"]).
    - verbose (bool): If True, displays information about the split (default=True).

    Returns:
    - DataFrame: Data with an additional 'dataset' column indicating whether each entry belongs to the 'TRAIN' or 'TEST' set.
    """
    
    if "dataset" in data.columns:
        return data
        
    data["dataset"] = None
    total_size = len(data)
    test_size = int(total_size * test_size)
    logging.info(f"Data dimensions: {total_size}")

    test = data.copy()
    value_counts = data.groupby(attributes)[attributes[-1]].value_counts()
    value_counts = pd.DataFrame(value_counts).reset_index()
    
    for _, row in value_counts.iterrows():
        subtest = test.copy()
        for i in range(len(attributes)):
            subtest = subtest[subtest[attributes[i]] == row[attributes[i]]]
        subtest_size = int(test_size * (row['count'] / total_size))
        if subtest_size > len(subtest):
            subtest_size = len(subtest)
        subtest_files = subtest.sample(
            subtest_size, random_state=SEED
        ).UID.values
        in_test = data["UID"].isin(subtest_files)
        data.loc[in_test, "dataset"] = "test"
    data.dataset = data.dataset.fillna("train")

    return data
    