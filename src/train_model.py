import os
import argparse
import joblib
import pandas as pd
import logging
import torch

import sys
sys.path.insert(0, "../utils/")
import data_utils
import config_utils
import model_utils
import embed_utils
import eval_utils
import wandb

cwd = os.path.dirname(os.getcwd())


def main(config):
    exp_name = config['config_name']
    wandb.run.name = exp_name
    results_dir = os.path.join(cwd, config["exp_dir"], exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    model = torch.hub.load("facebookresearch/dinov2", config["embed_model"])
    model.name = config["embed_model"]
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)

    data = model_utils.load_data(config, attributes=["rurban", "iso"], verbose=True)
    embeddings = embed_utils.compute_embeddings(config, data, model)
    embeddings.columns = [str(x) for x in embeddings.columns]
    
    test = embeddings [embeddings.dataset == "test"]
    train = embeddings[embeddings.dataset == "train"]
    logging.info(train.columns)

    logging.info(f"Test size: {test.shape}")
    logging.info(f"Train size: {train.shape}")
    
    target = "class"
    features = [str(x) for x in embeddings.columns[1:-2]]
    classes = list(embeddings[target].unique())
    logging.info(f"No. of features: {len(features)}")
    logging.info(f"Classes: {classes}")

    cv = model_utils.model_trainer(c, train, features, target)
    logging.info(cv.best_estimator_)
    logging.info(cv.best_score_)

    model = cv.best_estimator_
    model.fit(train[features], train[target].values)
    preds = model.predict(test[features])

    model_file = os.path.join(results_dir, f"{config['config_name']}.pkl")
    joblib.dump(model, model_file)

    results = eval_utils.evaluate(test[target], preds, config["pos_class"])
    cm = eval_utils.get_confusion_matrix(test[target], preds, classes)
    eval_utils.save_results(results, cm, results_dir)
    wandb.log(results)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--model_config", help="Config file")
    args = parser.parse_args()

    # Load config
    config_file = os.path.join(cwd, args.model_config)
    c = config_utils.load_config(config_file)
    log_c = {
        key: val for key, val in c.items() 
        if ('url' not in key) and ('dir' not in key)
    }
    iso_code = c["iso_codes"][0]
    if "name" in c: iso_code = c["name"]
    log_c["iso_code"] = iso_code
    logging.info(log_c)
    
    wandb.init(
        project="UNICEFv1",
        config=log_c,
        tags=[c["embed_model"], c["model"]]
    )

    main(c)