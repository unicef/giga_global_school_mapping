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

def main(iso, config):
    exp_name = f"{iso}-{config['config_name']}"
    wandb.run.name = exp_name
    results_dir = os.path.join(cwd, config["exp_dir"], exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model = embed_utils.load_model(config)
    data = model_utils.load_data(config, attributes=["rurban", "iso"], verbose=True)
    columns = ["iso", "rurban", "dataset", "class"]

    out_dir = os.path.join(config["vectors_dir"], "embeddings")
    embeddings = embed_utils.get_image_embeddings(
        config, data, model, out_dir, in_dir=None, columns=columns
    )
    embeddings.columns = [str(x) for x in embeddings.columns]
    
    test = embeddings [embeddings.dataset == "test"]
    train = embeddings[embeddings.dataset == "train"]
    logging.info(train.columns)

    logging.info(f"Test size: {test.shape}")
    logging.info(f"Train size: {train.shape}")
    
    target = "class"
    features = [str(x) for x in embeddings.columns[:-len(columns)]]
    classes = list(embeddings[target].unique())
    logging.info(f"No. of features: {len(features)}")
    logging.info(f"Classes: {classes}")

    logging.info("Training model...")
    cv = model_utils.model_trainer(c, train, features, target)
    logging.info(f"Best estimator: {cv.best_estimator_}")
    logging.info(f"Best CV score: {cv.best_score_}")

    model = cv.best_estimator_
    model.fit(train[features], train[target].values)
    preds = model.predict(test[features])

    model_file = os.path.join(results_dir, f"{iso}-{config['config_name']}.pkl")
    joblib.dump(model, model_file)

    test["pred"] = preds
    pos_class = config["pos_class"]
    results = eval_utils.save_results(test, target, pos_class, classes, results_dir)

    for rurban in ["urban", "rural"]:
        subresults_dir = os.path.join(results_dir, rurban)
        subtest = test[test.rurban == rurban]
        results = eval_utils.save_results(subtest, target, pos_class, classes, subresults_dir, rurban)
    
    if len(config["iso_codes"]) > 1:
        for iso_code in config["iso_codes"]:
            subresults_dir = os.path.join(results_dir, iso_code)
            subtest = test[test.iso == iso_code]
            results = eval_utils.save_results(
                subtest, 
                target, 
                pos_class, 
                classes, 
                subresults_dir, 
                iso_code
            )
            for rurban in ["urban", "rural"]:
                subsubresults_dir = os.path.join(subresults_dir, rurban)
                subsubtest = subtest[subtest.rurban == rurban]
                results = eval_utils.save_results(
                    subsubtest, 
                    target, 
                    pos_class, 
                    classes, 
                    subsubresults_dir, 
                    f"{iso_code}_{rurban}"
                )
            
            
if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--model_config", help="Config file")
    parser.add_argument("--iso", help="ISO code", default=[], nargs='+')
    args = parser.parse_args()

    # Load config
    config_file = os.path.join(cwd, args.model_config)
    c = config_utils.load_config(config_file)
    c["iso_codes"] = args.iso
    log_c = {
        key: val for key, val in c.items() 
        if ('url' not in key) 
        and ('dir' not in key)
        and ('file' not in key)
    }
    iso = args.iso[0]
    if "name" in c: iso = c["name"]
    log_c["iso_code"] = iso
    logging.info(log_c)
    
    wandb.init(
        project="UNICEFv1",
        config=log_c,
        tags=[c["embed_model"], c["model"]]
    )

    main(iso, c)