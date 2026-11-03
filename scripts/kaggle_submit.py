"""Entrypoint for a Kaggle internet-disabled Notebook; all paths come from YAML."""
import argparse
from fhf.config import load_config, resolve_path
from predict import submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    conf = load_config(args.config, args.set)
    base_override = (resolve_path(conf, "inference_base_encoder")
                     if conf["paths"].get("inference_base_encoder") else None)
    submit(resolve_path(conf, "checkpoint_dir"), resolve_path(conf, "test_csv"),
           resolve_path(conf, "submission_parquet"), conf["inference"]["batch_size"],
           conf["model"]["max_length"], base_encoder_override=base_override)


if __name__ == "__main__":
    main()
