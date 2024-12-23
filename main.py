import importlib
import argparse
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(
    description="TNVision script for the neural network")
parser.add_argument("--task_type", required=False,
                    default='anomaly', help="Type of trainning")
parser.add_argument("--model_type", required=False,
                    default='unsupervised', help="model_type to train")
parser.add_argument("--model_name", required=False,
                    default='draem', help="model to train")
parser.add_argument('--yaml_config', type=str,
                    default='configs/anomaly/cdo/bottle.yaml')
args = parser.parse_args()


def get_train_function(task_type, model_type, model_name):
    task_path = f"task.{task_type}.{model_type}.models.{model_name}"
    try:
        module = importlib.import_module(task_path)
    except ModuleNotFoundError:
        raise ValueError(f"No such module: {task_path}")
    try:
        train_fn = module.run
    except AttributeError:
        raise ValueError(f"No 'train' function in module {task_path}")
    return train_fn


def main():
    # config
    cfg = OmegaConf.load(args.yaml_config)
    train = get_train_function(
        args.task_type, args.model_type, args.model_name)
    train(cfg)


if __name__ == "__main__":
    main()
