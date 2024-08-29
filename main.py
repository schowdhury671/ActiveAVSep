import argparse
import logging

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import tensorflow as tf
import torch

from audio_separation.common.baseline_registry import baseline_registry
from audio_separation.config.default import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default='baselines/config/pointnav_rgb.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--val-cfg",
        type=str,
        default='baselines/config/pointnav_rgb.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
    )
    args = parser.parse_args()

    # run exp
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type)
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    #print("config is ", config.TRAINER_NAME)
    #import pdb; pdb.set_trace()

    if args.run_type == "train":
        #import pdb;pdb.set_trace()
        val_config = get_config(args.val_cfg, args.opts, args.model_dir, "eval")

        # print("==================val_config: ", val_config)

        if args.pretrain:
            trainer = trainer_init(config)
        else:
            trainer = trainer_init(config, val_config=val_config)

        level = logging.DEBUG if config.DEBUG else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S")
        trainer.train()
    elif args.run_type == "eval":
        val_config = None
        trainer = trainer_init(config, val_config=None)

        level = logging.DEBUG if config.DEBUG else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S")
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, separate_eval=True)


if __name__ == "__main__":
    main()
