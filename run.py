import argparse
import os
import shutil
from src.engine import Engine
from src.utils.util import load_log

# python run.py --config_path ./configs/mln_resnet18_voc.yml --save_dir ./ckpt

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./configs/uncertainty_mask.yml',
                        help="Path to a config")
    parser.add_argument('--save_dir', default='./ckpt_uncertainty_mask',
                        help='Path to dir to save checkpoints and logs')
    parser.add_argument('--gpu_device', default='cuda',
                        help='Device name for gpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = load_log(args.save_dir)

    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))

    engine = Engine(config_path=args.config_path, logger=logger,
                    save_dir=args.save_dir, device=args.gpu_device)
    engine.run()
