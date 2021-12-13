
from utils.yaml_wrapper import YamlHandler
import argparse
from container import Container


def main(args):
    config = YamlHandler(args.settings).read_yaml()
    container = Container(config=config)
    container()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stk-conn")
    parser.add_argument("--settings", default='../configs/config.yaml')
    args = parser.parse_args()
    main(args)