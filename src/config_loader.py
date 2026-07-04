
import yaml

def load_config(path: str = "config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config
