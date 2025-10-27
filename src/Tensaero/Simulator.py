from pathlib import Path
import yaml
from Tensaero.Core import Configuration


class Simulator:
    def __init__(self, config_file_path: str | Path):
        self.config_file_path = config_file_path

        self.config = self._load_and_validate_config()


    def _load_and_validate_config(self):
        conf = yaml.safe_load(open(self.config_file_path))

        # Validate config matches schema
        conf = Configuration.ConfigSchema(**conf)

        return conf