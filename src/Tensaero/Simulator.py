# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path
import yaml
from Tensaero.Core import Configuration
from Tensaero.SimObjects import SimObjects


class Simulator:
    def __init__(self, config_file_path: str | Path):
        self.config_file_path = config_file_path

        self.config = self._load_and_validate_config()

        self._sim_objects = {}

        self._initialize_sim_objects()


    def _load_and_validate_config(self):
        conf = yaml.safe_load(open(self.config_file_path))

        # Validate config matches schema
        conf = Configuration.ConfigSchema(**conf)

        return conf


    def _initialize_sim_objects(self):
        for entry in self.config.sim_objects:
            match entry.object_type:
                case Configuration.SimObjectTypes.fixed_point:
                    self._sim_objects[entry.name] = (
                        SimObjects.FixedGroundPoint())
                    # TODO: initialize object state
                case Configuration.SimObjectTypes.general:
                    continue
                case Configuration.SimObjectTypes.ground:
                    continue
                case _:
                    raise RuntimeError(f'Unknown SimObjectType:'
                                       f' {entry.object_type}')

