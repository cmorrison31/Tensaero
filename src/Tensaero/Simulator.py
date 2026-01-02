# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from TerraFrame import Earth
from TerraFrame.Utilities.Time import JulianDate

from Tensaero.Core import Configuration, State
from Tensaero.Earth import EarthState
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

    @staticmethod
    def _preprocess_initial_conditions(entry):
        position = State.Position.from_vector_data(
            entry.initial_conditions.position)

        velocity = State.Velocity.from_vector_data(
            entry.initial_conditions.velocity)

        return position, velocity

    def _initialize_sim_objects(self):
        # Convert the start time into a time object usable by the TerraFrame
        # library. The TerraFrame library is not timezone aware (only timescale
        # aware), so we must convert to base UTC first.
        jd_utc = JulianDate.julian_date_from_pydatetime(
            self.config.start_time.astimezone(ZoneInfo('UTC')))

        if (self.config.earth_type == Configuration.EarthType.default or
                self.config.earth_type == Configuration.EarthType.geoid):
            earth_transform = EarthState.EarthStateGeoid()
            earth = Earth.WGS84Ellipsoid()
        else:
            earth_transform = EarthState.EarthStateSphere()
            earth = Earth.SphericalEarth()

        for entry in self.config.sim_objects:
            match entry.object_type:
                case Configuration.SimObjectTypes.fixed_point:
                    self._sim_objects[entry.name] = (
                        SimObjects.FixedGroundPoint(earth, earth_transform))

                    position, velocity = (
                        self._preprocess_initial_conditions(entry))

                    self._sim_objects[entry.name].update_state(jd_utc,
                                                               position,
                                                               velocity)

                    self._sim_objects[entry.name].initialize()

                case Configuration.SimObjectTypes.general:
                    continue
                case Configuration.SimObjectTypes.ground:
                    continue
                case _:
                    raise RuntimeError(f'Unknown SimObjectType:'
                                       f' {entry.object_type}')
