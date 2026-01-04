# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import copy
from pathlib import Path
from zoneinfo import ZoneInfo
import h5py

import yaml
from TerraFrame import Earth
from TerraFrame.Utilities.Time import JulianDate
from TerraFrame.Utilities import Conversions

from Tensaero.Core import Configuration, State, Solvers
from Tensaero.Earth import EarthState
from Tensaero.SimObjects import SimObjects


class Simulator:
    def __init__(self, config_file_path: str | Path):
        self.config_file_path = config_file_path
        self._log_file: None | h5py.File = None

        self.config = self._load_and_validate_config()

        self._setup_log_file()

        self._sim_objects = {}
        self._solvers = {}
        self._initialize_sim_objects()

    def run(self, time_max=None):
        jd_utc = JulianDate.julian_date_from_pydatetime(
            self.config.start_time.astimezone(ZoneInfo('UTC')))
        jd_tt = Conversions.any_to_tt(jd_utc)
        jd_tt_start = copy.deepcopy(jd_tt)

        while True:
            for sim_obj in self._sim_objects.values():
                state_frame = (self._solvers[sim_obj.name].
                next_state(sim_obj.state, self.config.time_step))

                (self._sim_objects[sim_obj.name].
                 update_state_from_state(state_frame))

            jd_tt += self.config.time_step

            if jd_tt is not None and float(jd_tt - jd_tt_start) >= time_max:
                break


    def _load_and_validate_config(self):
        conf = yaml.safe_load(open(self.config_file_path))

        # Validate config matches schema
        conf = Configuration.ConfigSchema(**conf)

        return conf

    def _setup_log_file(self):
        log_file_path: Path = self.config.log_file_path

        if not log_file_path.suffix == '.hdf5':
            log_file_path = log_file_path.with_suffix('.hdf5')

        log_file_path = log_file_path.absolute()

        self._log_file = h5py.File(log_file_path, "w")

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
        jd_tt = Conversions.any_to_tt(jd_utc)

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
                        SimObjects.FixedGroundPoint(entry.name, earth,
                                                    earth_transform))

                    position, velocity = (
                        self._preprocess_initial_conditions(entry))

                    self._sim_objects[entry.name].update_state(jd_tt,
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

            match entry.solver:
                case Configuration.SolverType.default:
                    self._solvers[entry.name] = (
                        Solvers.SolverVelocityVerlet(
                            entry.acceleration_function, self._sim_objects[
                        entry.name].new_state))

                case Configuration.SolverType.fixed:
                    self._solvers[entry.name] = (
                        Solvers.SolverFixed(entry.acceleration_function,
                            self._sim_objects[entry.name].new_state))

                case Configuration.SolverType.euler:
                    self._solvers[entry.name] = (
                        Solvers.SolverEuler(
                            entry.acceleration_function,
                            self._sim_objects[entry.name].new_state))

                case Configuration.SolverType.velocity_verlet:
                    self._solvers[entry.name] = (
                        Solvers.SolverVelocityVerlet(
                            entry.acceleration_function,
                            self._sim_objects[entry.name].new_state))
