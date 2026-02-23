# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import copy
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from TerraFrame import Earth
from TerraFrame.Utilities import Conversions
from TerraFrame.Utilities.Conversions import seconds_to_days
from TerraFrame.Utilities.Time import JulianDate

from Tensaero.Core import Configuration, State, Solvers
from Tensaero.Earth import EarthState
from Tensaero.Logging.DataLogger import DataLogger
from Tensaero.SimObjects import SimObjects


class Simulator:
    def __init__(self, config_file_path: str | Path):
        self.config_file_path = config_file_path

        self.config = self._load_and_validate_config()
        self.logger = DataLogger(self.config.log_file_path)

        self._sim_objects = {}
        self._solvers = {}
        self._initialize_sim_objects()

    def run(self, time_max=None):
        jd_utc = JulianDate.julian_date_from_pydatetime(
            self.config.start_time.astimezone(ZoneInfo('UTC')))
        jd_tai = Conversions.utc_to_tai(jd_utc)
        jd_tai_start = copy.deepcopy(jd_tai)

        while True:
            ts_days = seconds_to_days(self.config.time_step)

            for sim_obj in self._sim_objects.values():
                state_frame = (
                    self._solvers[sim_obj.name].next_state(sim_obj.state,
                                                           ts_days))

                (self._sim_objects[sim_obj.name].update_state_from_state(
                    state_frame))

            self.logger.log_all()

            jd_tai += ts_days

            if (jd_tai is not None and float(
                    jd_tai - jd_tai_start) / seconds_to_days(1) >= time_max):
                self.logger.flush_buffer()
                break

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

                    self._sim_objects[entry.name].update_state(jd_tt, position,
                                                               velocity)

                    self._sim_objects[entry.name].initialize()

                    loggable_state = self._sim_objects[
                        entry.name].loggable_state()

                    for log_signal in loggable_state:
                        self.logger.register_sim_object_signal(entry.name,
                            log_signal)

                case Configuration.SimObjectTypes.general:
                    continue
                case Configuration.SimObjectTypes.ground:
                    continue
                case _:
                    raise RuntimeError(f'Unknown SimObjectType:'
                                       f' {entry.object_type}')

            match entry.solver:
                case Configuration.SolverType.default:
                    self._solvers[entry.name] = (Solvers.SolverVelocityVerlet(
                        entry.acceleration_function,
                        self._sim_objects[entry.name].new_state))

                case Configuration.SolverType.fixed:
                    self._solvers[entry.name] = (
                        Solvers.SolverFixed(entry.acceleration_function,
                                            self._sim_objects[
                                                entry.name].new_state))

                case Configuration.SolverType.euler:
                    self._solvers[entry.name] = (
                        Solvers.SolverEuler(entry.acceleration_function,
                            self._sim_objects[entry.name].new_state))

                case Configuration.SolverType.velocity_verlet:
                    self._solvers[entry.name] = (Solvers.SolverVelocityVerlet(
                        entry.acceleration_function,
                        self._sim_objects[entry.name].new_state))
