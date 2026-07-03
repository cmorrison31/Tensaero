# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import copy
from abc import ABC, abstractmethod

from TerraFrame.Utilities.Time import JulianDate

from Tensaero.Core import State
from Tensaero.Core.State import StateFrame
from Tensaero.Logging.DataLogger import LogSignalSpecification

import numpy as np


class BaseObject(ABC):
    def __init__(self, name, earth, earth_transform):
        self.name = name
        self.state = State.StateFrame()
        self.earth = earth
        self.earth_transform = earth_transform

    @abstractmethod
    def loggable_state(self):
        # TODO: Add a function to classes like position and velocity that
        #  tell the logger what their attributes are and what their size and
        #  type is. Fall back to numpy info if this function isn't provided
        #  by a class.
        l_state  = [
            LogSignalSpecification("time", lambda: float(self.state.time),
                                   'state'),

            LogSignalSpecification("s_bi_i", lambda: self.state.s_bi_i.data,
                                   'state'),
            LogSignalSpecification("v_bi_i", lambda: self.state.v_bi_i.data,
                                   'state'),

            LogSignalSpecification("T_GE", lambda:
            self.state.T_GE.data.flatten(),
                                   'state'),
            LogSignalSpecification("T_EI", lambda: self.state.T_EI.data,
                                   'state'),
            LogSignalSpecification("T_IG", lambda: self.state.T_IG.data,
                                   'state'),
            LogSignalSpecification("T_VG", lambda: self.state.T_VG.data,
                                   'state'),

            LogSignalSpecification("omega_ei_i",
                                   lambda: self.state.omega_ei_i.data,
                                   'state'),

            LogSignalSpecification("longitude", lambda: self.state.longitude,
                                   'state'),
            LogSignalSpecification("latitude", lambda: self.state.latitude,
                                   'state'),
            LogSignalSpecification("altitude", lambda: self.state.altitude,
                                   'state'),
            LogSignalSpecification("heading_angle",
                                   lambda: self.state.heading_angle, 'state'),
            LogSignalSpecification("flight_path_angle",
                                   lambda: self.state.flight_path_angle,
                                   'state'),
                ]

        return l_state

    def new_state(self, time: JulianDate.JulianDate, position: State.Position,
                     velocity: State.Velocity):
        position = self._vector_to_cartesian(position)
        velocity = self._vector_to_cartesian(velocity)

        position = self._position_to_inertial_frame(time, position)
        velocity = self._velocity_to_inertial_frame(time, velocity, position)

        state_frame = State.StateFrame()

        state_frame.time = time

        state_frame.T_EI = self.earth_transform.transformation_matrix(time)
        state_frame.omega_ei_i = self.earth_transform.angular_velocity(time)

        state_frame.s_bi_i = position
        state_frame.v_bi_i = velocity

        s_bi_e = state_frame.T_EI @ state_frame.s_bi_i

        lat, lon, alt = self.earth.lat_lon_alt_from_cartesian(*s_bi_e)

        state_frame.longitude = lon
        state_frame.latitude = lat
        state_frame.altitude = alt

        data = np.array(((-np.sin(lat) * np.cos(lon),
                   -np.sin(lat) * np.sin(lon), np.cos(lat)),
                  (-np.sin(lon), np.cos(lon), 0.0),
                  (-np.cos(lat) * np.cos(lon),
                   -np.cos(lat) * np.sin(lon), -np.sin(lat))),
                 dtype='float')

        state_frame.T_GE = (
            State.Transformation(data,
                                 State.ReferenceFrames.EarthCenteredEarthFixed,
                                 State.ReferenceFrames.Geographic))

        state_frame.T_IG = state_frame.T_EI.T @ state_frame.T_GE.T

        v_bi_g = state_frame.T_IG.T @ (
                state_frame.v_bi_i -
                State.Velocity.from_vector_data(state_frame.omega_ei_i @
                state_frame.s_bi_i))

        heading_angle, flight_path_angle = (
            self.flight_path_angles_from_geographic(v_bi_g))

        state_frame.heading_angle = heading_angle
        state_frame.flight_path_angle = flight_path_angle

        chi = heading_angle
        gamma = flight_path_angle

        data = np.array(
            ((np.cos(gamma) * np.cos(chi), np.cos(gamma) * np.sin(chi),
              -np.sin(gamma)), (-np.sin(chi), np.cos(chi), 0.0),
             (np.sin(gamma) * np.cos(chi), np.sin(gamma) * np.sin(chi),
              np.cos(gamma))), dtype='float')

        state_frame.T_VG = (
            State.Transformation(data,
                                State.ReferenceFrames.Geographic,
                                State.ReferenceFrames.FlightPath))

        return state_frame

    def update_state_from_state(self, state_frame: StateFrame):
        self.state = state_frame

    @abstractmethod
    def update_state(self, time: JulianDate.JulianDate, position: State.Position,
                     velocity: State.Velocity):
        self.state = self.new_state(time, position, velocity)

    @abstractmethod
    def initialize(self):
        pass

    def _vector_to_cartesian(self, vector):
        """
        Takes the input vector and applies necessary coordinate
        transformations to ensure the data is in cartesian coordinates. The
        coordinate transformations use the given earth model for LLA -> xyx
        or LLR -> xyz transformations.

        Note that this function does not adjust the reference frame, only the
        coordinate system used to express values in the reference frame.

        :param vector: Input vector to apply the coordinate transform to.
        :return: Coordinate transformed vector.
        :rtype: State.Vector | State.Velocity | State.Position
        """

        if vector.coordinate_system == State.CoordinateSystems.WGS84:
            new_vec = self.earth.cartesian_from_lat_lon_alt(*vector)
            vector.data = new_vec
            vector.coordinate_system = State.CoordinateSystems.Cartesian

        elif vector.coordinate_system == State.CoordinateSystems.Spherical:
            new_vec = (self.earth.
                       cartesian_from_geocentric_lat_lon_radius(*vector))
            vector.data = new_vec
            vector.coordinate_system = State.CoordinateSystems.Cartesian

        return vector

    def _position_to_inertial_frame(self, jd_utc: JulianDate.JulianDate,
                                    position: State.Position):

        t_ei = self.earth_transform.transformation_matrix(jd_utc)

        if (position.reference_frame ==
                State.ReferenceFrames.EarthCenteredEarthFixed):
            pos_new = t_ei.T @ position
        elif (position.reference_frame ==
              State.ReferenceFrames.EarthCenteredInertial):
            pos_new = position
        else:
            raise RuntimeError(f'Unsupported reference frame "'
                               f'{position.reference_frame}" in position to '
                               f'inertial frame transformation.')

        return pos_new

    def _velocity_to_inertial_frame(self, jd_utc: JulianDate.JulianDate,
                                    velocity: State.Velocity,
                                    position: State.Position):

        assert (position.reference_frame ==
                State.ReferenceFrames.EarthCenteredInertial)

        omega_ei_i = self.earth_transform.angular_velocity(jd_utc)

        if (velocity.reference_frame ==
                State.ReferenceFrames.EarthCenteredEarthFixed):
            pos_tmp = copy.deepcopy(position)
            pos_tmp.reference_frame = velocity.reference_frame
            v_tmp = omega_ei_i @ pos_tmp
            v_tmp = State.Velocity.from_vector_data(v_tmp)
            v_bi_i = velocity + v_tmp
            v_bi_i.reference_frame = (
                State.ReferenceFrames.EarthCenteredInertial)
        elif (velocity.reference_frame ==
              State.ReferenceFrames.EarthCenteredInertial):
            v_bi_i = velocity
        else:
            raise RuntimeError(f'Unsupported reference frame "'
                               f'{velocity.reference_frame}" in position to '
                               f'inertial frame transformation.')

        return v_bi_i

    @staticmethod
    def flight_path_angles_from_geographic(v_bi_g):
        heading_angle = np.atan2(v_bi_g[1], v_bi_g[0])

        flight_path_angle = 0

        denominator = np.sqrt(v_bi_g[0] ** 2 + v_bi_g[1] ** 2)
        if denominator > 0.0:
            flight_path_angle = np.atan2(-v_bi_g[2], denominator)
        else:
            if v_bi_g[2] > 0:
                flight_path_angle = -np.pi / 2.0
            if v_bi_g[2] < 0:
                flight_path_angle = np.pi / 2.0
            if v_bi_g[2] == 0:
                flight_path_angle = 0.0

        return heading_angle, flight_path_angle


class FixedGroundPoint(BaseObject):
    def __init__(self, name, earth, earth_transform):
        super().__init__(name, earth, earth_transform)

    def update_state(self, time, position, velocity):
        super().update_state(time, position, velocity)

    def initialize(self):
        # Nothing required for a fixed ground point
        pass

    def loggable_state(self):
        # Fixed ground point only needs to call the base class function
        return super().loggable_state()