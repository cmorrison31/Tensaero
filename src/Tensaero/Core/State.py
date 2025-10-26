# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC
from enum import Enum

import numpy as np
import numpy.typing as npt


class ReferenceFrames(Enum):
    EarthCenteredInertial = 0
    EarthCenteredEarthFixed = 1
    Geographic = 2
    Body = 3
    Wind = 4
    FlightPath = 6

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.name.lower() == value:
                return member
        return None

class Vector(ABC):
    data: npt.NDArray[np.float64]
    reference_frames: ReferenceFrames

    def __init__(self, data, reference_frame):
        self.data = data
        self.reference_frame = reference_frame


class Matrix(ABC):
    data: npt.NDArray[np.float64]
    reference_frame_from: ReferenceFrames
    reference_frame_to: ReferenceFrames

    def __init__(self, data, reference_frame_from, reference_frame_to):
        self.data = data
        self.reference_frame_from = reference_frame_from
        self.reference_frame_to = reference_frame_to


class Velocity(Vector):
    def __init__(self, data, reference_frame):
        super().__init__(data, reference_frame)


class Position(Vector):
    def __init__(self, data, reference_frame):
        super().__init__(data, reference_frame)


class Transformation(Matrix):
    def __init__(self, data, reference_frame_from, reference_frame_to):
        super().__init__(data, reference_frame_from, reference_frame_to)


class AngularVelocity(Matrix):
    def __init__(self, data, reference_frame_from, reference_frame_to):
        super().__init__(data, reference_frame_from, reference_frame_to)


class StateFrame:
    def __init__(self):
        self.s_bi_i = Position(np.nan * np.zeros((3, 1)),
                               ReferenceFrames.EarthCenteredInertial)
        self.s_bi_g = Position(np.nan * np.zeros((3, 1)),
                               ReferenceFrames.Geographic)

        self.v_bi_i = Velocity(np.nan * np.zeros((3, 1)),
                               ReferenceFrames.EarthCenteredInertial)
        self.v_bi_g = Velocity(np.nan * np.zeros((3, 1)),
                               ReferenceFrames.Geographic)

        self.T_GE = Transformation(np.nan * np.zeros((3, 1)),
                                     ReferenceFrames.EarthCenteredEarthFixed,
                                     ReferenceFrames.Geographic)
        self.T_EI = Transformation(np.nan * np.zeros((3, 1)),
                                     ReferenceFrames.EarthCenteredInertial,
                                     ReferenceFrames.EarthCenteredEarthFixed)
        self.T_IG = Transformation(np.nan * np.zeros((3, 1)),
                                     ReferenceFrames.Geographic,
                                     ReferenceFrames.EarthCenteredInertial)
        self.T_VG = Transformation(np.nan * np.zeros((3, 1)),
                                     ReferenceFrames.Geographic,
                                     ReferenceFrames.FlightPath)

        self.omega_ei_i = Transformation(np.nan * np.zeros((3, 1)),
                                     ReferenceFrames.EarthCenteredInertial,
                                     ReferenceFrames.EarthCenteredEarthFixed)

        self.time = np.nan
        self.potential_energy = np.nan
        self.kinetic_energy = np.nan
        self.total_energy = np.nan

        self.longitude = np.nan
        self.latitude = np.nan
        self.altitude = np.nan
        self.speed = np.nan
        self.heading_angle = np.nan
        self.flight_path_angle = np.nan
