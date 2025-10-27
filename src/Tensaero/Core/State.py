# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from TerraFrame.Utilities.Time import JulianDate


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


@dataclass
class StateFrame:
    def __init__(self):
        s_bi_i: Position
        s_bi_g: Position

        v_bi_i: Velocity
        v_bi_g: Velocity

        T_GE: Transformation
        T_EI: Transformation
        T_IG: Transformation
        T_VG: Transformation

        omega_ei_i: AngularVelocity

        time: JulianDate.JulianDate
        potential_energy: float
        kinetic_energy: float
        total_energy: float

        longitude: float
        latitude: float
        altitude: float
        speed: float
        heading_angle: float
        flight_path_angle: float
