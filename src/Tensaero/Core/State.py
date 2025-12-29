# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC
from dataclasses import dataclass
from enum import Enum
import copy

import numpy as np
import numpy.typing as npt
from TerraFrame.Utilities.Time import JulianDate


class CoordinateSystems(Enum):
    Cartesian = 0
    Spherical = 1
    WGS84 = 2

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.name.lower() == value:
                return member
        return None


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
    coordinate_system: CoordinateSystems

    def __init__(self, data, reference_frame,
                 coordinate_system=CoordinateSystems.Cartesian):
        self.data = data
        self.reference_frame = reference_frame
        self.coordinate_system = coordinate_system

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other):
        if self.reference_frame != other.reference_frame:
            raise RuntimeError(f'Reference frame mismatch during vector '
                               f'addition. First reference frame: '
                               f'{self.reference_frame}, second reference '
                               f'frame: {other.reference_frame}.')

        rvalue = copy.deepcopy(other)
        rvalue.data += self.data

        return rvalue

    def __sub__(self, other):
        if self.reference_frame != other.reference_frame:
            raise RuntimeError(f'Reference frame mismatch during vector '
                               f'addition. First reference frame: '
                               f'{self.reference_frame}, second reference '
                               f'frame: {other.reference_frame}.')

        rvalue = copy.deepcopy(other)
        rvalue.data -= self.data

        return rvalue


class Matrix(ABC):
    data: npt.NDArray[np.float64]
    reference_frame_from: ReferenceFrames
    reference_frame_to: ReferenceFrames

    def __init__(self, data, reference_frame_from, reference_frame_to):
        self.data = data
        self.reference_frame_from = reference_frame_from
        self.reference_frame_to = reference_frame_to


class Velocity(Vector):
    def __init__(self, data, reference_frame,
                 coordinate_system=CoordinateSystems.Cartesian):
        super().__init__(data, reference_frame, coordinate_system)

    @staticmethod
    def from_vector_data(vec_data):
        vec = Velocity(vec_data.data, vec_data.reference_frame,
                       vec_data.coordinate_system)

        return vec


class Position(Vector):
    def __init__(self, data, reference_frame,
                 coordinate_system=CoordinateSystems.Cartesian):
        super().__init__(data, reference_frame, coordinate_system)

    @staticmethod
    def from_vector_data(vec_data):
        vec = Position(vec_data.data, vec_data.reference_frame,
                       vec_data.coordinate_system)

        return vec


class Transformation(Matrix):
    def __init__(self, data, reference_frame_from, reference_frame_to):
        super().__init__(data, reference_frame_from, reference_frame_to)

    # noinspection PyPep8Naming
    @property
    def T(self):
        return Transformation(self.data.T, self.reference_frame_to,
                              self.reference_frame_from)

    def __matmul__(self, other):
        if isinstance(other, Vector):
            data = self.data @ other.data

            rvalue = copy.deepcopy(other)
            rvalue.data = data
            rvalue.reference_frame =  self.reference_frame_to

            return rvalue

        elif isinstance(other, Transformation):
            data = self.data @ other.data

            rvalue = copy.deepcopy(other)
            rvalue.data = data
            rvalue.reference_frame_to = self.reference_frame_to
            rvalue.reference_frame_from = other.reference_frame_from

            return rvalue
        else:
            raise RuntimeError(f'Unsupported type "{type(other)}" for '
                               f'Transformation matrix multiplication.')


class AngularVelocity(Matrix):
    def __init__(self, data, reference_frame_from, reference_frame_to):
        super().__init__(data, reference_frame_from, reference_frame_to)

    # noinspection PyPep8Naming
    @property
    def T(self):
        return AngularVelocity(self.data.T, self.reference_frame_to,
                              self.reference_frame_from)

    def __matmul__(self, other):
        if isinstance(other, Position):
            data = self.data @ other.data

            rvalue = copy.deepcopy(other)
            rvalue.data = data
            rvalue.reference_frame =  self.reference_frame_to

            return rvalue

        elif isinstance(other, AngularVelocity):
            data = self.data @ other.data

            rvalue = copy.deepcopy(other)
            rvalue.data = data

            return rvalue
        else:
            raise RuntimeError(f'Unsupported type "{type(other)}" for '
                               f'Transformation matrix multiplication.')


@dataclass
class StateFrame:
    def __init__(self):
        s_bi_i: Position

        v_bi_i: Velocity

        T_GE: Transformation
        T_EI: Transformation
        T_IG: Transformation
        T_VG: Transformation

        omega_ei_i: AngularVelocity

        time: JulianDate.JulianDate

        longitude: float
        latitude: float
        altitude: float
        heading_angle: float
        flight_path_angle: float
