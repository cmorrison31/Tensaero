# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod

from TerraFrame.CelestialTerrestrial import (
    CelestialTerrestrialTransformation as CelTel)
from TerraFrame.Utilities import Time, Conversions, TransformationMatrices

from Tensaero.Core import State
from Tensaero.Core.State import ReferenceFrames as RefFra
from Tensaero.Utilities import Cache


class EarthStateBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transformation_matrix(self, time):
        pass

    def angular_velocity(self, time):
        pass


class EarthStateSphere(EarthStateBase):
    def __init__(self):
        super().__init__()

    def transformation_matrix(self, time):
        """
        :type time: Time.JulianDate.JulianDate
        """

        time_tt = Conversions.any_to_tt(time)
        time_ut1 = Conversions.tt_to_ut1(time_tt)

        t_ei = TransformationMatrices.earth_rotation_matrix(time_ut1)

        t_ei = State.Transformation(t_ei, RefFra.EarthCenteredInertial,
                                    RefFra.EarthCenteredEarthFixed)

        return t_ei

    def angular_velocity(self, time):
        """
        :type time: Time.JulianDate.JulianDate
        """

        time_tt = Conversions.any_to_tt(time)
        time_ut1 = Conversions.tt_to_ut1(time_tt)

        w_ei = TransformationMatrices.earth_rotation_matrix_derivative(time_ut1)

        w_ei = State.AngularVelocity(w_ei, RefFra.EarthCenteredInertial,
                                     RefFra.EarthCenteredEarthFixed)

        return w_ei


class EarthStateGeoid(EarthStateBase):
    def __init__(self):
        super().__init__()

        self._ct = CelTel(use_polar_motion=True, use_nutation_corrections=True)

        self._cache = Cache.Cache(max_size=5)

    def _get_transformation_and_angular_velocity(self, time):
        """
        :type time: Time.JulianDate.JulianDate
        """

        time_tt = Conversions.any_to_tt(time)

        if time_tt in self._cache:
            return self._cache[time_tt]
        else:
            t_ei, w_ei = self._ct.gcrs_to_itrs_angular_vel(time_tt)

            t_ei = State.Transformation(t_ei, RefFra.EarthCenteredInertial,
                                        RefFra.EarthCenteredEarthFixed)

            w_ei = State.AngularVelocity(w_ei, RefFra.EarthCenteredInertial,
                                         RefFra.EarthCenteredEarthFixed)

            self._cache.add(time_tt, (t_ei, w_ei))

            return t_ei, w_ei

    def transformation_matrix(self, time):
        """
        :type time: Time.JulianDate.JulianDate
        """

        answer = self._get_transformation_and_angular_velocity(time)

        return answer[0]

    def angular_velocity(self, time):
        """
        :type time: Time.JulianDate.JulianDate
        """

        answer = self._get_transformation_and_angular_velocity(time)

        return answer[1]
