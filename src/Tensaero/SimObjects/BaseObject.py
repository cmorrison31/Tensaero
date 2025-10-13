# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod
from Tensaero.Core import State


class BaseObject(ABC):
    def __init__(self):
        self.state = State.StateFrame


    @abstractmethod
    def init_state(self, time, position, velocity):
        """
        :type time: float
        :type velocity: State.Velocity
        :type position: State.Position
        """
        pass


class FixedGroundPoint(BaseObject):
    def __init__(self):
        super().__init__()

    def init_state(self, time, position, velocity=None):
        self.state.time = time

