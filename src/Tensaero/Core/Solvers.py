# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from abc import ABC, abstractmethod
from collections.abc import Callable
from Tensaero.Core import State


class SolverBase(ABC):
    function_accelerations: Callable
    functions_new_state: Callable

    def __init__(self, function_accelerations, functions_new_state):
        self.function_accelerations = function_accelerations
        self.functions_new_state = functions_new_state

    @abstractmethod
    def next_state(self, current_state, dt):
        pass


class SolverFixed(SolverBase):
    """
    This solver class is for objects that are fixed in place in some
    reference frame. No actual physics update is performed.
    """

    def __init__(self, function_accelerations, functions_new_state):
        super().__init__(function_accelerations, functions_new_state)

    def next_state(self, current_state, dt):
        v_bi_i = current_state.v_bi_i
        p_bi_i = current_state.s_bi_i

        state_frame = self.functions_new_state(dt + current_state.time, p_bi_i,
                                               v_bi_i)

        return state_frame


class SolverEuler(SolverBase):
    """
    This solver class is for objects whose physics state is integrated using
    Euler integration. Euler integration is first-order accurate and not
    recommended.
    """

    def __init__(self, function_accelerations, functions_new_state):
        super().__init__(function_accelerations, functions_new_state)


    def next_state(self, current_state, dt):
        accel_i = self.function_accelerations(current_state)
        accel_i = State.Acceleration(accel_i.reshape(3, ),
                                     State.ReferenceFrames.EarthCenteredInertial,
                                     State.CoordinateSystems.Cartesian)

        v_bi_i = (current_state.v_bi_i +
                  State.Velocity.from_vector_data(accel_i * dt))
        p_bi_i = (current_state.s_bi_i +
                  State.Position.from_vector_data(v_bi_i * dt))

        state_frame = self.functions_new_state(dt + current_state.time, p_bi_i,
                                               v_bi_i)

        return state_frame


class SolverVelocityVerlet(SolverBase):
    """
    This solver class is for objects whose physics state is integrated using
    Velocity Verlet integration. Velocity Verlet is second-order accurate and
    is the recommended solver scheme.
    """

    def __init__(self, function_accelerations, functions_new_state):
        super().__init__(function_accelerations, functions_new_state)


    def next_state(self, current_state, dt):
        accel_bi_i = self.function_accelerations(current_state)

        # Calculate velocity half-step
        v_bi_i_half = current_state.v_bi_i + 0.5 * accel_bi_i * dt

        # Calculate position update
        p_bi_i = current_state.s_bi_i + v_bi_i_half * dt

        # Calculate the new acceleration by calculating the new state
        state_frame = self.functions_new_state(dt + current_state.time, p_bi_i,
                                               v_bi_i_half)

        accel_bi_i = self.function_accelerations(state_frame)

        v_bi_i = v_bi_i_half + 0.5 * accel_bi_i * dt

        # Calculate the final state
        state_frame = self.functions_new_state(dt + current_state.time, p_bi_i,
                                               v_bi_i)

        return state_frame
