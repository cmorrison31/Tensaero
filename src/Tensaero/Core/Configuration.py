# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import importlib
import inspect
import re
import string
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Annotated, Callable
from zoneinfo import ZoneInfo

from pydantic import BaseModel, field_validator, Field, BeforeValidator

from Tensaero.Core.State import ReferenceFrames, CoordinateSystems


class SimObjectTypes(Enum):
    general = 'general'
    ground = 'ground'
    fixed_point = 'fixed point'


class SolverType(Enum):
    default = 'default'
    velocity_verlet = 'velocity verlet'
    euler = 'euler'
    fixed = 'fixed'


class EarthType(Enum):
    default = 'default'
    geoid = 'geoid'
    spherical = 'spherical'


def reference_frames_validator(value: str):
    value = value.strip().replace(",", "")
    value = value.translate(str.maketrans('', '', string.whitespace))
    value = value.title()

    try:
        val = ReferenceFrames(value)
    except ValueError:
        raise ValueError(f"Invalid value '{value}'. Must be one of: "
                         f"{[x.name for x in ReferenceFrames]}")

    if (val != ReferenceFrames.EarthCenteredInertial and
            val != ReferenceFrames.EarthCenteredEarthFixed):
        raise ValueError('Initial conditions can only be provided in '
                         'Earth Centered Inertial or Earth Centered, '
                         'Earth Fixed reference frames.')

    return val


def coordinate_systems_validator(value: str):
    try:
        return CoordinateSystems(value)
    except ValueError:
        raise ValueError(f"Invalid value '{value}'. Must be one of: "
                         f"{[x.name for x in CoordinateSystems]}")


class VectorData(BaseModel):
    data: List[float] = Field(..., min_length=3, max_length=3)
    reference_frame: Annotated[
        ReferenceFrames, BeforeValidator(reference_frames_validator)] = (
        Field(..., alias="reference frame"))
    coordinate_system: Annotated[
        CoordinateSystems, BeforeValidator(coordinate_systems_validator)] = (
        Field(CoordinateSystems.Cartesian, alias="coordinate system"))

    @field_validator("data")
    @classmethod
    def validate_vector(cls, v):
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("All elements in 'data' must be numeric.")
        return v


class OrientationData(BaseModel):
    heading_angle: float = Field(default=0.0, alias="heading angle")
    flight_path_angle: float = Field(default=0.0, alias="flight path angle")


class InitialConditions(BaseModel):
    position: VectorData
    velocity: VectorData
    orientation: OrientationData


def start_time_validator(value: str | datetime):
    if not isinstance(value, datetime) and value.lower().strip() == "now":
        value = datetime.now(ZoneInfo("localtime"))

    return value


def user_function_validator(v: str):
    if not re.match(r"^[\w.]+:[A-Za-z_][\w_]*$", v):
        raise ValueError("Must be of form '<module.path>:<function_name>'")

    module_name, func_name = v.split(":", 1)

    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        raise ValueError(f"Module '{module_name}' cannot be imported")

    if not hasattr(mod, func_name):
        raise ValueError(
            f"Function '{func_name}' not found in module '{module_name}'")

    func = getattr(mod, func_name)
    if not callable(func):
        raise ValueError(
            f"'{func_name}' in module '{module_name}' is not callable")

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if len(params) != 1:
        raise ValueError(
            f"Function '{func_name}' must take exactly one argument, found {len(params)}")

    return func


class SimObjects(BaseModel):
    name: str
    object_type: SimObjectTypes = Field(default='general', alias="object type")

    acceleration_function: Annotated[
        Callable, BeforeValidator(user_function_validator)] = (
        Field(..., alias="acceleration function"))

    logging_function: Annotated[
        Callable, BeforeValidator(user_function_validator)] = (
        Field(..., alias="logging function"))

    initial_conditions: InitialConditions = (
        Field(..., alias="initial conditions"))

    solver: SolverType = Field(SolverType.default)


class ConfigSchema(BaseModel):
    sim_objects: List[SimObjects] = Field(..., alias="sim objects",
                                          min_length=1)
    start_time: Annotated[datetime, BeforeValidator(start_time_validator)] = (
        Field(default=datetime.now(ZoneInfo("localtime")), alias="start time"))
    time_step: float = Field(default=1e-3, alias="time step", gt=0)
    earth_type: EarthType = Field(EarthType.default, alias="earth")
    log_file_path: Path = Field(default=Path.cwd(), alias="log file path")
