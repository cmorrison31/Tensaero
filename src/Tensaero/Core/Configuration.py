import importlib
import inspect
import re
import string
from enum import Enum
from pathlib import Path
from typing import List, Annotated, Callable

from pydantic import BaseModel, field_validator, Field
from pydantic import DirectoryPath, BeforeValidator

from Tensaero.Core.State import ReferenceFrames


class SimObjectTypes(Enum):
    general = 'general'
    ground = 'ground'
    fixed_point = 'fixed point'


def reference_frames_validator(value: str):
    value = value.strip().replace(",", "")
    value = value.translate(str.maketrans('', '', string.whitespace))
    value = value.title()

    try:
        return ReferenceFrames(value)
    except ValueError:
        raise ValueError(f"Invalid value '{value}'. Must be one of: "
                         f"{[x.name for x in ReferenceFrames]}")


class VectorData(BaseModel):
    data: List[float] = Field(..., min_length=3, max_length=3)
    reference_frame: Annotated[
        ReferenceFrames, BeforeValidator(reference_frames_validator)] = (
        Field(..., alias="reference frame"))

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
    functions_path: DirectoryPath = Field(default=Path.cwd(),
                                          alias="functions path")

    accelerations_function: Annotated[
        Callable, BeforeValidator(user_function_validator)] = (
        Field(..., alias="accelerations function"))

    logging_function: Annotated[
        Callable, BeforeValidator(user_function_validator)] = (
        Field(..., alias="logging function"))

    initial_conditions: InitialConditions = (
        Field(..., alias="initial conditions"))


class ConfigScheme(BaseModel):
    sim_objects: SimObjects = Field(..., alias="sim objects")