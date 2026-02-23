# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any, List

import h5py
import numpy as np


@dataclass(frozen=True)
class LogSignalSpecification:
    name: str
    getter: Callable[[], Any]
    group: str | None = None


class DataLogger:
    _signals: dict[str, LogSignalSpecification] | None
    _log_file: None | h5py.File
    _buffer: dict[str, List[Any]] | None

    def __init__(self, log_file_path: Path):
        self._signals = {}
        self._log_file = None
        self._buffer = {}
        self.chunk_size = 4

        self._log_file_path = log_file_path
        self._setup_log_file()

    def log_all(self):
        need_flush = False

        for key, val in self._signals.items():
            if key in self._buffer:
                self._buffer[key].append(val.getter())

                need_flush = len(self._buffer[key]) >= self.chunk_size

            else:
                self._buffer[key] = [val.getter()]

        if need_flush:
            self.flush_buffer()

    def _create_dataset(self, key, val):
        val = np.asarray(val[0])
        val_shape = val.shape
        val_dtype = val.dtype

        self._log_file.create_dataset(key, shape=(0,) + val_shape,
                                      maxshape=(None,) + val_shape,
                                      dtype=val_dtype,
                                      chunks=(self.chunk_size,) + val_shape)

    def flush_buffer(self):
        for key, val in self._buffer.items():
            if len(val) == 0:
                continue

            if key not in self._log_file.keys():
                self._create_dataset(key, val)

            dset = self._log_file[key]

            current_shape = dset.shape[0]

            # 2. Resize the dataset to accommodate the new data
            dset.resize((current_shape + len(val)), axis=0)

            # 3. Assign the new data to the newly allocated space
            dset[-len(val):] = val

            self._buffer[key].clear()

        self._log_file.flush()

    def register_sim_object_signal(self, obj_name: str,
                                   signal: LogSignalSpecification):
        namespace = 'sim objects'

        if signal.group is not None:
            full_path = f'{namespace}/{obj_name}/{signal.group}/{signal.name}'
        else:
            full_path = f'{namespace}/{obj_name}/{signal.name}'

        if full_path not in self._signals:
            self._signals[full_path] = signal

    def _setup_log_file(self):
        log_file_path: Path = self._log_file_path

        if not log_file_path.suffix == '.hdf5':
            log_file_path = log_file_path.with_suffix('.hdf5')

        log_file_path = log_file_path.absolute()

        self._log_file = h5py.File(log_file_path, "w")
