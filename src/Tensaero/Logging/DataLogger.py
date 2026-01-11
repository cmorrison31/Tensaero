

class LogSignal:
    def __init__(self, name, period=None):
        self.name: str = name
        self.period: float | None = period

        self._data: list[tuple] = []

    def add_data(self, time, data):
        if len(self._data) == 0:
            t0 = time
        else:
            t0 = self._data[-1][0]

        if (len(self._data) == 0 or self.period is None or
                time - t0 >= self.period):
            self._data.append((time, data))


class DataLogger:
    _signals: dict[str, LogSignal] | None = None

    def __init__(self):
        if DataLogger._signals is None:
            DataLogger._signals = {}

    def register_signal(self, path, signal: LogSignal):
        full_path = path + '/' + signal.name

        if full_path not in self._signals:
            DataLogger._signals[full_path] = signal


def get_logger():
    return DataLogger()
