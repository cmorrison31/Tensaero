# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import heapq


class Cache:
    def __init__(self, max_size=None):
        self.max_size = max_size

        self._queue = dict()
        self._ordering = []

    def add(self, key, value):
        if key not in self._queue:
            self._queue[key] = value
            heapq.heappush(self._ordering, key)

            self._reduce_to_size()

    def remove_oldest(self):
        key = heapq.heappop(self._ordering)

        del self._queue[key]

    def _reduce_to_size(self):
        if self.max_size is None:
            return

        while len(self._queue) > self.max_size:
            self.remove_oldest()

    def __contains__(self, key):
        return key in self._queue

    def __getitem__(self, key):
        return self._queue[key]
