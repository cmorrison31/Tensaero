# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


class SimEngine:
    def __init__(self, environment, solver, logging):
        self.environment = environment
        self.solver = solver
        self.logging = logging

