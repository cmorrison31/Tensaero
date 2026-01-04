# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from Tensaero import Simulator


def main():
    config_file = "config.yml"
    sim = Simulator.Simulator(config_file)

    sim.run(time_max=1.0)


if __name__ == "__main__":
    main()