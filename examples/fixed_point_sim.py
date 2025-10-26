import yaml
from Tensaero.Core import Configuration


def main():
    conf = yaml.safe_load(open("config.yml"))

    conf = Configuration.ConfigScheme(**conf)

    print(conf)


if __name__ == "__main__":
    main()