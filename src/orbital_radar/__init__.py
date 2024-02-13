import os

from dotenv import load_dotenv

load_dotenv()

from .suborbital import Suborbital
from .helpers import db2li, li2db
from .radarspec import RadarBeam
from .readers.radar import Radar
from .simulator import Simulator
from .version import __version__

# access environment variable
CONFIG_PATH = os.getenv("ORBITAL_RADAR_CONFIG_PATH", None)

# check environment variable
if CONFIG_PATH is None:
    print(
        "Warning: Environment variable ORBITAL_RADAR_CONFIG_PATH is not set. "
        "Set this variable to the path of the config file to avoid this "
        "warning"
    )

elif not os.path.exists(CONFIG_PATH):
    print(
        "Warning: Environment variable ORBITAL_RADAR_CONFIG_PATH is set, but "
        "the path does not exist. Set this variable to the path of the config "
        "file to avoid this warning. The current path is "
        f"{CONFIG_PATH}"
    )
