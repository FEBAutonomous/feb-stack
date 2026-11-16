"""Internal typed contract. Public simulator topics remain configurable."""
from .qos import SENSOR_QOS, QOS, LATCHED

CONTEXT = '/feb/run/context'
WALLS = '/feb/perception/walls'
MOTION = '/feb/estimation/motion'
STATE = '/feb/slam/state'
MAP = '/feb/slam/track_map'
WALL_CONTEXT = '/feb/slam/wall_context'
PLAN = '/feb/planning/race_plan'
REFERENCE = '/feb/planning/local_reference'
PREDICTION = '/feb/control/prediction'


def health(node):
    return '/feb/health/'+node
