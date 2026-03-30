"""Example where simulator receives proprioceptive data in world coordinates.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from pyspace import FrameGraph, FrameTransform

graph = FrameGraph()

# Define the reference frames.
world = graph.add_frame("world")
agent = graph.add_frame("agent")
sensor = graph.add_frame("sensor")

# Let's the proprioceptive data we get is always in world coordinates.
# Then we might receive agent and
# - agent pose, relative to the world
agent_location = np.array([0.0, 1.5, 0.0])
agent_orientation = Rotation.from_euler("xyz", [0, 30, 0], degrees=True)
# - sensor pose, also relative to the world
sensor_location = np.array([0.0, 1.5, 0.0])
sensor_orientation = Rotation.from_euler("xyz", [20, 30, 0], degrees=True)

# We'd then add the transforms to the graph like so:
graph.add_transform(FrameTransform.from_translation_and_rotation(
    translation=agent_location,
    rotation=agent_orientation,
    from_frame=agent,
    to_frame=world,
))
graph.add_transform(FrameTransform.from_translation_and_rotation(
    translation=sensor_location,
    rotation=sensor_orientation,
    from_frame=sensor,
    to_frame=world,
))

# We can now move locations, displacements, orientations, and poses between
# frames without considering how frames are related to each other.

# location example
loc_rel_sensor = sensor.location([12.0, 1.0, -0.5])
loc_rel_world = loc_rel_sensor.to(world)

# displacement example
disp_rel_world = world.displacement([12.0, 1.0, -0.5])
disp_rel_sensor = disp_rel_world.to(sensor)

# orientation example
ori_rel_world = world.orientation(Rotation.from_euler("xyz", [20, 0, 0], degrees=True))
ori_rel_agent = ori_rel_world.to(agent)

# pose example
pose_rel_world = world.pose(
    translation=[12.0, 1.0, -0.5],
    rotation=Rotation.from_euler("xyz", [20, 0, 0], degrees=True),
)
pose_rel_sensor = pose_rel_world.to(agent)

# When new data comes in, clear the transforms and add new ones.
graph.clear_transforms()
graph.add_transform(FrameTransform.from_translation_and_rotation(
    translation=agent_location,
    rotation=agent_orientation,
    from_frame=agent,
    to_frame=world,
))
graph.add_transform(FrameTransform.from_translation_and_rotation(
    translation=sensor_location,
    rotation=sensor_orientation,
    from_frame=sensor,
    to_frame=world,
))