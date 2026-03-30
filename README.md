# pyspace

A library for managing rigid reference frames as a graph and transforming
geometric objects between them.

You define **frames** (nodes) and **frame transforms** (edges), then ask for any
object to be expressed in any other frame — pyspace finds the shortest path
through the graph and composes transforms automatically.

One of the main ideas here is that we don't want to care about what the
kinematic chain is, or whether we get location and orientation information
about our bodies in local or world coordinates. See Quick Start below for
an example.

## Install

```bash
uv add pyspace
```

## Quick start

```python
import numpy as np
from scipy.spatial.transform import Rotation

from pyspace import FrameGraph, FrameTransform

graph = FrameGraph()

# Define the reference frames.
world = graph.add_frame("world")
agent = graph.add_frame("agent")
sensor = graph.add_frame("sensor")

# Let's the proprioceptive data we get is in local coordinates
# i.e., defined relative to a parent frame. Then we might receive agent and
# sensor poses like this:
# - agent pose, relative to the world
agent_location = np.array([0.0, 1.5, 0.0])
agent_orientation = Rotation.from_euler("xyz", [0, 30, 0], degrees=True)
# - sensor pose, relative to the agent
sensor_location = np.array([0.0, 0.0, 0.0])
sensor_orientation = Rotation.from_euler("xyz", [20, 0, 0], degrees=True)

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
    to_frame=agent,
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

```
Instead of receiving data in local coordinates, let's say we receive it in
world coordinates. The only difference on our end is how we add the transforms.
We'll reuse the graph/frames, clear the existing transforms, and add new
transforms.

```python
graph.clear_transforms()

# - agent pose, relative to the world
agent_location = np.array([0.0, 1.5, 0.0])
agent_orientation = Rotation.from_euler("xyz", [0, 30, 0], degrees=True)
# - sensor pose, also relative to the world
sensor_location = np.array([0.0, 1.5, 0.0])
sensor_orientation = Rotation.from_euler("xyz", [20, 30, 0], degrees=True)

# Add the new transforms.
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
```
Note that we're just supplying diffreent `from_frame` and `to_frame` arguments
when we add the transform.


## Main Componenents and Usage

### Frames and FrameGraph

A `FrameGraph` holds `Frame` objects (nodes) and `FrameTransform` objects
(directed edges). Frames MUST be created by the graph like so:

```python
graph = FrameGraph()
a = graph.add_frame("a")
```

You can look up frames by id and inspect the graph:

```python
graph["a"]              # Frame lookup by id
graph.frames            # bidict[FrameID, Frame]
graph.transforms        # bidict[(Frame, Frame), FrameTransform]
```

### Transforms
`FrameTransform` objects are the edges in the graph. When we create them, we
specify which frames they operate between.

```python
t = FrameTransform.from_translation_and_rotation(
    translation=np.array([1.0, 0.0, 0.0]),
    rotation=Rotation.from_euler("z", 90, degrees=True),
    from_frame=a,
    to_frame=b,
)
graph.add_transform(t)
```

### Geometric types

All geometric types carry a reference to their frame and implement `.to(frame)`:

| Type | Meaning | Transform behaviour |
|---|---|---|
| `Location` | A point in 3-D space | Full rigid transform (rotation + translation) |
| `Displacement` | A free vector (e.g. velocity) | Rotation only — translation is ignored |
| `Orientation` | A rotation in a frame | Rotation composition |
| `Pose` | Location + Orientation | Both components transformed independently |

They can be instantiated directly if given a `Frame`, but it's easiest to let
the `Frame` create them with its methods like so:

```python
loc  = frame.location([1, 2, 3])
disp = frame.displacement([0, 0, 1])
ori  = frame.orientation(Rotation.from_euler("z", 90, degrees=True))
pose = frame.pose([1, 2, 3], Rotation.identity())
```

Transform to another frame:

```python
loc_world = loc.to(world)
```

## AI-generated API summary (looks right to me)

### `FrameGraph`

| Method | Description |
|---|---|
| `add_frame(id=None)` | Add a frame; returns `Frame` |
| `remove_frame(frame)` | Remove a frame and its transforms |
| `add_transform(t)` | Add a `FrameTransform` edge |
| `remove_transform(t)` | Remove a transform edge |
| `clear_transforms()` | Remove all transforms |
| `path(from, to)` | Shortest path as `list[PathStep]` |
| `transform(obj, to)` | Transform a geometric object to a target frame |
| `show(...)` | Render the graph with Graphviz |
| `graph[id]` | Look up a frame by id |

### `FrameTransform`

| Method | Description |
|---|---|
| `from_translation_and_rotation(t, r, from, to)` | Create from components |
| `from_rigid_transform(rt, from, to)` | Create from a SciPy `RigidTransform` |
| `inv()` | Return the inverse transform |
| `apply(obj)` | Apply to a geometric object (must be in `from_frame`) |
| `translation` / `rotation` | Access components |
| `as_translation_and_rotation()` | Return `(ndarray, Rotation)` tuple |
| `as_rigid_transform()` | Return the underlying `RigidTransform` |

### Geometric types

All implement `.to(frame)` and `.apply_frame_transform(t)`:

| Type | Constructor helpers |
|---|---|
| `Location(array, frame)` | `frame.location(array)`, `Location.from_array(...)` |
| `Displacement(array, frame)` | `frame.displacement(array)`, `Displacement.from_array(...)` |
| `Orientation(rotation, frame)` | `frame.orientation(rot)`, `Orientation.from_euler(...)`, `.from_matrix(...)`, `.from_quat(...)`, `.from_rotation(...)` |
| `Pose(location, orientation)` | `frame.pose(translation, rotation)` |
