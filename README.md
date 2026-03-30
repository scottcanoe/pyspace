## pyspace

`pyspace` includes a reference-frame graph for on-demand coordinate transforms.

### Quick start

```python
import numpy as np
from scipy.spatial.transform import Rotation

from pyspace import FrameGraph, FrameTransform

graph = FrameGraph()
world, robot, camera = graph.add_frames(["world", "robot", "camera"])

# camera -> robot
graph.add_transform(
    FrameTransform.from_components(
        translation=np.array([0.2, 0.0, 0.1]),
        rotation=Rotation.identity(),
        from_frame=camera,
        to_frame=robot,
    )
)

# robot -> world
graph.add_transform(
    FrameTransform.from_components(
        translation=np.array([1.0, 2.0, 0.0]),
        rotation=Rotation.identity(),
        from_frame=robot,
        to_frame=world,
    )
)

point_camera = camera.location(np.array([0.5, 0.0, 0.0]))
point_world = point_camera.to(world)
```

The transform is resolved on demand by searching the frame graph and composing
the transforms along the shortest path.

### Animate a moving object pose

```python
import numpy as np
from scipy.spatial.transform import Rotation

from pyspace import FrameGraph, animate_object_poses, interpolate_poses

graph = FrameGraph()
world = graph.add_frame("world")
keyframes = [
    world.pose(
        translation=np.array([0.0, 0.0, 0.0]),
        rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
    ),
    world.pose(
        translation=np.array([1.0, 0.5, 0.0]),
        rotation=Rotation.from_euler("xyz", [0, 45, 45], degrees=True),
    ),
    world.pose(
        translation=np.array([2.0, 1.0, 0.5]),
        rotation=Rotation.from_euler("xyz", [45, 45, 90], degrees=True),
    ),
]

poses = interpolate_poses(keyframes, frames_per_segment=25)
animate_object_poses(poses)
```

This shows a wireframe object moving through 3D space while updating its
orientation axes each frame.
