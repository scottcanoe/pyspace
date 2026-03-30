## pyspace

`pyspace` includes a reference-frame graph for on-demand coordinate transforms.

### Quick start

```python
import numpy as np

from pyspace import FrameGraph, Transform

graph = FrameGraph()
graph.add_frames(["world", "robot", "camera"])

# camera -> robot
graph.add_transform(
    from_frame="camera",
    to_frame="robot",
    transform=Transform(
        rotation=np.eye(3),
        translation=np.array([0.2, 0.0, 0.1]),
    ),
)

# robot -> world
graph.add_transform(
    from_frame="robot",
    to_frame="world",
    transform=Transform(
        rotation=np.eye(3),
        translation=np.array([1.0, 2.0, 0.0]),
    ),
)

point_camera = np.array([0.5, 0.0, 0.0])
point_world = graph.transform_coordinates(
    point_camera,
    from_frame="camera",
    to_frame="world",
)
```

The transform is resolved on demand by searching the frame graph and composing
the transforms along the shortest path.

### Animate a moving object pose

```python
import numpy as np
from scipy.spatial.transform import Rotation

from pyspace import Transform, animate_object_poses, interpolate_transforms

keyframes = [
    Transform(
        rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix(),
        translation=np.array([0.0, 0.0, 0.0]),
    ),
    Transform(
        rotation=Rotation.from_euler("xyz", [0, 45, 45], degrees=True).as_matrix(),
        translation=np.array([1.0, 0.5, 0.0]),
    ),
    Transform(
        rotation=Rotation.from_euler("xyz", [45, 45, 90], degrees=True).as_matrix(),
        translation=np.array([2.0, 1.0, 0.5]),
    ),
]

poses = interpolate_transforms(keyframes, frames_per_segment=25)
animate_object_poses(poses)
```

This shows a wireframe object moving through 3D space while updating its
orientation axes each frame.
