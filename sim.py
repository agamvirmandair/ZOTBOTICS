import pybullet as p
import pybullet_data
import time
import random
# 1. Connect to physics server with GUI
physicsClient = p.connect(p.GUI)

# 2. Add default data path (for plane.urdf)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Set gravity
p.setGravity(0, 0, -9.8)

# 4. Load ground plane
plane_id = p.loadURDF("plane.urdf")


# Should be robot urdf, but for now we use a box
box_size = [0.2, 0.2, 0.2]

collision_shape = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=box_size
)

visual_shape = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=box_size,
    rgbaColor=[0, 0, 1, 1]  # Blue
)

robot_id = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=collision_shape,
    baseVisualShapeIndex=visual_shape,
    basePosition=[0, 0, 0.2]
)

def create_obstacle(position):
    half_extents = [0.2, 0.2, 0.5]

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[1, 0, 0, 1]
    )

    obstacle_id = p.createMultiBody(
        baseMass=0,  # 0 = static object
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )

    return obstacle_id


for _ in range(8):
    x = random.uniform(1, 4)
    y = random.uniform(-2, 2)
    create_obstacle([x, y, 0.5])

# 6. Simulation loop
while True:
    p.applyExternalForce(
        robot_id,
        linkIndex=-1,
        forceObj=[5, 0, 0],
        posObj=[0, 0, 0],
        flags=p.LINK_FRAME
    )
    p.stepSimulation()
    time.sleep(1./240.)