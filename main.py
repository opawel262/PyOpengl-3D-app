import pygame as pg
from OpenGL.GL import *
import numpy as np
import pyrr
from helper_functions import create_shader,  load_model_from_file
from constants import *
#  CONSTANTS

positions_of_lights = [[9, 5, 0], [11.9, 6.7, 0], [11.9, 4, 0], [9, 4, 0], [1, 2, 2], [15.1, 15.2, 1.05],
                       [15.1, 15.1, 1.05], [15.1, 15.1, 1.05]]

################### Model #####################################################


class Entity:
    """ Represents a general object with a position and rotation applied"""

    def __init__(
        self, position: list[float], 
        eulers: list[float],
        scale: list[float],  # Added scale parameter
        objectType: int
    ):
        """
        Initialize the entity, store its state and update its transform.

        Parameters:
            position: The position of the entity in the world (x,y,z)
            eulers: Angles (in degrees) representing rotations around the x,y,z axes.
            scale: Scaling factors along the x, y, and z axes.
            objectType: The type of object which the entity represents,
                        this should match a named constant.
        """

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)
        self.objectType = objectType

    def get_model_transform(self) -> np.ndarray:
        """
        Calculates and returns the entity's transform matrix,
        based on its position, rotation, and scaling.
        """

        # Inicjalizacja macierzy transformacji jako macierzy jednostkowej
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)

        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        scale_matrix = pyrr.matrix44.create_from_scale(self.scale)

        # Rotation around Z-axis
        if self.eulers[2] != 0:
            rotation_matrix_z = pyrr.matrix44.create_from_z_rotation(
                theta=np.radians(self.eulers[2]),
                dtype=np.float32
            )
            model_transform = np.dot(model_transform, rotation_matrix_z)

        # Rotation around Y-axis
        if self.eulers[1] != 0:
            rotation_matrix_y = pyrr.matrix44.create_from_y_rotation(
                theta=np.radians(self.eulers[1]),
                dtype=np.float32,
            )
            model_transform = np.dot(model_transform, rotation_matrix_y)

        # Rotation around X-axis
        if self.eulers[0] != 0:
            rotation_matrix_x = pyrr.matrix44.create_from_x_rotation(
                theta=np.radians(self.eulers[0]),
                dtype=np.float32
            )
            model_transform = np.dot(model_transform, rotation_matrix_x)
        # Skalowanie

        # Przesunięcie na podstawie pozycji jednostki
        translation_matrix = pyrr.matrix44.create_from_translation(
            vec=self.position,
            dtype=np.float32
        )

        # Kolejność operacji: skalowanie -> obrót -> translacja
        model_transform = pyrr.matrix44.multiply(model_transform, scale_matrix)

        model_transform = pyrr.matrix44.multiply(model_transform, translation_matrix)

        return model_transform

    def update(self) -> None:

        raise NotImplementedError


class FloorTile(Entity):

    def __init__(
        self, position: list[float],
        eulers: list[float],
        scale: list[float]):

        super().__init__(position, eulers, scale, OBJECT_FLOOR_TILE)
        self.rotation_done = False  # Flag to track whether rotation has been done

    def update(self):
        pass


class Rack(Entity):

    def __init__( self, position: list[float],eulers: list[float],  scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_RACK)

    def update(self):
        pass


class Arm(Entity):
    def __init__(self, position: list[float], eulers: list[float], scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_ARM)
        self.attached_arm: Arm = None
        self.rotation_speed = 0.05  # Adjust the rotation speed as needed
        self.rotate = np.radians(0)
        self.rotate_child = np.radians(0)
        self.rotate_grand_child = np.radians(0)
        self.rotated = False
        self.add = 0.0
        self.added = False
        self.offset = [0.0, 0.0, 0.4]
        self.rotate_by_player = np.radians(0)
        self.end_of_arm = False

    def update(self):

        if self.attached_arm:
            self.attached_arm.rotate = self.rotate
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(-np.radians(self.rotate_by_player + self.rotate)), -np.sin(-np.radians(self.rotate_by_player + self.rotate ))],
                [0, np.sin(-np.radians(self.rotate_by_player + self.rotate )), np.cos(-np.radians(self.rotate_by_player + self.rotate ))]
            ])
            self.eulers[0] += self.rotate_by_player/2

            new_position = np.dot(rotation_matrix, self.offset) + self.position
            self.attached_arm.position = new_position.tolist()
            self.attached_arm.eulers = [
                self.eulers[0] + self.rotate + self.rotate_child,
                self.eulers[1] + self.rotate * 5,
                self.eulers[2] + self.rotate / 2 + self.rotate_child

            ]

            self.attached_arm.update()

    def attach_arm(self, other_arm):
        self.attached_arm: Arm = other_arm
        self.update()


class EndingOfArm(Entity):
    def __init__(self, position: list[float], eulers: list[float], scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_ENDING_OF_ARM)
        self.attached_arm: Arm = None
        self.offset = [0, 0.0, -0.45]


    def update(self):
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(-np.radians(self.attached_arm.eulers[0] * 2)),
             -np.sin(-np.radians(self.attached_arm.eulers[0]* 2))],
            [0, np.sin(-np.radians(self.attached_arm.eulers[0]* 2)),
             np.cos(-np.radians(self.attached_arm.eulers[0]* 2))]
        ])


        new_position = np.dot(rotation_matrix, self.offset) + self.attached_arm.position
        self.position = new_position.tolist()
        self.eulers = [
            self.attached_arm.eulers[0] * 2,
            0,
            0,
        ]

    def attach_to(self, arm: Arm):
        self.attached_arm = arm
        self.update()


class Wall(Entity):

    def __init__(
            self, position: list[float],
            eulers: list[float],
            scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_WALL)
        self.rotation_done = False  # Flag to track whether rotation has been done

    def update(self):
        pass


class TextEntity(Entity):

    def __init__(
            self, position: list[float],
            eulers: list[float],
            scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_TEXT)
        self.rotation_done = False  # Flag to track whether rotation has been done

    def update(self):
        pass


class LightEntity(Entity):
    def __init__(self, position: list[float], eulers: list[float], scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_SPHERE)
        self.rotation_done = False  # Flag to track whether rotation has been done

    def update(self):
        pass


class Table(Entity):
    def __init__(
            self, position: list[float],
            eulers: list[float],
            scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_TABLE)
        self.rotation_done = False  # Flag to track whether rotation has been done

    def update(self):
        pass


class StandingForRobot(Entity):
    def __init__(self, position: list[float], eulers: list[float], scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_STANDING_FOR_ROBOT)
        self.attached_arm: Arm = None
        self.rotated = False
        self.rotate = 0.0
        self.rotate_by_player = [0, 0, 0]
        self.destinated_to = False
        self.enable_auto_rotation = True

    def attach_arm(self, arm: Arm):
        """ Attach an arm entity to this StandingForRobot """
        self.attached_arm = arm

    def update(self):

        if self.attached_arm:
            self.attached_arm.rotate = self.rotate
            # Update the attached arm's position and orientation based on the standing robot
            if self.destinated_to and self.enable_auto_rotation:
                self.position[0] += 0.01
                if self.position[0] >= 12:
                    self.destinated_to = False
            else:
                if self.enable_auto_rotation:
                    self.position[0] -= 0.01
                    if self.position[0] <= 7.6:
                        self.destinated_to = True

            self.attached_arm.position = [
                self.position[0] ,
                self.position[1] + -0.08,
                self.position[2] + 0.48
            ]
            if self.enable_auto_rotation:
                if not self.rotated:
                    self.rotate += 1
                    if self.rotate >= 0:
                        self.rotated = True
                else:
                    self.rotate -= 1
                    if self.rotate <= -90:
                        self.rotated = False


            self.attached_arm.eulers = [
                self.eulers[0] + self.rotate,
                self.eulers[1] + self.rotate,
                self.eulers[2]
                ]
            # Now, update the attached arm
            self.attached_arm.update()


class Chair(Entity):
    def __init__(
            self, position: list[float],
            eulers: list[float],
            scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_CHAIR)
        self.rotation_done = False  # Flag to track whether rotation has been done

    def update(self):
        pass


class Cat(Entity):
    def __init__(
            self, position: list[float],
            eulers: list[float],
            scale: list[float]):
        super().__init__(position, eulers, scale, OBJECT_CAT)
        self.rotation_done = False  # Flag to track whether rotation has been done
        self.is_falling_down = False

    def update(self):
        pass


class Cylinder(Entity):
    def __init__(
            self, position: list[float],
            eulers: list[float],
            scale: list[float], ):
        super().__init__(position, eulers, scale, OBJECT_SHAPE_O)
        self.attached_to_object: Entity = None
        self.is_falling_down = False

    def fall_down(self):
        self.position[2] -= 0.02

    def attach_to(self, object: Entity):
        self.attached_to_object = object
        self.update()

    def detach(self):
        self.attached_to_object = None

    def update(self):
        if self.attached_to_object:
            self.position = self.attached_to_object.position + [1, 1, 1]
            self.eulers = self.attached_to_object.eulers

        if self.is_falling_down:
            if 4.0 <= self.position[1] <= 5.7:
                if self.position[2] >= -0.152:
                    self.fall_down()
            elif 5.8 < self.position[1] <= 6.45:
                if self.position[2] >= -0.4:
                    self.fall_down()

            elif 6.45 < self.position[1] <= 7.9:
                if self.position[2] >= -1.1:
                    self.fall_down()
            else:
                self.is_falling_down = False


class ShapeO(Entity):
    def __init__(
            self, position: list[float],
            eulers: list[float],
            scale: list[float], ):
        super().__init__(position, eulers, scale, OBJECT_SHAPE_O)
        self.attached_to_object: Entity = None
        self.is_falling_down = False

    def fall_down(self):
        self.position[2] -= 0.02

    def attach_to(self, object: Entity):
        self.attached_to_object = object
        self.update()

    def detach(self):
        self.attached_to_object = None

    def update(self):
        if self.attached_to_object:
            self.position = self.attached_to_object.position + [1, 1, 1]
            self.eulers = self.attached_to_object.eulers

        if self.is_falling_down:
            if 4.0 <= self.position[1] <= 5.7:
                if self.position[2] >= -0.152:
                    self.fall_down()
            elif 5.8 < self.position[1] <= 6.45:
                if self.position[2] >= -0.4:
                    self.fall_down()

            elif 6.45 < self.position[1] <= 7.9:
                if self.position[2] >= -1.1:
                    self.fall_down()
            else:
                self.is_falling_down = False

class ArmOtherRobot(Entity):
    def __init__(self, position: list[float], eulers: list[float], scale: list[float], attached_arm: Arm = None):
        super().__init__(position, eulers, scale, ARM_ROBOT)
        self.rotation_done = False  # Flag to track whether rotation has been done
        self.attached_arm: Arm = attached_arm

    def update(self):
        if self.attached_arm:
            self.attached_arm.position = self.position
            self.attached_arm.eulers = self.eulers
            self.attached_arm.update()

    def attach_arm(self):
        pass


class Player(Entity):
    """ A first person camera controller. """

    def __init__(self, position: list[float], eulers: list[float], scale: list[float]):

        super().__init__(position, eulers, scale, OBJECT_CAMERA)

        self.localUp = np.array([0,0,1], dtype=np.float32)

        #directions after rotation
        self.up = np.array([0,0,1], dtype=np.float32)
        self.right = np.array([0,1,0], dtype=np.float32)
        self.forwards = np.array([1,0,0], dtype=np.float32)
    
    def calculate_vectors(self) -> None:
        """ 
            Calculate the camera's fundamental vectors.

            There are various ways to do this, this function
            achieves it by using cross products to produce
            an orthonormal basis.
        """

        #calculate the forwards vector directly using spherical coordinates
        self.forwards = np.array(
            [
                np.cos(np.radians(self.eulers[2])) * np.cos(np.radians(self.eulers[1])),
                np.sin(np.radians(self.eulers[2])) * np.cos(np.radians(self.eulers[1])),
                np.sin(np.radians(self.eulers[1]))
            ],
            dtype=np.float32
        )
        self.right = pyrr.vector.normalise(np.cross(self.forwards, self.localUp))
        self.up = pyrr.vector.normalise(np.cross(self.right, self.forwards))

    def update(self) -> None:
        """ Updates the camera """

        self.calculate_vectors()
    
    def get_view_transform(self) -> np.ndarray:
        """ Return's the camera's view transform. """

        return pyrr.matrix44.create_look_at(
            eye=self.position - [0,0,-0.2],
            target=self.position + self.forwards ,
            up = self.up,
            dtype = np.float32
        )


class Scene:
    """ 
        Manages all logical objects in the game,
        and their interactions.
    """

    def __init__(self):
        """ Create a scene """

        self.renderables: dict[int, list] = dict()
        self.renderables[OBJECT_PLAYER] = [
            Player(position=[2, 2, 0.5], eulers=[44, 1, 15], scale=[0.1, 0.1, 0.1]),
            ]
        self.renderables[OBJECT_FLOOR_TILE] = [
            FloorTile(position=[1 + i*0.6 * 2, 1 + j*0.6, -1.5],eulers=[90, 0, 0],scale=[2, 2, 2]) for i in range(12) for j in range(12)

        ]
        self.renderables[OBJECT_WALL] = [
            Wall(position=[7.6, 0.35, 0.3], eulers=[0, 0, 0], scale=[24, 5, 6.2]),
            Wall(position=[7.6, 8.2, 0.3], eulers=[0, 0, 0], scale=[24, 5, 6.2]),
            Wall(position=[0.4, 4.2, 0.2], eulers=[0, 0, 90], scale=[4, 13.4, 6.5]), # sciana krotsza
            Wall(position=[14.75, 4.2, 0.2], eulers=[0, 0, 90], scale=[4, 13.4, 6.5]), # sciana krotsza
            Wall(position=[7.5, 4.35, 2.05], eulers=[90, 0, 0], scale=[24.1, 13, 5]) # sufit
            ]
        self.renderables[OBJECT_RACK] = [
            Rack(position=[10.9, 6, -1], eulers=[0, 0, 0], scale=[3.5, 0.4, 0.5]),
            Rack(position=[10.9, 3, -1], eulers=[0, 0, 0], scale=[3.5, 0.4, 0.6]),
            ]

        self.renderables[OBJECT_TEXT] = [
            TextEntity(position=[14.68, 7, 0.5], eulers=[0, -180,90], scale=[1, 1, 1]),
            ]
        self.renderables[OBJECT_STANDING_FOR_ROBOT] = [
            StandingForRobot(position=[12, 6.1, -0.4], eulers=[-90, -90, 0], scale=[2, 2, 2]),
            StandingForRobot(position=[8, 3, -0.4], eulers=[-90, -90, 0], scale=[2, 2, 2]),
            ]
        self.renderables[OBJECT_ENDING_OF_ARM] = [
            EndingOfArm(position=[0, 0, 0.0], eulers=[0, 90, 0], scale=[0.2, 0.2, 0.3]),
            EndingOfArm(position=[0, 0, 0.0], eulers=[0, 90, 0], scale=[0.2, 0.2, 0.3]),
        ]
        self.renderables[OBJECT_ARM] = [
            Arm(position=[0, 0, 0], eulers=[0, 0, 0], scale=[0.6, 0.6, 0.6]),
            Arm(position=[0, 0, 0], eulers=[0, 0, 0], scale=[0.6, 0.6, 0.6]),
            Arm(position=[0, 0, 0], eulers=[0, 90, 0], scale=[0.6, 0.6, 0.6]),
            Arm(position=[0, 0, 0], eulers=[0, 0, 0], scale=[0.6, 0.6, 0.6]),
            Arm(position=[0, 0, 0], eulers=[0, 0, 0], scale=[0.6, 0.6, 0.6]),

            ]
        self.renderables[OBJECT_TABLE] = [
            Table(position=[10.2, 5.1, -1.5], eulers=[-90, 90, 0], scale=[1, 1, 1.2]),
            Table(position=[10.2, 2.1, -1.5], eulers=[-90, 90, 0], scale=[1, 1, 1.2]),

        ]
        self.renderables[OBJECT_CHAIR] = [
            Chair(position=[1, 1, -1.5], eulers=[-90, 210, 0], scale=[2, 2, 2]),
            Chair(position=[13.5, 7.2, -1.5], eulers=[-90, 90, 0], scale=[2, 2, 2]),
            Chair(position=[13.5, 4.2, -1.5], eulers=[-90, 90, 0], scale=[2, 2, 2]),
            Chair(position=[1, 7.2, -1.5], eulers=[-90, -30, 0], scale=[2, 2, 2]),
        ]
        self.renderables[OBJECT_CAT] = [
            Cat(position=[1, 1, -0.75], eulers=[-90, 210, 0], scale=[0.025, 0.025, 0.025]),
            Cat(position=[13.5, 7.2, -0.75], eulers=[-90, 30, 0], scale=[0.025, 0.025, 0.025]),
            ]


        #  Create Robot 1
        first_arm = self.renderables[OBJECT_ARM][0]
        self.renderables[OBJECT_STANDING_FOR_ROBOT][0].enable_auto_rotation = False

        self.renderables[OBJECT_STANDING_FOR_ROBOT][0].attach_arm(first_arm)


        # Create the second arm attached to the first arm
        second_arm = self.renderables[OBJECT_ARM][1]
        first_arm.attach_arm(second_arm)
        ending_of_arm_1 = self.renderables[OBJECT_ENDING_OF_ARM][0]
        ending_of_arm_2 = self.renderables[OBJECT_ENDING_OF_ARM][1]
        ending_of_arm_1.attach_to(second_arm)
        ending_of_arm_1.offset = [0, 0.04, -0.40]
        ending_of_arm_2.attach_to(second_arm)
        ending_of_arm_2.offset = [0, -0.04, -0.40]


        #  Create Robot 2
        third_arm = self.renderables[OBJECT_ARM][2]
        self.renderables[OBJECT_STANDING_FOR_ROBOT][1].attach_arm(third_arm)
        # Create the second arm attached to the first arm
        forth_arm = self.renderables[OBJECT_ARM][3]
        third_arm.attach_arm(forth_arm)

        self.renderables[OBJECT_SPHERE] = [
            LightEntity(position=positions_of_lights[0], eulers=[0, 0, 0], scale=[1, 1, 1]),
            LightEntity(position=positions_of_lights[1], eulers=[0, 0, 0], scale=[0.1, 0.1, 0.1]),
            LightEntity(position=positions_of_lights[2], eulers=[0, 0, 0], scale=[0.1, 0.1, 0.1]),
            ]
        self.renderables[ARM_ROBOT] = [
            ArmOtherRobot(position=[1, 1, 1], eulers=[0, 0, 0], scale=[0.2, 0.6, 0.2]),
        ]
        self.renderables[OBJECT_CYLINDER] = [
            Cylinder(position=[10.2, 5.3, -0.15], eulers=[0, 0, 0], scale=[1, 1, 1]),
            Cylinder(position=[9.1, 5.4, -0.15], eulers=[0, 0, 0], scale=[1, 1, 1]),
            Cylinder(position=[12.6, 5.6, -0.15], eulers=[0, 0, 0], scale=[1, 1, 1]),
            Cylinder(position=[9.9, 5.2, -0.15], eulers=[0, 0, 0], scale=[1, 1, 1]),
        ]
        self.renderables[OBJECT_SHAPE_O] = [
            ShapeO(position=[9.8, 5.6, -0.15], eulers=[0, 0, 0], scale=[1, 1, 1]),
            ShapeO(position=[11.6, 5.4, -0.15], eulers=[0, 0, 0], scale=[1, 1, 1]),
            ShapeO(position=[9.3, 5.7, -0.15], eulers=[0, 0, 0], scale=[1, 1, 1]),
            ShapeO(position=[12.4, 5.6, -0.15], eulers=[0, 0, 0], scale=[1, 1, 1]),
        ]


        self.lights = [
            Light(position=positions_of_lights[0], color=[1, 1, 1], strength=0.2),
            Light(position=positions_of_lights[1], color=[1, 1, 1], strength=0.2),
            Light(position=positions_of_lights[2], color=[1, 1, 1], strength=0.2),
            Light(position=positions_of_lights[3], color=[1, 1, 1], strength=0.1),
            Light(position=positions_of_lights[4], color=[0.7, 1, 1], strength=0.1),
            Light(position=positions_of_lights[5], color=[0.7, 1, 1], strength=0),
            Light(position=positions_of_lights[6], color=[0.7, 1, 1], strength=0),
            Light(position=positions_of_lights[7], color=[0.7, 1, 1], strength=0),
        ]

        self.camera = self.renderables[OBJECT_PLAYER][0]

    def update(self) -> None:
        """ 
            Update all objects managed by the scene.
        """

        for _,objectList in self.renderables.items():
            for object in objectList:
                object.update()
        
        self.camera.update()

    def move_camera(self, dPos: np.ndarray) -> None:
        """ Moves the camera by the given amount """

        self.camera.position += dPos
    
    def spin_camera(self, dEulers: np.ndarray) -> None:
        """ 
            Change the camera's euler angles by the given amount,
            performing appropriate bounds checks.
        """

        self.camera.eulers += dEulers

        if self.camera.eulers[2] < 0:
            self.camera.eulers[2] += 360
        elif self.camera.eulers[2] > 360:
            self.camera.eulers[2] -= 360
        
        self.camera.eulers[1] = min(89, max(-89, self.camera.eulers[1]))


################### Control ###################################################
class App:
    """ The main program """

    def __init__(self, screenWidth, screenHeight):
        """ Set up the program """
        self.grabbed = False
        self.grabbed_object = None
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight

        self.set_up_pygame()

        self.make_objects()

        self.set_up_input_systems()

        self.mainLoop()


    def set_up_pygame(self) -> None:
        """ Set up the pygame environment """
        
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(
            (self.screenWidth, self.screenHeight), 
            pg.OPENGL|pg.DOUBLEBUF
        )
        pg.display.set_caption("Program na graficzke")

        self.clock = pg.time.Clock()

    def make_objects(self) -> None:
        """ Make any object used by the App"""

        self.scene = Scene()

        self.renderer = Renderer(self.screenWidth, self.screenHeight, self.scene)
    
    def set_up_input_systems(self) -> None:
        """ Run any mouse/keyboard configuration here. """

        pg.mouse.set_visible(False)
        pg.mouse.set_pos(
            self.screenWidth // 2, 
            self.screenHeight // 2
        )

        self.walk_offset_lookup = {
            1: 0,
            2: 90,
            3: 45,
            4: 180,
            6: 135,
            7: 90,
            8: 270,
            9: 315,
            11: 0,
            12: 225,
            13: 270,
            14: 180
        }

    def mainLoop(self) -> None:
        """ Run the App """

        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
            
            self.handle_keys()
            self.handleMouse()

            #  update scene
            self.scene.update()
            
            self.renderer.render(
                camera=self.scene.camera,
                renderables=self.scene.renderables
            )

            #  timing
        self.quit()

    def handle_keys(self) -> None:
        """
            Handle keys.
        """

        combo = 0

        keys = pg.key.get_pressed()

        if keys[pg.K_w]:
            combo += 1
        if keys[pg.K_a]:
            combo += 2
        if keys[pg.K_s]:
            combo += 4
        if keys[pg.K_d]:
            combo += 8

        # 1
        elif keys[pg.K_1] and keys[pg.K_DOWN]:
            if self.scene.lights[0].strength > 0:
                self.scene.lights[0].strength -= 0.01

        elif keys[pg.K_1] and keys[pg.K_UP]:
            if self.scene.lights[0].strength < 1:
                self.scene.lights[0].strength += 0.01

        # 2
        elif keys[pg.K_2] and keys[pg.K_DOWN]:
            if self.scene.lights[1].strength > 0:
                self.scene.lights[1].strength -= 0.01

        elif keys[pg.K_2] and keys[pg.K_UP]:
            if self.scene.lights[1].strength < 1:
                self.scene.lights[1].strength += 0.01

        # 3
        elif keys[pg.K_3] and keys[pg.K_DOWN]:
            if self.scene.lights[2].strength > 0:
                self.scene.lights[2].strength -= 0.01

        elif keys[pg.K_3] and keys[pg.K_UP]:
            if self.scene.lights[2].strength < 1:
                self.scene.lights[2].strength += 0.01

        # 4

        elif keys[pg.K_4] and keys[pg.K_DOWN]:
            if self.scene.lights[3].strength > 0:
                self.scene.lights[3].strength -= 0.01

        elif keys[pg.K_4] and keys[pg.K_UP]:
            if self.scene.lights[3].strength < 1:
                self.scene.lights[3].strength += 0.01

        # 5

        elif keys[pg.K_5] and keys[pg.K_DOWN]:
            if self.scene.lights[4].strength > 0:
                self.scene.lights[4].strength -= 0.01

        elif keys[pg.K_5] and keys[pg.K_UP]:
            if self.scene.lights[4].strength < 1:
                self.scene.lights[4].strength += 0.01

        elif keys[pg.K_r] and keys[pg.K_UP]:
            self.scene.renderables[OBJECT_ARM][0].rotate_by_player += 1
        elif keys[pg.K_r] and keys[pg.K_DOWN]:
            if self.scene.renderables[OBJECT_ENDING_OF_ARM][0].position[2] >= self.scene.renderables[OBJECT_TABLE][0].position[2] + 1.35:
                self.scene.renderables[OBJECT_ARM][0].rotate_by_player -= 1

        elif keys[pg.K_r] and keys[pg.K_LEFT]:
            self.scene.renderables[OBJECT_ARM][0].rotate_child += 1
        elif keys[pg.K_r] and keys[pg.K_RIGHT]:
            if self.scene.renderables[OBJECT_ENDING_OF_ARM][0].position[2] >= self.scene.renderables[OBJECT_TABLE][0].position[2] + 1.35:
                self.scene.renderables[OBJECT_ARM][0].rotate_child -= 1
        elif keys[pg.K_UP]:
            if self.scene.renderables[OBJECT_STANDING_FOR_ROBOT][0].position[0] >= 7.6:
                self.scene.renderables[OBJECT_STANDING_FOR_ROBOT][0].position[0] += 0.01
        elif keys[pg.K_DOWN]:
            if self.scene.renderables[OBJECT_STANDING_FOR_ROBOT][0].position[0] <= 13.6:
                self.scene.renderables[OBJECT_STANDING_FOR_ROBOT][0].position[0] -= 0.01

        elif keys[pg.K_SPACE]:
            if not self.grabbed:
                for i in PICKABLE_OBJECTS:
                    for j, object in enumerate(self.scene.renderables[i]):
                        if abs(object.position[0] - self.scene.renderables[OBJECT_ENDING_OF_ARM][0].position[0]) <= 0.1 \
                            and abs(object.position[1] - self.scene.renderables[OBJECT_ENDING_OF_ARM][0].position[1]) <= 0.1 \
                            and abs(object.position[2] - self.scene.renderables[OBJECT_ENDING_OF_ARM][0].position[2]) <= 0.1:
                            object.attach_to(self.scene.renderables[OBJECT_ENDING_OF_ARM][1])
                            self.grabbed = True
                            self.grabbed_object = object
                            self.scene.renderables[OBJECT_ENDING_OF_ARM][0].offset = [0, -0.03, -0.40]
                            self.scene.renderables[OBJECT_ENDING_OF_ARM][1].offset = [0, 0.03, -0.40]

            else:
                self.scene.renderables[OBJECT_ENDING_OF_ARM][0].offset = [0, -0.04, -0.40]
                self.scene.renderables[OBJECT_ENDING_OF_ARM][1].offset = [0, 0.04, -0.40]
                self.grabbed_object.is_falling_down = True
                self.grabbed_object.detach()
                self.grabbed = False
                self.grabbed_object = None

        if combo in self.walk_offset_lookup:

            directionModifier = self.walk_offset_lookup[combo]
            
            dPos = 0.1 * np.array(
                [
                    np.cos(np.deg2rad(self.scene.camera.eulers[2] + directionModifier)),
                    np.sin(np.deg2rad(self.scene.camera.eulers[2] + directionModifier)),
                    0
                ],
                dtype=np.float32
            )

            self.scene.move_camera(dPos)



    def handleMouse(self) -> None:
        """
            Handle mouse movement.
        """

        (x,y) = pg.mouse.get_pos()
        theta_increment = (self.screenWidth / 2.0) - x
        phi_increment = (self.screenHeight / 2.0) - y
        dEulers = np.array([0, phi_increment * 0.1, theta_increment * 0.1], dtype=np.float32)
        self.scene.spin_camera(dEulers)
        pg.mouse.set_pos(self.screenWidth // 2, self.screenHeight // 2)

    def quit(self):
        
        self.renderer.destroy()
        pg.quit()

################### View  #####################################################


class Renderer:

    def __init__(self, screenWidth: int, screenHeight: int, scene: Scene):

        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.scene = scene

        self.set_up_opengl()
        
        self.make_assets()

        self.set_onetime_uniforms()

        self.get_uniform_locations()

    def set_up_opengl(self) -> None:
        """
            Set up any general options used in OpenGL rendering.
        """

        glClearColor(0.0, 0.0, 0.0, 1)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def make_assets(self) -> None:
        """
            Load/Create assets (eg. meshes and materials) that 
            the renderer will use.
        """

        self.meshes: dict[int, Mesh] = {
            OBJECT_FLOOR_TILE: ObjMesh("models/floortile.obj"),
            OBJECT_WALL: ObjMesh("models/wall.obj"),
            OBJECT_RACK: ObjMesh("models/armrobot.obj"),
            OBJECT_TEXT: ObjMesh("models/text.obj"),
            OBJECT_TABLE: ObjMesh("models/table.obj"),
            OBJECT_STANDING_FOR_ROBOT: ObjMesh("models/standing_for_robot.obj"),
            OBJECT_ARM: ObjMesh("models/arm.obj"),
            OBJECT_SPHERE: ObjMesh("models/sphere.obj"),
            OBJECT_PLAYER: ObjMesh("models/player.obj"),
            OBJECT_CHAIR: ObjMesh("models/chair.obj"),
            OBJECT_CAT: ObjMesh("models/cat1.obj"),
            ARM_ROBOT: ObjMesh("models/armrobot.obj"),
            OBJECT_ENDING_OF_ARM: ObjMesh("models/floortile.obj"),
            OBJECT_CYLINDER: ObjMesh("models/cylinder.obj"),
            OBJECT_SHAPE_O: ObjMesh("models/object_shape_o.obj"),
        }

        self.materials: dict[int, Material] = {
            OBJECT_FLOOR_TILE: Material("gfx/floortile.jpg"),
            OBJECT_WALL: Material("gfx/wall.jpg"),
            OBJECT_RACK: Material("gfx/metal.jpg"),
            OBJECT_TEXT: Material("gfx/robot_texture.jpg"),
            OBJECT_TABLE: Material("gfx/blackmarble.jpg"),
            OBJECT_STANDING_FOR_ROBOT: Material("gfx/robot_texture.jpg"),
            OBJECT_ARM: Material("gfx/robot_texture.jpg"),
            OBJECT_SPHERE: Material("gfx/light.jpg"),
            OBJECT_PLAYER: Material("gfx/wood.jpeg"),
            OBJECT_CHAIR: Material("gfx/wood.jpeg"),
            OBJECT_CAT: Material("gfx/Cat_diffuse.jpg"),
            ARM_ROBOT: Material("gfx/robot2_texture.jpg"),
            OBJECT_ENDING_OF_ARM: Material("gfx/robot2_texture.jpg"),
            OBJECT_SHAPE_O: Material("gfx/robot2_texture.jpg"),
            OBJECT_CYLINDER: Material("gfx/robot2_texture.jpg"),
        }

        self.shader = create_shader("shaders/vertex.glsl", "shaders/fragment.glsl")

    def set_onetime_uniforms(self) -> None:
        """ Set any uniforms which can simply get set once and forgotten """
        
        glUseProgram(self.shader)
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=75, aspect=self.screenWidth / self.screenHeight,
            near=0.1, far=100, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"), 
            1, GL_FALSE, projection_transform
        )
        glUniform1i(
            glGetUniformLocation(self.shader, "imageTexture"), 0)
        self.light_location = {
            "position": [
                glGetUniformLocation(self.shader, f"Lights[{i}].position")
                for i in range(8)
                ],
            "color": [
                glGetUniformLocation(self.shader, f"Lights[{i}].color")
                for i in range(8)
                ],
            "strength": [
                glGetUniformLocation(self.shader, f"Lights[{i}].strength")
                for i in range(8)
                ],
        }
        self.camera_location = glGetUniformLocation(self.shader, "cameraPosition")

    def get_uniform_locations(self) -> None:
        """ 
            Query and store the locations of any uniforms 
            on the shader 
        """

        glUseProgram(self.shader)
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
    
    def render(self, camera: Player, renderables: dict[int, list[Entity]]) -> None:
        """
            Render a frame.

            Parameters:

                camera: the camera to render from

                renderables: a dictionary of entities to draw, keys are the
                            entity types, for each of these there is a list
                            of entities.
        """

        #  refresh screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

        glUniformMatrix4fv(
            self.viewMatrixLocation, 
            1, GL_FALSE, camera.get_view_transform()
        )
        for i, light in enumerate(self.scene.lights):
            glUniform3fv(self.light_location["position"][i], 1, light.position)
            glUniform3fv(self.light_location["color"][i], 1, light.color)
            glUniform1f(self.light_location["strength"][i], light.strength)

        glUniform3fv(self.camera_location, 1, camera.position)

        for objectType,objectList in renderables.items():
            mesh = self.meshes[objectType]
            material = self.materials[objectType]
            glBindVertexArray(mesh.vao)
            material.use()
            for object in objectList:
                glUniformMatrix4fv(
                    self.modelMatrixLocation,
                    1, GL_FALSE,
                    object.get_model_transform()
                )
                glDrawArrays(GL_TRIANGLES, 0, mesh.vertex_count)

        pg.display.flip()

    def destroy(self) -> None:
        """ Free any allocated memory """

        for (_, mesh) in self.meshes.items():
            mesh.destroy()
        for (_, material) in self.materials.items():
            material.destroy()
        glDeleteProgram(self.shader)


class Light:
    """ A light in the scene """

    def __init__(self, position: list[float], color: list[float], strength: float):

        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength

    def destroy(self):
        pass


class Mesh:
    """ A general mesh """

    def __init__(self):

        self.vertex_count = 0

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

    def destroy(self):
        
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


class ObjMesh(Mesh):

    def __init__(self, filename):

        super().__init__()

        # x, y, z, s, t, nx, ny, nz
        vertices = load_model_from_file(filename)
        self.vertex_count = len(vertices)//8
        vertices = np.array(vertices, dtype=np.float32)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        #  position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #  texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #  normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))


class Material:

    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(filepath).convert()
        image_width, image_height = image.get_rect().size
        img_data = pg.image.tostring(image, 'RGBA')
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        glDeleteTextures(1, (self.texture,))


myApp = App(1260, 960)