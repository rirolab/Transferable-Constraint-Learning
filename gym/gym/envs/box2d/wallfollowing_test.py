import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, prismaticJointDef)

import gym
from gym import spaces
#from gym.envs.box2d.car_dynamics import Car
from gym.utils import colorize, seeding, EzPickle
from collections import deque

import time
VIEWPORT_W = 900
VIEWPORT_H = 900
FPS    = 30
SCALE  = 40.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_NUM = 8
LIDAR_RANGE   = 200/SCALE

INITIAL_RANDOM = 5

CAR_POLY= [
        (-5,+10), (+5, +10), (+5, -10), (-5, -10)
        ]
HULL_POLY =[
    (-20,+15), (+0,+30), (+20,+15),
    (+20,-3), (-20,-3)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE

WINDOW_W = 600
WINDOW_H = 600

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5


CAR_FD = fixtureDef(
                # shape=circleShape(center=(0,0), radius=10/SCALE),
                shape=circleShape(radius=5/SCALE, pos=(0,0)),
                # shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in CAR_POLY ]),
                density=1.0,
                friction=0.0,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy


HULL_FD = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.0,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy

LEG_FD = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

class Car:
    def __init__(self,world, init_angle=0.0, init_x=0.0, init_y=0.0, vel=(0.0, 0.0), w=0.0):
        self.world = world
        self.hull = self.world.CreateDynamicBody(
                position=(init_x, init_y),
                angle=init_angle,
                fixtures = CAR_FD,
                fixedRotation = True,
                )
        self.hull.color=(1.0, 0.0, 0.0)
        self.gas = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.omega = 0.0
        self.phase = 0.0
        self.drawlist = []
        self.drawlist += [self.hull]
        self.hull.linearVelocity = vel
        self.hull.angularVelocity = w

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None

    def brake(self, b):
        self.brake = b
    
    def steer(self, s):
        self.steer = s

    def step(self, dt):
        self.hull.linearVelocity = (self.gas, self.steer)


    def draw(self, viewer):
        from gym.envs.classic_control import rendering
        for obj in self.drawlist:
            for f in obj.fixtures:
                # trans = f.body.transform
                # path = [trans*v for v in f.shape.vertices]
                # viewer.draw_polygon(path, color=(0,0,0))
                
            
                if isinstance(f.shape, circleShape):
                    trans = f.body.transform
                    t = rendering.Transform(translation=trans*f.shape.pos)

                    viewer.draw_circle(f.shape.radius*2.3, 30, color=(0,0,0)).add_attr(t)
                    viewer.draw_circle(f.shape.radius*2, 30, color=(1,1,0)).add_attr(t)
                    # t = f.body.transform
                    # viewer.draw_circle(f.shape.radius, 30, pos=t,color=(1,0,0) )#.add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    viewer.draw_polygon(path, color=(1,0,0))

    def getLinearVelocity(self):
        return self.hull.linearVelocity

    def getAngularVelocity(self):
        return self.hull.angularVelocity

    def _getFrontalVelocity(self):
        normal = self.hull.GetWorldVector( (0,1) )
        vel = self.hull.linearVelocity
        #print(self.world.dot(normal, vel))
        return normal.dot(vel)*normal


    def _getLateralVelocity(self):
        normal = self.hull.GetWorldVector( (1,0) )
        vel = self.hull.linearVelocity
        #print(self.world.dot(normal, vel))
        return normal.dot(vel)*normal

    def getPosition(self):
        return self.hull.position

    def getFrontVector(self):
        return self.hull.GetWorldVector( (0,1))

    def getLateralVector(self):
        return self.hull.GetWorldVector( (1,0))

    def getCustomVector(self, x, y ):
        vec = self.hull.GetWorldVector( (x,y))
        vec /= vec.Normalize()
        return vec 

    def _updateFriction(self):
        impulse = self.hull.mass* -1.0 * self._getLateralVelocity()
        ang_impulse = -0.1 * self.hull.inertia * self.hull.angularVelocity
        return impulse, ang_impulse


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.car.hull==contact.fixtureA.body or self.env.car.hull==contact.fixtureB.body:
            self.env.game_over = 'lose'

class WallFollowingTest(gym.Env):
    metadata = { 'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second':30}

    def __init__(self, speed_limit=-1):
        # self.action_space = spaces.Discrete(6)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        obs_num =2 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_num,), dtype=np.float32)
        self.viewer = None
        self.car = None
        self.current_max = None
        if speed_limit == -1:
            self.speed_limit = float('inf')

        self.world = Box2D.b2World(gravity=(0,0))
        self.bg = None
        self.track = None
        self.goal = None
        self.game_over = 'in_game'
        self.time = 0
        self.obs = None
        self.trajectory = None
        self.trajectories =  deque(maxlen=10)

        self.clouds = []
    def _destroy(self):
        self.world.contactListener = None
        if self.track is not None:
            for t in self.track:
                self.world.DestroyBody(t)
        self.track = None
        if self.car is not None:
            self.car.destroy()
        if self.clouds is not None:
            for cloud in self.clouds:
                self.world.DestroyBody(cloud)

        if self.trajectory is not None:
            self.trajectories.append(self.trajectory.copy())
            # for t in self.trajectory:
                # self.world.DestroyBody(t)

        if len(self.trajectories) > 5:
            leftmost = self.trajectories.popleft()
            if leftmost is not None:
                for t in leftmost:
                    self.world.DestroyBody(t)
    def _generate_bg(self):
        # Sorry for the clouds, couldn't resist
        self.clouds   = []
       
        for x in np.linspace(-1.2, 1.2, 80):
            for y in np.linspace(-1.2, 1.2, 80):
                
                cloud = self.world.CreateStaticBody(position=(x * SCALE, y * SCALE))

                cloud.CreateCircleFixture(radius=0.0035 * SCALE,
                            categoryBits=0x0400,
                            maskBits=0x001)

                self.clouds.append(cloud)

    def _generate_obs(self):
        init_x= 0
        init_y= 0
        self.obs = self.world.CreateStaticBody(position=(init_x * SCALE, init_y * SCALE))
        
        self.obs.CreateCircleFixture(radius=0.05 * SCALE,
                    categoryBits=0x0001,
                    maskBits=0x001)
   
    def _generate_trajectory(self, segment=None, constraint=False):
        if segment is None: 
            self.trajectory = []
        else:
            
            segment_fixture = fixtureDef(
                    shape = edgeShape(vertices = 
                        segment),
                        # categoryBits=0x0001,
                        categoryBits=0x0400,
                        maskBits=0x001) 
            edge = self.world.CreateStaticBody(
                    fixtures = segment_fixture)

            color = (0, 1.0, 0) if constraint else (1, 0.0, 0.0)
            # color = (1, 0.0, 0.0)
            edge.color1 = color
            edge.color2 = color  
            self.trajectory.append(edge)
            
    def dist(self, x, y):
        def dist_equation(x1, y1, x2, y2, x0, y0):
            numerator = np.abs( (x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1) )
            denominator = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
            return numerator / denominator
       
        def point_to_line_segment_distance(x1, y1, x2, y2, x, y):
            # Calculate vector v = P2 - P1
            v = (x2 - x1, y2 - y1)
            
            # Calculate vector w = P - P1
            w = (x - x1, y - y1)
            
            # Calculate the dot product of v and w
            dot_product_vw = v[0] * w[0] + v[1] * w[1]
            
            # Calculate the dot product of v and v
            dot_product_vv = v[0] * v[0] + v[1] * v[1]
            
            # Calculate the parameter t
            t = dot_product_vw / dot_product_vv
            
            if t <= 0:
                # The closest point is P1
                q = (x1, y1)
            elif t >= 1:
                # The closest point is P2
                q = (x2, y2)
            else:
                # The closest point is on the line segment
                q = (x1 + t * v[0], y1 + t * v[1])
            
            # Calculate the distance between P and Q
            distance = math.sqrt((x - q[0])**2 + (y - q[1])**2)
            
            return distance , q
        x1, y1 = self._track_points[0]
        min_dist = np.sqrt( (x1 - x)**2 + (y1 - y)**2)
        min_pts = (x1, y1)
        for x2, y2 in self._track_points[1:]:
            d1, mp1 = point_to_line_segment_distance(x1, y1, x2, y2, x,y)
            d2 = np.sqrt( (x2 - x)**2 + (y2 - y)**2 )
            if min_dist >= d2:
                min_pts = (x2, y2)
                min_dist = d2
            if min_dist >= d1:
                min_pts = (mp1[0], mp1[1])
                min_dist = d1
            # min_dist = min(min_dist, d1, d2)
            x1, y1 = x2, y2
        return min_dist, min_pts
    
    def is_constraint(self, x, y):
        min_dist, min_pts = self.dist(x,y)
        if min_dist > 0.090 and min_dist < 0.11:
            return False
        return True

    def set_states(self, car=None, goal=None, obs=None):
        if car is not None:
            self.car.hull.position = (car[0] * SCALE, car[1] * SCALE)
        if goal is not None:
            self.goal.position = goal * SCALE
        if obs is not None:
            self.obs.position = obs * SCALE
        return self.step([0, 0])
        
    def _generate_track(self):
        self.track = []
        # Not random yet
        self.track_edge = fixtureDef(
                shape = edgeShape(vertices = 
                    [(0, 0),
                    (1, 1)]),
                    # categoryBits=0x0001,
                    categoryBits=0x0400,
                    maskBits=0x001)
        # self._track_points= [(-1,-1), (-1, 1), (1, 1), (1, -1), (-1, -1) ]
        self._track_points = []
        
        def half_circle_generator(center_x, center_y, radius, smooth, right = True):
            res = []
            for i in range(smooth+1):
                #angle1 = math.pi * i / smooth
                angle2 = 0.66 * math.pi * (i) / smooth
                if not right:
                    angle2 += math.pi/2 + 0.16*math.pi
                #x1 = center_x + radius * math.sin(angle1)
                #y1 = center_x - radius * math.cos(angle1)
                x2 = center_x + radius * math.sin(angle2)
                y2 = center_y - radius * math.cos(angle2)
                res.append((x2,y2) )
            print(res)
            return res

        def quad_circle_generator(center_x, center_y, radius, smooth, right = True):
            res = []
            for i in range(smooth):
                #angle1 = math.pi * i / smooth
                angle2 = -0.4 * math.pi * (i-1) / smooth
                if not right:
                    angle2 +=0# math.pi/2 #+ math.pi/4
                #x1 = center_x + radius * math.sin(angle1)
                #y1 = center_x - radius * math.cos(angle1)
                x2 = center_x + radius * math.sin(angle2)
                y2 = center_y - radius * math.cos(angle2)
                res.append((x2,y2) )
            res.reverse()
            return res

        def qquad_circle_generator(center_x, center_y, radius, smooth, right = True):
            res = []
            for i in range(smooth):
                #angle1 = math.pi * i / smooth
                angle2 = -0.4 * math.pi * (i+1) / smooth
                if not right:
                    angle2 += 0.4 * math.pi # math.pi/2 #+ math.pi/4
                #x1 = center_x + radius * math.sin(angle1)
                #y1 = center_x - radius * math.cos(angle1)
                x2 = center_x + radius * math.sin(angle2)
                y2 = center_y - radius * math.cos(angle2)
                res.append((x2,y2) )
            res.reverse()
            return res
        def spl_generator(p1, p2, p3, x_init, x_fin, num_points):
            import numpy as np
            from scipy.interpolate import splrep, splev
            import matplotlib.pyplot as plt

            # Define the three points
            x_points = np.array([p1[0], p2[0], p3[0]])
            y_points = np.array([p1[1], p2[1], p3[1]])

            # Generate the spline representation (tck) for quadratic spline (k=2)
            tck = splrep(x_points, y_points, k=2)

            # Evaluate the spline at a series of x values
            x_new = np.linspace(x_init, x_fin, num_points)
            y_new = splev(x_new, tck)
            spline_points = list(zip(x_new, y_new))
            return spline_points
        def qhl_generator(p1, p2, p3, p4, num_points):
            from scipy.interpolate import UnivariateSpline, PchipInterpolator

            x_points = np.array([p1[0], p2[0], p3[0], p4[0]])
            y_points = np.array([p1[1], p2[1], p3[1], p4[1]])
            dydx_2nd_point = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
            dydx_3rd_point = (y_points[3] - y_points[2]) / (x_points[3] - x_points[2])

            # Create a Cubic Hermite Spline using the 2nd and 3rd points and their slopes
            # hermite_spline = CubicHermiteSpline(x_points[1:3], y_points[1:3], [dydx_2nd_point, dydx_3rd_point])
            hermite_spline = PchipInterpolator(x_points, y_points)

            # Evaluate the spline at a series of x values
            x_new = np.linspace(x_points[1], x_points[2], num_points)
            y_new = hermite_spline(x_new)

            # Combine the x and y coordinates into a list of tuples
            spline_points = list(zip(x_new, y_new))

            return spline_points

        # self._track_points= [(-1,-1), (-1, 1), (1, 1), (1, -1), (-1, -1) ]
        # self._track_points= []
        # self._track_points =[ (1, 0), (0.7, 0.28) ]

        def bump_function(x, x_min, x_max, y_max):
            # Find the midpoint and scale for the Gaussian
            x_mid = (x_min + x_max) / 2
            sigma = (x_max - x_min) / 6  # Controls the width of the bump

            # Create the bump using a Gaussian function
            y = np.where(x < 0, 
                            np.where(x>= x_min,
                                     0.5*y_max * np.exp(-((x - x_mid) ** 2) / (2 * sigma ** 2))+0.5*y_max,
                                     y_max*0.5),
                            np.where(x< x_max,
                                     y_max * np.exp(-((x - x_mid) ** 2) / (2 * sigma ** 2)),
                                     0.0) ,
                         )
            y = np.where((x >= x_min) & (x <= x_max), 
                                     y_max * np.exp(-((x - x_mid) ** 2) / (2 * sigma ** 2)) , 
                         0)

            return y

        # Parameters for the bump function
        x_min = -0.2
        x_max = 0.2
        y_max = 0.2
        self._track_points = [(-0.7, y_max*0.5)]

        # Evaluate the bump function at a series of x values
        x_values = np.linspace(-0.5, 0.5, 200)
        y_values = bump_function(x_values, x_min, x_max, y_max)
        lsls = list(zip(x_values, y_values))
        self._track_points += lsls#sine_generater(2, 100, 1.0, 0.0)
        self._track_points += [(0.7, 0.)]
        self.track_points = []

        self.track_points = []
        for x, y in self._track_points:
            self.track_points.append( (x*SCALE, y*SCALE))
        self.track_poly=[]
        for i in range(len(self.track_points)-1):
            poly = [
                    (self.track_points[i][0], self.track_points[i][1]),
                    (self.track_points[i+1][0], self.track_points[i+1][1])
                    ]
            self.track_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                    fixtures = self.track_edge)

            color = (1.0, 0,0)
            t.color1 = color
            t.color2 = color
            self.track.append(t)
            #poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.track_poly.append( (poly, color))


    def _generate_goal(self):
        init_x= +0.40
        init_y= 0.10

        while True:
            
            init_x = +0.40
            init_y = 0.10
            init_x += np.random.uniform(-0.01, 0.01)
            init_y += np.random.uniform(-0.01, 0.01) 
            # break
            if not self.is_constraint(init_x, init_y):
                break
        
        self.goal = self.world.CreateStaticBody(position=(init_x * SCALE, init_y * SCALE))
        
        self.goal.CreateCircleFixture(radius=0.01 * SCALE,
                    categoryBits=0x0020,
                    maskBits=0x001)

    def _generate_car(self, configs=None):
        if configs is None:
            # init_x = np.random.uniform(100, 400)/SCALE
            # init_x= np.random.choice([0.9, -0.9])
            
            init_x = -0.30
            y_max = 0.0
            init_y = 0.10 + 0.5 * y_max
            while True:
                
                init_x = -0.30
                init_y = 0.10  + 0.5 * y_max
                init_x += np.random.uniform(-0.01, 0.01)
                init_y += np.random.uniform(-0.01, 0.01) 
                if not self.is_constraint(init_x, init_y):
                    break
                # break
            init_x, init_y = init_x*SCALE, init_y*SCALE
                
            self.car = Car(self.world, 0, init_x, init_y)
        else:
            self.car = Car(self.world, configs['angle'], configs['x'], configs['y'],
                    vel = configs['vel'], w = configs['w'])
        self.drawlist = []

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self, ckpt=None):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self._generate_track()
        self._generate_car(ckpt)
        self._generate_goal()
        self._generate_trajectory()

        # self._generate_bg()
        
        self.game_over='in_game'
        self.time = 0
        self.current_max = 0

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction
        self.lidar = [LidarCallback() for _ in range(LIDAR_NUM)]

        state = [self.car.getLinearVelocity().length, 
                self.car.getAngularVelocity()]
        for i in range(len(self.lidar)):
            angle = 2 * math.pi * i / (len(self.lidar))
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = self.car.getPosition()
            self.lidar[i].p2 = self.lidar[i].p1 + self.car.getCustomVector(math.sin(angle),math.cos(angle)) * LIDAR_RANGE
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state += [l.fraction for l in self.lidar]

        return self.step([0, 0])[0]

    def step(self, action, dt=3.0):
        self.time +=1
        a = action
        old_car_pos = self.car.hull.position.copy()
        
        upper_limit = 1.3
        lower_limit = -1.3
        right_limit = 1.3
        left_limit = -1.3
        if  self.car.hull.position[0] <= left_limit * SCALE:
            a[0] = max(0, a[0])
        if  self.car.hull.position[0] >= right_limit * SCALE:
            a[0] = min(0, a[0])
            
        if  self.car.hull.position[1] <= lower_limit * SCALE :
            a[1] = max(0, a[1])
            
        if  self.car.hull.position[1] >= upper_limit * SCALE:
            a[1] = min(0, a[1])
        
        self.car.gas = 2 * a[0]
        self.car.steer= 2 * a[1]
        self.car.step(1.0/dt)
        self.world.Step(1.0/dt, 30, 30)
        done = False

        self.car.gas = 0
        self.car.steer= 0
        self.car.step(1.0/dt)

        if  self.car.hull.position[0] <= left_limit * SCALE:
            self.car.hull.position[0] =  left_limit * SCALE
        if  self.car.hull.position[0] >= right_limit * SCALE:
            self.car.hull.position[0] = right_limit * SCALE
        if  self.car.hull.position[1] <= lower_limit * SCALE :
            self.car.hull.position[1] = lower_limit * SCALE 
        if  self.car.hull.position[1] >= upper_limit * SCALE:
            self.car.hull.position[1] = upper_limit * SCALE
            
        x, y = self.car.hull.position / SCALE
        
        rel_pos = self.car.hull.position - self.goal.position
        ddd, ppp = self.dist(x,y)
        state = [rel_pos[0] / SCALE * 2, 
                rel_pos[1] / SCALE * 2,
                # self.dist(x, y)[0]
                ddd * 5
                # rel_obs[0],
                # rel_obs[1],
                ]
                # self.car.hull.position[0] - 250/SCALE,
                # self.car.hull.position[0] - 250/SCALE]
        
        if np.linalg.norm(rel_pos[0]) < SCALE * 0.05:
            reward = +1
        else:
            reward = -0.1 * np.linalg.norm(rel_pos[0])
            
        # rel_obs = self.car.hull.position - self.obs.position
        
        reward -= 1e-3 * np.linalg.norm(action)

        config = {}
        car_pos = self.car.hull.position
        goal_pos = self.goal.position
        config['angle'] = self.car.hull.angle
        config['x'] = self.car.hull.position[0]
        config['y'] = self.car.hull.position[1]
        
        config['w'] = self.car.getAngularVelocity()
        config['constraint'] = self.is_constraint(x,y)
        config['is_success'] = (np.linalg.norm(rel_pos) * SCALE < 20) * 1.0
        config['primary'] = reward
      
        def angle_with_y_axis(x1, y1, x2, y2):
            # Handle the vertical line case
            if x2 == x1:
                return 0 if y2 > y1 else 180

            # Compute the slope
            m = (y2 - y1) / (x2 - x1)

            # Compute the angle with the x-axis
            angle_with_x_axis = math.degrees(math.atan(abs(m)))

            # Compute the angle with the y-axis
            angle_with_y_axis = 90 - angle_with_x_axis

            return angle_with_y_axis 

        config['pose'] = (x, y, angle_with_y_axis(x, y, ppp[0], ppp[1]))
        segment = [ old_car_pos, car_pos]
        
        self._generate_trajectory(segment, config['constraint'])
        return np.array(state), reward, done, config

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.scroll = 0.0
        self.viewer.set_bounds(self.scroll - VIEWPORT_W/SCALE, VIEWPORT_W/SCALE + self.scroll, -VIEWPORT_H/SCALE * 0.5, 1.5 * VIEWPORT_H/SCALE)

        if self.car is not None:

            # self.viewer.draw_polyline(self.track_points, color=(1,0,0))
            for poly in self.track:
                """
                if poly[1][0] < self.scroll: continue
                if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
                self.viewer.draw_polygon(poly, color=color)
                """
                for f in poly.fixtures:
                    trans = f.body.transform
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polyline([path[0],path[1]], linewidth=4, color=(0,0,0))
            for edge in self.trajectory:
                """
                if poly[1][0] < self.scroll: continue
                if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
                self.viewer.draw_polygon(poly, color=color)
                """
                for f in edge.fixtures:
                    trans = f.body.transform
                    path = [trans*v for v in f.shape.vertices]

                    self.viewer.draw_polyline([path[0],path[1]], linewidth=2, color=edge.color1)
            
            # for old_traj in self.trajectories:
            #     for edge in old_traj:
            #         for f in edge.fixtures:
            #             trans = f.body.transform
            #             path = [trans*v for v in f.shape.vertices]

            #             self.viewer.draw_polyline([path[0],path[1]], linewidth=1, color=edge.color1, alpha=0.1)
            """
            for f in self.goal.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                self.viewer.draw_line(path[0],path[1], color=(0,1,0))
            """




            for f in self.goal.fixtures:
                trans = f.body.transform
                t = rendering.Transform(translation=trans*f.shape.pos)
                self.viewer.draw_circle(f.shape.radius*2, 30, color=(0,1,0)).add_attr(t)

            if self.obs is not None:
                for f in self.obs.fixtures:
                    trans = f.body.transform
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    # self.viewer.draw_circle(f.shape.radius*1.2, 30, color=(1.0,0.85,0.85)).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius*0.8, 30, color=(1.0,0.0,0.0)).add_attr(t)

            for obj in self.drawlist:
                for f in obj.fixtures:
                    trans = f.body.transform
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=(0,0,0))

            # for i in range(len(self.lidar)):
            #     l = self.lidar[i]
            #     self.viewer.draw_polyline( [l.p1, l.p2], color=(0,0,0), linewidth=0.5)

                    # t = f.body.transform
            for obj in self.clouds:
                for f in obj.fixtures:
                    trans = f.body.transform
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    x,y = trans*f.shape.pos / SCALE
                    if self.is_constraint(x, y):
                        self.viewer.draw_circle(f.shape.radius*2.3, 30, color=(1,0,0)).add_attr(t)
                    else:
                        self.viewer.draw_circle(f.shape.radius*2.3, 30, color=(0,1,0)).add_attr(t)
                    # self.viewer.draw_circle(f.shape.radius*2.3, 30, color=(0,1*self.dist(x,y),0)).add_attr(t)
                        
                    # self.viewer.draw_circle(f.shape.radius*2, 30, color=(1,1,0)).add_attr(t)
            """
            for obj in self.robot_bodies:
                for f in obj.fixtures:
                    trans = f.body.transform
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=(0,0,0))


            for obj in self.obj:
                for f in obj.fixtures:
                    trans = f.body.transform
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30).add_attr(t)
            """
               
                
            # for i in range(len(self.lidar)):
            #     l = self.lidar[i]
            #     angle = 2*math.pi * i / (len(self.lidar))
            #     # angle -= math.pi/2

            #     lidar_x1 = l.p1[0] + math.sin(angle) * 1.5
            #     lidar_y1 = l.p1[1] + math.cos(angle) * 1.5
                
            #     lidar_x2 = l.p1[0] * 0.7 + l.p2[0] * 0.3
            #     lidar_y2 = l.p1[1] * 0.7 + l.p2[1] * 0.3
            #     # lidar_point = l.p1 * 0.8 + l.p2 * 0.1
            #     t = rendering.Transform(translation=(lidar_x1, lidar_y1))
            #     self.viewer.draw_circle(0.35, color=(0,0,0)).add_attr(t)
            #     self.viewer.draw_circle(0.3, color=(1.0 , 1.0 * l.fraction, 1.0* l.fraction )).add_attr(t) 
            self.car.draw(self.viewer)
            
            x, y = self.car.hull.position / SCALE
            _, min_pts = self.dist(x, y)
            mx, my = min_pts
            self.viewer.draw_polyline( [(x*SCALE, y*SCALE),
                (mx*SCALE, my*SCALE)], color=(0,1,0), linewidth=0.5)
        return self.viewer.render(return_rgb_array = mode =='rgb_array')


if __name__=="__main__":
    from pyglet.window import key

    a = np.array([0.0,0.0])
    def key_press(k, mod):
        global restart
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = 1.0
        if k == key.DOWN:
            a[1] = -1.0

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1:
            a[0] = 0.0
        if k == key.RIGHT and a[0] == +1:
            a[0] = 0.0
        if k == key.UP and a[1] == +1:
            a[1] = 0.0
        if k == key.DOWN and a[1] == -1:
            a[1] = 0.0

    env = WallFollowingTest()
    env.reset()
    env.render()
    env.viewer.window.on_key_press =key_press
    env.viewer.window.on_key_release = key_release

    for x, y in env._track_points:
        print("{},{}".format(1000*x,1000*y))
    cnt = 0
    
    b = [0.0,0.0]
    while True:
        b[0] = a[0]
        b[1] = a[1]
        o, _, _, config = env.step(b)
        print(o)
        env.render()
        cnt += 1
        if cnt > 60:
            env.reset()
            cnt = 0
        time.sleep(0.1)


