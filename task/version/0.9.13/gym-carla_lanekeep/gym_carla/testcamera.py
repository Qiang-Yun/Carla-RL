import abc
import glob
import os
import sys
from types import LambdaType
from collections import deque
from collections import namedtuple
 
import sys
try:
  sys.path.append('/home/yq/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')
except IndexError:
    pass
import carla

import random 
import time
import numpy as np
import cv2
import math
 
IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_PREVIEW = True
 

class Car_Env():
    SHOW_CAM = SHOW_PREVIEW  #设置是否显示摄像头画面
    STEER_AMT = 1.0
    im_width = IM_WIDTH  #设置摄像头传感器图像宽度
    im_height = IM_HEIGHT  #设置摄像头传感器图像高度
    front_camera = None
 
    def __init__(self):
        self.client = carla.Client('localhost',2000)  #与carla模拟器建立连接
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()  #读取carla已经定义好的环境-world
        self.blueprint_library = self.world.get_blueprint_library()  #读取carla中的blue-print库，包括车辆，传感器等
        self.model_3 = self.blueprint_library.filter('model3')[0]  #选取一辆车作为actor
    
    #初始化场景
    def reset(self):
        self.collision_hist = []
        self.radar_hist = []
        self.actor_list = []
        self.transform = self.world.get_map().get_spawn_points()[100]
        #spwan_points共265个点，随机选一个点可作为初始化小车的位置
        
        self.vehicle = self.world.spawn_actor(self.model_3 , self.transform)
 
        self.actor_list.append(self.vehicle)
        
        #定义传感器——RGB三通道摄像头
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x',f'{self.im_width}')  #设置图像宽度
        self.rgb_cam.set_attribute('image_size_y',f'{self.im_height}')  #设置图像高度
        self.rgb_cam.set_attribute('fov',f'110')  #设置摄像头位置，fov(front of view)表示前置摄像头
 
        transform = carla.Transform(carla.Location(x=2.5 ,z=0.7 ))
        self.sensor = self.world.spawn_actor(self.rgb_cam,transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
 
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        time.sleep(4)
 
        
        #collision sensor
        colsensor = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to = self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
       
 
        #target_transform
        self.target_transform = self.world.get_map().get_spawn_points()[101]
        self.target_dis = self.target_transform.location.distance(self.vehicle.get_location())
    
        while self.front_camera is None:
            time.sleep(0.01)
 
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
               
        return self.front_camera
        
    def collision_data(self, event):
        self.collision_hist.append(event)
     
                
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width , 4))
        i3 = i2[: , : , : 3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.front_camera = i3
        return i3/255.0

def reset():
        client = carla.Client('localhost',2000)  #与carla模拟器建立连接
        client.set_timeout(10.0)
        world = client.get_world()  #读取carla已经定义好的环境-world
        blueprint_library = world.get_blueprint_library()  #读取carla中的blue-print库，包括车辆，传感器等
        model_3 = blueprint_library.filter('model3')[0]  #选取一辆车作为actor

if __name__ == '__main__':
    env=Car_Env()
    s=env.reset()