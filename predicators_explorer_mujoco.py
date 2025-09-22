import sys
import numpy as np
# import pybullet as p
import time
import logging
from typing import List, Tuple, Optional, Sequence, Collection, Dict, Any, cast, Set
import random
import json

from predicators.structs import Action, Array, GroundAtom, Object, State, Type, ParameterizedOption, EnvironmentTask,\
    Image, Predicate, State, Type, Video
from predicators import utils
from predicators.settings import CFG
from gymnasium.spaces import Box

#Import core environment methods, robot function etc.

from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import JointPositions, get_joint_infos, get_joint_positions
from predicators.pybullet_helpers.link import get_link_state

#Import the functions that are to be tested:

from predicators.pybullet_helpers.motion_planning import run_motion_planning
#The pick/place options to be tested are accessed via the env instance
from predicators.pybullet_helpers.controllers import create_move_end_effector_to_pose_option,\
                                                    create_change_fingers_option

import copy

import matplotlib
import PIL
from PIL import ImageDraw

try:
    import gymnasium as mujoco_kitchen_gym
    from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, \
        get_site_xmat, get_site_xpos
    from gymnasium_robotics.utils.rotations import mat2quat
    _MJKITCHEN_IMPORTED = True
except (ImportError, RuntimeError):
    _MJKITCHEN_IMPORTED = False
from predicators.envs import BaseEnv
from predicators.envs.kitchen import KitchenEnv

#Configure logging for better debugging outputs:
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.basicConfig(
    level=logging.WARNING,                    
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

#Defining test configuration, and overriding some default ones:
CFG.use_gui = True

CFG.seed = random.randint(0,10000)

CFG.kitchen_use_perfect_samplers = True
CFG.kitchen_goals = "knob_only"
#Num of PyBullet physics steps per high-level Action in visualize_action_sequence
CFG.pybullet_sim_steps_per_action = 20

# 提高分辨率
CFG.pybullet_camera_width = 1674  # 从335提高到1674
CFG.pybullet_camera_height = 900  # 从180提高到900

# 提高DPI
CFG.render_state_dpi = 300  # 从150提高到300

# 提高帧率
CFG.video_fps = 10  # 从2提高到10

CFG.make_test_videos = False
CFG.make_failure_videos = False

CFG.env = "kitchen"

# 设置较少的训练任务数量，避免重复执行
CFG.num_train_tasks = 1

# 使用最简单的方法：直接调用reset然后只渲染
print("正在创建Mujoco Kitchen环境...")

# 导入gymnasium
import gymnasium as mujoco_kitchen_gym

# 直接创建gym环境
gym_env = mujoco_kitchen_gym.make("FrankaKitchen-v1", 
                                  render_mode="human",
                                  ik_controller=True)

print("Mujoco Kitchen环境创建成功！")

# 定义获取物体信息的函数
def get_object_info(gym_env):
    """获取环境中所有物体的信息"""
    mujoco_model = gym_env.model
    mujoco_data = gym_env.data
    
    object_info = {
        'bodies': {},
        'joints': {},
        'sites': {},
        'geoms': {}
    }
    
    # 获取物体信息（简化版本）
    for i in range(mujoco_model.nbody):
        position = mujoco_data.xpos[i]
        quat = mujoco_data.xquat[i]
        object_info['bodies'][f'body_{i}'] = {
            'id': i,
            'position': position.copy(),
            'quaternion': quat.copy()
        }
    
    # 获取关节信息（简化版本）
    for i in range(mujoco_model.njnt):
        joint_pos = mujoco_data.qpos[i] if i < len(mujoco_data.qpos) else 0
        object_info['joints'][f'joint_{i}'] = {
            'id': i,
            'position': joint_pos
        }
    
    # 获取site信息（简化版本）
    for i in range(mujoco_model.nsite):
        site_pos = mujoco_data.site_xpos[i]
        object_info['sites'][f'site_{i}'] = {
            'id': i,
            'position': site_pos.copy()
        }
    
    return object_info

# 定义添加新物体的函数
def add_object_to_scene(gym_env, object_name, position, size=(0.05, 0.05, 0.05), color=(1, 0, 0, 1)):
    """向场景中添加一个新的物体"""
    print(f"尝试添加物体 {object_name} 到位置 {position}")
    
    try:
        # 方法1：使用Mujoco的运行时添加功能
        import mujoco
        
        # 获取模型和数据
        model = gym_env.model
        data = gym_env.data
        
        # 创建一个简单的盒子几何体
        # 注意：这需要修改模型结构，在运行时添加比较复杂
        print("注意：在运行时添加物体需要修改Mujoco模型结构")
        print("建议的方法：")
        print("1. 修改XML模型文件添加新物体")
        print("2. 使用Mujoco的mocap功能添加可移动物体")
        print("3. 使用Mujoco的site功能添加标记点")
        
        # 这里演示如何添加一个site（标记点）
        # 注意：这需要模型支持动态添加
        return False
        
    except Exception as e:
        print(f"添加物体时出错: {e}")
        return False

def add_mocap_object(gym_env, object_name, position, size=(0.05, 0.05, 0.05)):
    """使用Mocap功能添加可移动物体"""
    try:
        # 检查是否有可用的mocap body
        model = gym_env.model
        data = gym_env.data
        
        # 查找可用的mocap body
        mocap_bodies = []
        for i in range(model.nbody):
            if model.body_mocapid[i] >= 0:  # 这是一个mocap body
                mocap_bodies.append((i, f"body_{i}"))
        
        if mocap_bodies:
            print(f"找到 {len(mocap_bodies)} 个可用的mocap物体:")
            for body_id, body_name in mocap_bodies:
                print(f"  {body_id}: {body_name}")
            
            # 使用第一个可用的mocap body
            mocap_id = model.body_mocapid[mocap_bodies[0][0]]
            data.mocap_pos[mocap_id] = position
            print(f"设置mocap物体 {mocap_bodies[0][1]} 位置为 {position}")
            return True
        else:
            print("没有找到可用的mocap物体")
            return False
            
    except Exception as e:
        print(f"添加mocap物体时出错: {e}")
        return False

# 直接调用reset来满足gymnasium的要求
print("正在调用reset以满足渲染要求...")
obs, info = gym_env.reset()
print("Reset完成，现在可以渲染了")
print("注意：环境已重置但不会执行任务，只进行渲染")

# 获取环境信息
print("\n=== 环境信息 ===")
print(f"观察空间形状: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
print(f"动作空间: {gym_env.action_space}")

# 获取Mujoco模型信息
mujoco_model = gym_env.model
mujoco_data = gym_env.data

print(f"\n=== Mujoco模型信息 ===")
print(f"模型名称: {mujoco_model.names}")
print(f"物体数量: {mujoco_model.nbody}")
print(f"关节数量: {mujoco_model.njnt}")
print(f"几何体数量: {mujoco_model.ngeom}")

# 获取物体信息（使用更简单的方法）
print(f"\n=== 物体信息 ===")
print(f"总物体数量: {mujoco_model.nbody}")
print("前10个物体的位置:")
for i in range(min(10, mujoco_model.nbody)):
    position = mujoco_data.xpos[i]
    print(f"物体 {i}: 位置 [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")

# 获取关节信息
print(f"\n=== 关节信息 ===")
print(f"总关节数量: {mujoco_model.njnt}")
print("前10个关节的位置:")
for i in range(min(10, mujoco_model.njnt)):
    joint_pos = mujoco_data.qpos[i] if i < len(mujoco_data.qpos) else 0
    print(f"关节 {i}: 位置 {joint_pos:.3f}")

# 获取site信息（这些是跟踪的物体）
print(f"\n=== Site信息（跟踪的物体）===")
print(f"总Site数量: {mujoco_model.nsite}")
print("所有Site的位置:")
for i in range(mujoco_model.nsite):
    site_pos = mujoco_data.site_xpos[i]
    print(f"Site {i}: 位置 [{site_pos[0]:.3f}, {site_pos[1]:.3f}, {site_pos[2]:.3f}]")

# 尝试添加一个mocap物体（如果可用）
print(f"\n=== 尝试添加新物体 ===")
test_position = [0.0, 0.5, 1.6]  # 在厨房中央上方
add_mocap_object(gym_env, "test_object", test_position)

print("Mujoco环境已打开。按Ctrl+C退出。")
print("注意：环境已准备就绪，只进行渲染。")

try:
    step_count = 0
    print("开始渲染循环...")
    print("按 'i' 键显示物体信息，按 'a' 键添加物体，按 Ctrl+C 退出")
    
    while True:
        # 直接渲染gym环境，不执行任何动作
        gym_env.render()
        
        # 每100步打印一次状态
        if step_count % 100 == 0:
            print(f"环境渲染中... 步骤: {step_count}")
            
            # 获取并显示关键物体信息
            object_info = get_object_info(gym_env)
            
            # 显示一些关键物体的位置（使用索引而不是名称）
            print("关键物体位置:")
            # 显示前几个site的位置
            for i in range(min(5, len(object_info['sites']))):
                site_key = f'site_{i}'
                if site_key in object_info['sites']:
                    pos = object_info['sites'][site_key]['position']
                    print(f"  Site {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            # 显示前几个body的位置
            for i in range(min(5, len(object_info['bodies'])):
                body_key = f'body_{i}'
                if body_key in object_info['bodies']:
                    pos = object_info['bodies'][body_key]['position']
                    print(f"  Body {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        step_count += 1
        time.sleep(0.1)  # 控制渲染频率
        
        # 检查键盘输入（这是一个简化的实现）
        # 在实际应用中，你可能需要使用更复杂的输入处理
        # 这里只是演示如何获取物体信息
        
except KeyboardInterrupt:
    print("\n正在关闭环境...")
    print("感谢使用Mujoco Kitchen环境查看器！")
except Exception as e:
    print(f"运行时出错: {e}")
    print(f"错误类型: {type(e)}")
    print(f"环境类型: {type(gym_env)}")