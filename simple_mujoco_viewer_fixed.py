"""
修复版本的Mujoco Kitchen环境查看器
获取物体信息和添加新物体
"""

import sys
import numpy as np
import time
import logging
import random
import mujoco

from predicators.structs import Action, Array, GroundAtom, Object, State, Type, ParameterizedOption, EnvironmentTask,\
    Image, Predicate, State, Type, Video
from predicators import utils
from predicators.settings import CFG
from gymnasium.spaces import Box

# 导入环境
from predicators.envs.kitchen import KitchenEnv

# 配置日志
logging.basicConfig(
    level=logging.WARNING,                    
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 配置环境参数
CFG.use_gui = True
CFG.seed = random.randint(0, 10000)
CFG.kitchen_use_perfect_samplers = True
CFG.kitchen_goals = "knob_only"
CFG.pybullet_sim_steps_per_action = 20

# 提高分辨率
CFG.pybullet_camera_width = 1674
CFG.pybullet_camera_height = 900
CFG.render_state_dpi = 300
CFG.video_fps = 10

CFG.make_test_videos = False
CFG.make_failure_videos = False
CFG.env = "kitchen"
CFG.num_train_tasks = 1

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
    
    # 获取物体信息（包含名称）
    for i in range(mujoco_model.nbody):
        position = mujoco_data.xpos[i]
        quat = mujoco_data.xquat[i]
        
        # 尝试获取物体名称
        body_name = None
        try:
            # 方法1：使用mujoco的name2id反向查找
            for name_id in range(mujoco_model.names):
                if mujoco_model.name_bodyid[name_id] == i:
                    body_name = mujoco_model.names[name_id]
                    break
        except:
            pass
        
        # 如果没找到名称，使用默认名称
        if body_name is None or body_name == "":
            body_name = f"body_{i}"
        
        object_info['bodies'][body_name] = {
            'id': i,
            'name': body_name,
            'position': position.copy(),
            'quaternion': quat.copy()
        }
    
    # 获取关节信息（包含名称）
    for i in range(mujoco_model.njnt):
        joint_pos = mujoco_data.qpos[i] if i < len(mujoco_data.qpos) else 0
        
        # 尝试获取关节名称
        joint_name = None
        try:
            for name_id in range(mujoco_model.names):
                if mujoco_model.name_jntid[name_id] == i:
                    joint_name = mujoco_model.names[name_id]
                    break
        except:
            pass
        
        if joint_name is None or joint_name == "":
            joint_name = f"joint_{i}"
        
        object_info['joints'][joint_name] = {
            'id': i,
            'name': joint_name,
            'position': joint_pos
        }
    
    # 获取site信息（包含名称）
    for i in range(mujoco_model.nsite):
        site_pos = mujoco_data.site_xpos[i]
        
        # 尝试获取site名称
        site_name = None
        try:
            for name_id in range(mujoco_model.names):
                if mujoco_model.name_siteid[name_id] == i:
                    site_name = mujoco_model.names[name_id]
                    break
        except:
            pass
        
        if site_name is None or site_name == "":
            site_name = f"site_{i}"
        
        object_info['sites'][site_name] = {
            'id': i,
            'name': site_name,
            'position': site_pos.copy()
        }
    
    return object_info

def find_object_by_name(gym_env, target_name):
    """根据名称查找物体"""
    object_info = get_object_info(gym_env)
    
    # 在所有类型中查找
    for obj_type in ['bodies', 'joints', 'sites']:
        for obj_name, obj_info in object_info[obj_type].items():
            if target_name.lower() in obj_name.lower():
                print(f"找到匹配的{obj_type[:-1]}: {obj_name} (ID: {obj_info['id']})")
                if 'position' in obj_info:
                    pos = obj_info['position']
                    print(f"  位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                return obj_info
    
    print(f"未找到包含 '{target_name}' 的物体")
    return None

def list_all_object_names(gym_env):
    """列出所有物体的名称"""
    object_info = get_object_info(gym_env)
    
    print("\n=== 所有物体名称 ===")
    print("Bodies:")
    for body_name in object_info['bodies'].keys():
        print(f"  - {body_name}")
    
    print("\nJoints:")
    for joint_name in object_info['joints'].keys():
        print(f"  - {joint_name}")
    
    print("\nSites:")
    for site_name in object_info['sites'].keys():
        print(f"  - {site_name}")

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

def main():
    """主函数"""
    print("正在创建Mujoco Kitchen环境...")
    
    # 导入gymnasium
    import gymnasium as mujoco_kitchen_gym
    
    # 直接创建gym环境
    gym_env = mujoco_kitchen_gym.make("FrankaKitchen-v1", 
                                      render_mode="human",
                                      ik_controller=True)
    
    print("Mujoco Kitchen环境创建成功！")
    
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
    print(f"物体数量: {mujoco_model.nbody}")
    print(f"关节数量: {mujoco_model.njnt}")
    print(f"几何体数量: {mujoco_model.ngeom}")
    print(f"Site数量: {mujoco_model.nsite}")
    
    # 获取物体信息（包含名称）
    print(f"\n=== 物体信息 ===")
    print(f"总物体数量: {mujoco_model.nbody}")
    print("前10个物体的位置和名称:")
    object_info = get_object_info(gym_env)
    body_count = 0
    for body_name, body_info in object_info['bodies'].items():
        if body_count >= 10:
            break
        pos = body_info['position']
        print(f"物体 {body_info['id']}: {body_name} - 位置 [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        body_count += 1
    
    # 获取关节信息（包含名称）
    print(f"\n=== 关节信息 ===")
    print(f"总关节数量: {mujoco_model.njnt}")
    print("前10个关节的位置和名称:")
    joint_count = 0
    for joint_name, joint_info in object_info['joints'].items():
        if joint_count >= 10:
            break
        pos = joint_info['position']
        print(f"关节 {joint_info['id']}: {joint_name} - 位置 {pos:.3f}")
        joint_count += 1
    
    # 获取site信息（这些是跟踪的物体）
    print(f"\n=== Site信息（跟踪的物体）===")
    print(f"总Site数量: {mujoco_model.nsite}")
    print("所有Site的位置和名称:")
    for site_name, site_info in object_info['sites'].items():
        pos = site_info['position']
        print(f"Site {site_info['id']}: {site_name} - 位置 [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # 演示查找特定物体
    print(f"\n=== 查找特定物体 ===")
    # 查找包含"kettle"的物体
    find_object_by_name(gym_env, "kettle")
    # 查找包含"knob"的物体
    find_object_by_name(gym_env, "knob")
    # 查找包含"EEF"的物体
    find_object_by_name(gym_env, "EEF")
    
    # 列出所有物体名称
    list_all_object_names(gym_env)
    
    # 尝试添加一个mocap物体（如果可用）
    print(f"\n=== 尝试添加新物体 ===")
    test_position = [0.0, 0.5, 1.6]  # 在厨房中央上方
    add_mocap_object(gym_env, "test_object", test_position)
    
    print("Mujoco环境已打开。按Ctrl+C退出。")
    print("注意：环境已准备就绪，只进行渲染。")
    
    try:
        step_count = 0
        print("开始渲染循环...")
        print("按 Ctrl+C 退出")
        
        while True:
            # 直接渲染gym环境，不执行任何动作
            gym_env.render()
            
            # 每100步打印一次状态
            if step_count % 100 == 0:
                print(f"环境渲染中... 步骤: {step_count}")
                
                # 获取并显示关键物体信息
                object_info = get_object_info(gym_env)
                
                # 显示一些关键物体的位置和名称
                print("关键物体位置:")
                # 显示前几个site的位置
                site_count = 0
                for site_name, site_info in object_info['sites'].items():
                    if site_count >= 5:
                        break
                    pos = site_info['position']
                    print(f"  Site {site_info['id']}: {site_name} - [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    site_count += 1
                
                # 显示前几个body的位置
                body_count = 0
                for body_name, body_info in object_info['bodies'].items():
                    if body_count >= 5:
                        break
                    pos = body_info['position']
                    print(f"  Body {body_info['id']}: {body_name} - [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    body_count += 1
            
            step_count += 1
            time.sleep(0.1)  # 控制渲染频率
            
    except KeyboardInterrupt:
        print("\n正在关闭环境...")
        print("感谢使用Mujoco Kitchen环境查看器！")
    except Exception as e:
        print(f"运行时出错: {e}")
        print(f"错误类型: {type(e)}")
        print(f"环境类型: {type(gym_env)}")

if __name__ == "__main__":
    main()
