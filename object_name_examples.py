"""
物体名称获取和使用示例
展示如何获取和使用Mujoco环境中的物体名称
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
            for name_id in range(mujoco_model.names):
                if mujoco_model.name_bodyid[name_id] == i:
                    body_name = mujoco_model.names[name_id]
                    break
        except:
            pass
        
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

def main():
    """主函数 - 演示物体名称获取"""
    print("=== Mujoco Kitchen环境物体名称获取演示 ===")
    
    # 导入gymnasium
    import gymnasium as mujoco_kitchen_gym
    
    # 直接创建gym环境
    gym_env = mujoco_kitchen_gym.make("FrankaKitchen-v1", 
                                      render_mode="human",
                                      ik_controller=True)
    
    print("Mujoco Kitchen环境创建成功！")
    
    # 调用reset
    obs, info = gym_env.reset()
    print("环境重置完成")
    
    # 获取所有物体信息
    object_info = get_object_info(gym_env)
    
    print(f"\n=== 环境统计 ===")
    print(f"Bodies数量: {len(object_info['bodies'])}")
    print(f"Joints数量: {len(object_info['joints'])}")
    print(f"Sites数量: {len(object_info['sites'])}")
    
    # 列出所有物体名称
    list_all_object_names(gym_env)
    
    # 查找特定物体
    print(f"\n=== 查找特定物体 ===")
    
    # 查找水壶
    print("\n1. 查找水壶:")
    find_object_by_name(gym_env, "kettle")
    
    # 查找旋钮
    print("\n2. 查找旋钮:")
    find_object_by_name(gym_env, "knob")
    
    # 查找机器人末端执行器
    print("\n3. 查找机器人末端执行器:")
    find_object_by_name(gym_env, "EEF")
    
    # 查找门
    print("\n4. 查找门:")
    find_object_by_name(gym_env, "hinge")
    
    # 查找灯
    print("\n5. 查找灯:")
    find_object_by_name(gym_env, "light")
    
    # 查找微波炉
    print("\n6. 查找微波炉:")
    find_object_by_name(gym_env, "micro")
    
    # 查找滑动门
    print("\n7. 查找滑动门:")
    find_object_by_name(gym_env, "slide")
    
    print(f"\n=== 使用示例 ===")
    print("现在你可以使用以下功能:")
    print("1. find_object_by_name(gym_env, '物体名称') - 查找特定物体")
    print("2. list_all_object_names(gym_env) - 列出所有物体名称")
    print("3. get_object_info(gym_env) - 获取所有物体信息")
    
    print("\n按Ctrl+C退出...")
    
    try:
        while True:
            gym_env.render()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n演示结束！")

if __name__ == "__main__":
    main()





