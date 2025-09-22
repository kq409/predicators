# Mujoco Kitchen环境物体管理指南

## 概述

这个指南展示了如何在Mujoco Kitchen环境中获取物体信息和添加新物体。

## 获取物体信息

### 1. 基本物体信息
```python
def get_object_info(gym_env):
    """获取环境中所有物体的信息"""
    mujoco_model = gym_env.model
    mujoco_data = gym_env.data
    
    object_info = {
        'bodies': {},      # 物体位置和姿态
        'joints': {},      # 关节位置
        'sites': {},       # 跟踪点位置
        'geoms': {}        # 几何体信息
    }
    # ... 实现细节
```

### 2. 可获取的信息类型

#### Bodies（物体）
- **位置**: `mujoco_data.xpos[body_id]`
- **姿态**: `mujoco_data.xquat[body_id]`
- **ID**: `mujoco_model.name2id(body_name, 'body')`

#### Joints（关节）
- **位置**: `mujoco_data.qpos[joint_id]`
- **速度**: `mujoco_data.qvel[joint_id]`

#### Sites（跟踪点）
- **位置**: `mujoco_data.site_xpos[site_id]`
- **旋转矩阵**: `mujoco_data.site_xmat[site_id]`

### 3. 关键物体列表
- `kettle`: 水壶
- `EEF`: 机器人末端执行器
- `knob1-4`: 旋钮1-4
- `hinge1-2`: 铰链门1-2
- `microhandle`: 微波炉把手
- `light`: 灯开关
- `slide`: 滑动门

## 添加新物体

### 1. 方法概述

#### 方法1: 修改XML模型文件（推荐）
```xml
<!-- 在模型文件中添加新物体 -->
<body name="new_object" pos="0 0 1">
    <geom name="new_object_geom" type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
</body>
```

#### 方法2: 使用Mocap功能
```python
def add_mocap_object(gym_env, object_name, position):
    """使用Mocap功能添加可移动物体"""
    model = gym_env.model
    data = gym_env.data
    
    # 查找可用的mocap body
    mocap_bodies = []
    for i in range(model.nbody):
        if model.body_mocapid[i] >= 0:
            body_name = model.id2name(i, 'body')
            mocap_bodies.append((i, body_name))
    
    if mocap_bodies:
        mocap_id = model.body_mocapid[mocap_bodies[0][0]]
        data.mocap_pos[mocap_id] = position
        return True
    return False
```

#### 方法3: 使用Site功能
```python
# 添加一个跟踪点
def add_site(gym_env, site_name, position):
    """添加一个site跟踪点"""
    # 注意：这需要模型支持动态添加
    pass
```

### 2. 运行时添加物体的限制

- **Mujoco模型结构**: 物体必须在XML模型中预定义
- **动态添加**: 只能在有限的情况下动态添加（如mocap物体）
- **性能考虑**: 频繁修改模型结构会影响性能

### 3. 推荐做法

1. **预定义物体**: 在XML模型中预定义所有需要的物体
2. **使用Mocap**: 对于可移动物体，使用mocap功能
3. **使用Sites**: 对于标记点，使用site功能
4. **模型修改**: 需要新物体时，修改XML文件并重新加载

## 使用示例

### 获取水壶位置
```python
object_info = get_object_info(gym_env)
if 'kettle' in object_info['sites']:
    kettle_pos = object_info['sites']['kettle']['position']
    print(f"水壶位置: {kettle_pos}")
```

### 获取机器人末端执行器位置
```python
if 'EEF' in object_info['sites']:
    eef_pos = object_info['sites']['EEF']['position']
    print(f"机器人末端位置: {eef_pos}")
```

### 添加一个测试物体
```python
test_position = [0.0, 0.5, 1.6]  # 在厨房中央上方
success = add_mocap_object(gym_env, "test_object", test_position)
if success:
    print("成功添加测试物体")
```

## 注意事项

1. **坐标系**: Mujoco使用右手坐标系，Z轴向上
2. **单位**: 位置单位通常是米
3. **角度**: 关节角度单位通常是弧度
4. **性能**: 频繁查询物体信息可能影响性能
5. **线程安全**: 在多线程环境中需要小心处理

## 扩展功能

### 1. 物体碰撞检测
```python
def check_collision(gym_env, body1_name, body2_name):
    """检查两个物体是否碰撞"""
    # 使用mujoco的碰撞检测功能
    pass
```

### 2. 物体距离计算
```python
def calculate_distance(pos1, pos2):
    """计算两个位置之间的距离"""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))
```

### 3. 物体可见性检测
```python
def is_visible(gym_env, body_name, camera_name):
    """检查物体是否在相机视野内"""
    # 使用mujoco的相机功能
    pass
```

这个指南提供了在Mujoco Kitchen环境中管理物体的基本方法。根据具体需求，可以进一步扩展这些功能。




