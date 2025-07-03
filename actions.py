import h5py
import numpy as np

# 加载一个 demo 文件
file = h5py.File(r"C:\Users\liuxi\robocasa\datasets\v0.1\single_stage\kitchen_navigate\NavigateKitchen\2024-05-09\demo_gentex_im128_randcams.hdf5")

# 假设你取第一个 demo
demo_name = list(file["data"].keys())[0]
demo = file["data"][demo_name]

# 提取 actions
actions = demo["actions"][:]

# 查看基本信息
print("Actions shape:", actions.shape)  # 比如 (T, 12)
print("First action vector:", actions[0])

# 获取动作和末端执行器的位置
eef_pos = demo["obs"]["robot0_eef_pos"][:]  # (T, 3)

# 计算位置差（动作估计值）
eef_delta = eef_pos[1:] - eef_pos[:-1]       # (T-1, 3)
action_trimmed = actions[:-1, :3]            # (T-1, 3)

# 计算误差
error = np.abs(eef_delta - action_trimmed)

# 显示最大、平均误差
print("Max error (per axis):", error.max(axis=0))
print("Mean error (per axis):", error.mean(axis=0))

# 判断是否接近（允许小数误差）
if np.allclose(eef_delta, action_trimmed, atol=1e-4):
    print("✅ 动作前三维与末端位置差一致")
else:
    print("❌ 动作前三维与位置差不一致")