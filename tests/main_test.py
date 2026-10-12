# main_test.py
from v2x_scheduler import V2XTopologyManager, ModelPartitioner

# 1. 初始化路口车联网拓扑
topo = V2XTopologyManager()
topo.add_connection("CarA", "CarB")
topo.add_connection("CarB", "RSU_1") # 路侧单元
topo.add_connection("RSU_1", "CarA")

# 每台设备的实时可用显存 (GB)
node_memory = {
    "CarA": 8.0,
    "CarB": 12.0,
    "RSU_1": 24.0
}

print("========== 初始状态组网 ==========")
# 假设大模型需要 30GB 显存，总共 32 层
plan = ModelPartitioner.find_best_pipeline(topo, node_memory, model_layers=32, model_required_gb=30.0)
for node, details in plan.items():
    print(f"节点 {node}: 负责第 {details['start_layer']} 到 {details['end_layer']} 层 -> 算完发给 {details['send_output_to']}")


print("\n========== [突发] 车辆 CarB 驶离路口 (掉线) ==========")
topo.remove_node("CarB")
# 此时如果 CarB 的内存被拿掉，总内存可能不够了，为了演示，我们假定 RSU 释放了更多显存
node_memory["RSU_1"] = 32.0 

# 车A 和 RSU_1 之间刚好能双向通信组成新环路 (如果你的通信是全连通的，比如通过ZeroTier)
topo.add_connection("CarA", "RSU_1")
topo.add_connection("RSU_1", "CarA")

plan = ModelPartitioner.find_best_pipeline(topo, node_memory, model_layers=32, model_required_gb=30.0)
for node, details in plan.items():
    print(f"节点 {node}: 负责第 {details['start_layer']} 到 {details['end_layer']} 层 -> 算完发给 {details['send_output_to']}")