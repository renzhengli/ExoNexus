import asyncio
import json
import psutil
import rustworkx as rx
import zenoh
from datetime import datetime, timezone, timedelta

# ==========================================
# 1. 核心算法：拓扑管理与大模型算力切分
# (对应 exo 的 placement.py 和 placement_utils.py)
# ==========================================
class V2XTopologyManager:
    def __init__(self):
        self.graph = rx.PyDiGraph()
        self.node_to_idx = {}
        self.idx_to_node = {}
        
    def add_node_to_mesh(self, node_id: str):
        """添加节点，并将其与当前网络中的所有节点双向互联（全连接 Mesh 组网）"""
        if node_id in self.node_to_idx:
            return
        idx = self.graph.add_node(node_id)
        
        # 与现有所有节点建立 P2P 物理连接边
        for existing_id, existing_idx in self.node_to_idx.items():
            self.graph.add_edge(idx, existing_idx, 1.0)
            self.graph.add_edge(existing_idx, idx, 1.0)
            
        self.node_to_idx[node_id] = idx
        self.idx_to_node[idx] = node_id

    def remove_node(self, node_id: str):
        """节点掉线时的拓扑自愈，瞬间自动剪枝所有相关的网络边"""
        if node_id in self.node_to_idx:
            idx = self.node_to_idx[node_id]
            self.graph.remove_node(idx)
            del self.node_to_idx[node_id]
            del self.idx_to_node[idx]

    def get_all_cycles(self) -> list[list[str]]:
        cycle_idxs = rx.simple_cycles(self.graph)
        cycles = [[self.idx_to_node[idx] for idx in c] for c in cycle_idxs]
        for node_id in self.node_to_idx.keys():
            cycles.append([node_id]) # 加入单节点情况
        return cycles

class ModelPartitioner:
    @staticmethod
    def allocate_layers_proportionally(total_layers: int, memory_fractions: list[float]) -> list[int]:
        n = len(memory_fractions)
        if n == 0 or total_layers < n: return []
        raw = [f * total_layers for f in memory_fractions]
        result = [int(r) for r in raw]
        by_remainder = sorted(range(n), key=lambda i: raw[i] - result[i], reverse=True)
        for i in range(total_layers - sum(result)):
            result[by_remainder[i]] += 1
        for i in range(n):
            if result[i] == 0:
                max_idx = max(range(n), key=lambda j: result[j])
                result[max_idx] -= 1
                result[i] = 1
        return result

    @staticmethod
    def find_best_pipeline(topology: V2XTopologyManager, node_memory_gb: dict[str, float], model_layers: int, model_required_gb: float):
        cycles = topology.get_all_cycles()
        valid_cycles = []
        for cycle in cycles:
            total_mem = sum(node_memory_gb.get(n, 0) for n in cycle)
            if total_mem >= model_required_gb:
                valid_cycles.append((cycle, total_mem))
                
        if not valid_cycles:
            return None # 算力不足

        # 选出节点最少（通信延迟最低）、总内存最大的最优环路
        min_nodes = min(len(c) for c, _ in valid_cycles)
        best_cycle, total_mem = max([x for x in valid_cycles if len(x[0]) == min_nodes], key=lambda x: x[1])

        fractions = [node_memory_gb.get(n, 0) / total_mem for n in best_cycle]
        allocations = ModelPartitioner.allocate_layers_proportionally(model_layers, fractions)

        plan = {}
        current_layer = 0
        for i, node in enumerate(best_cycle):
            next_node = best_cycle[(i + 1) % len(best_cycle)] if len(best_cycle) > 1 else node
            plan[node] = {
                "start": current_layer,
                "end": current_layer + allocations[i],
                "next": next_node
            }
            current_layer += allocations[i]
        return plan

# ==========================================
# 2. 车辆节点：通过 Zenoh P2P 广播自身状态
# ==========================================
class V2XNodeWorker:
    def __init__(self, node_id: str, ram_override: float):
        self.node_id = node_id
        self.ram_override = ram_override # 测试用，强行指定显存大小
        self.running = True
        
        # 初始化 Zenoh P2P Session (去中心化网络绑定)
        conf = zenoh.Config()
        self.z_session = zenoh.open(conf)

    async def run_heartbeat_loop(self):
        print(f"🚗 [车辆启动] {self.node_id} 连入 V2X Zenoh P2P 网络，算力: {self.ram_override}GB")
        
        # Zenoh 的广播主题 (Topic)
        topic = f"v2x/cluster/heartbeat/{self.node_id}"
        
        # 声明一个发布者
        publisher = self.z_session.declare_publisher(topic)
        
        while self.running:
            # 组装状态
            payload = {
                "node_id": self.node_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ram_free_gb": self.ram_override
            }
            # 通过 Zenoh P2P 发送数据
            publisher.put(json.dumps(payload).encode('utf-8'))
            await asyncio.sleep(1.0)
            
    def shutdown(self):
        self.running = False
        self.z_session.close()

# ==========================================
# 3. 调度集群：通过 Zenoh 订阅并动态管理拓扑
# ==========================================
class V2XClusterMaster:
    def __init__(self):
        self.topology = V2XTopologyManager()
        self.node_memory_gb = {}
        self.last_seen = {}
        self.timeout = timedelta(seconds=2.5) # 车辆 2.5 秒无心跳视为掉线
        
        # 初始化 Zenoh P2P Session
        conf = zenoh.Config()
        self.z_session = zenoh.open(conf)

        # 【新增这行】提前获取并保存主线程的事件循环
        self.loop = asyncio.get_running_loop() 
        
        # Asyncio 队列，用于将 Zenoh 底层线程的回调安全传给 Python 协程
        self.event_queue = asyncio.Queue()
        
        # 声明订阅者：监听所有车辆的心跳
        print("🌐 [Master] 正在监听 P2P 网络中的车辆心跳 (v2x/cluster/heartbeat/*)...")
        self.subscriber = self.z_session.declare_subscriber(
            "v2x/cluster/heartbeat/*", 
            self._on_zenoh_message
        )

    def _on_zenoh_message(self, sample: zenoh.Sample):
        """Zenoh C++ 底层收到数据时的回调函数"""
        payload_str = sample.payload.to_bytes().decode('utf-8')
        data = json.loads(payload_str)
        # 【修改这行】使用我们保存的主线程 loop，跨线程安全地投递数据
        self.loop.call_soon_threadsafe(self.event_queue.put_nowait, data)
        # 抛给 asyncio 循环处理
        # asyncio.get_event_loop().call_soon_threadsafe(self.event_queue.put_nowait, data)

    async def process_network_events(self):
        """处理新车辆加入和资源更新"""
        while True:
            data = await self.event_queue.get()
            node_id = data["node_id"]
            
            # 如果是新车，加入物理拓扑图
            if node_id not in self.node_memory_gb:
                print(f"   👋 [网络发现] 发现新加入车辆: {node_id}")
                self.topology.add_node_to_mesh(node_id)
                
            self.node_memory_gb[node_id] = data["ram_free_gb"]
            self.last_seen[node_id] = datetime.fromisoformat(data["timestamp"])

    async def watchdog_loop(self):
        """看门狗：负责超时剔除和触发模型重切分"""
        last_nodes_count = 0
        while True:
            now = datetime.now(timezone.utc)
            dead_nodes = [nid for nid, last_t in self.last_seen.items() if now - last_t > self.timeout]
            
            # 1. 剔除掉线节点
            for node_id in dead_nodes:
                print(f"\n🚨 [网络断开] 车辆 {node_id} P2P 心跳超时！从拓扑图中修剪。")
                self.topology.remove_node(node_id)
                del self.last_seen[node_id]
                del self.node_memory_gb[node_id]
            
            current_nodes_count = len(self.last_seen)
            
            # 2. 如果集群规模发生变化（有新车来，或者有车掉线），触发算力重组！
            if current_nodes_count != last_nodes_count and current_nodes_count > 0:
                self._rebuild_ai_pipeline()
                last_nodes_count = current_nodes_count
                
            await asyncio.sleep(1.0)

    def _rebuild_ai_pipeline(self):
        print("-" * 50)
        print(f"🔄 [拓扑重组] 当前在线节点: {list(self.node_memory_gb.keys())}")
        print("⚙️  开始重组大模型算力 (模型: 32层, 需 20GB显存)...")
        
        plan = ModelPartitioner.find_best_pipeline(
            self.topology, self.node_memory_gb, model_layers=32, model_required_gb=20.0
        )
        
        if plan is None:
            print("❌ [警告] 当前剩余算力不足以运行该大模型！请等待其他车辆加入。")
        else:
            print("✅ [调度成功] 新的大模型分布式推理路由表下发：")
            for node, details in plan.items():
                print(f"   ➔ {node}: 负责第 {details['start']}~{details['end']-1} 层 | 算完后张量转发至 --> {details['next']}")
        print("-" * 50)

# ==========================================
# 4. 模拟运行环境
# ==========================================
async def main():
    # 启动 Master
    master = V2XClusterMaster()
    asyncio.create_task(master.process_network_events())
    asyncio.create_task(master.watchdog_loop())
    
    # 车辆 A (8G 显存) 和 车辆 B (16G 显存) 加入网络
    car_a = V2XNodeWorker("Car_A", ram_override=8.0)
    car_b = V2XNodeWorker("Car_B", ram_override=16.0)
    
    t_a = asyncio.create_task(car_a.run_heartbeat_loop())
    t_b = asyncio.create_task(car_b.run_heartbeat_loop())
    
    # 稳定运行 4 秒 (Master 会自动发现他们并组建 2台车的 Pipeline)
    await asyncio.sleep(4)
    
    # 突发事件：Car B 驶出通信范围
    print("\n💥 [突发事件] Car_B 驶出 V2X 覆盖范围，掉线！")
    car_b.shutdown()
    
    # 再运行 3 秒，Master 的看门狗会捕捉到超时，并发现 Car A 一个人算力不够
    await asyncio.sleep(3)
    
    # 救兵来了：一辆算力极强的路侧 RSU 或高级智驾车加入了网络
    print("\n🚑 [救兵到达] RSU_Center 连入 V2X 网络！")
    rsu = V2XNodeWorker("RSU_Center", ram_override=32.0)
    t_rsu = asyncio.create_task(rsu.run_heartbeat_loop())
    
    # 观察 Master 如何把断掉的网络和 Car A、RSU_Center 重新组成新的推理流水线
    await asyncio.sleep(4)

    # 停止任务
    car_a.shutdown()
    rsu.shutdown()
    for t in [t_a, t_b, t_rsu]:
        t.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass