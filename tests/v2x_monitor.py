import asyncio
import psutil
from datetime import datetime, timezone, timedelta

# 尝试导入 NVIDIA GPU 监控库 (仿照 exo 的实现)
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU = True
except Exception:
    HAS_GPU = False

# ==========================================
# 1. 车辆节点端 (Worker / InfoGatherer)
# 对应 exo 源码: src/exo/utils/info_gatherer/info_gatherer.py
# ==========================================
class V2XNodeWorker:
    def __init__(self, node_id: str, network_channel: asyncio.Queue):
        self.node_id = node_id
        self.network_channel = network_channel
        self.running = True

    def _get_gpu_memory_free_gb(self) -> float:
        """获取 GPU 剩余显存 (GB)"""
        if not HAS_GPU:
            return 0.0
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return meminfo.free / (1024 ** 3)
        except Exception:
            return 0.0

    def _get_ram_memory_free_gb(self) -> float:
        """获取系统剩余内存 (GB)"""
        return psutil.virtual_memory().available / (1024 ** 3)

    async def run_heartbeat_loop(self, interval_sec: float = 1.0):
        """后台循环：不断收集自身资源，并向网络发送心跳包"""
        print(f"[车辆 {self.node_id}] 引擎启动，开始广播资源心跳...")
        while self.running:
            # 1. 收集硬件资源
            ram_free = self._get_ram_memory_free_gb()
            gpu_free = self._get_gpu_memory_free_gb()
            
            # 2. 组装心跳数据包 (类似 exo 的 NodeGatheredInfo)
            heartbeat_pkt = {
                "node_id": self.node_id,
                "timestamp": datetime.now(timezone.utc),
                "ram_free_gb": round(ram_free, 2),
                "gpu_free_gb": round(gpu_free, 2),
                "has_gpu": HAS_GPU
            }
            
            # 3. 发送到网络 (这里用 asyncio.Queue 模拟 P2P 物理网络)
            await self.network_channel.put(heartbeat_pkt)
            
            # 休眠等待下一次心跳
            await asyncio.sleep(interval_sec)

    def simulate_crash(self):
        """模拟车辆驶出信号区或发生宕机"""
        print(f"\n💥 [模拟突发事件] 车辆 {self.node_id} 开进隧道，信号中断！停止发送心跳。")
        self.running = False


# ==========================================
# 2. 调度中心端 (Master / ClusterMonitor)
# 对应 exo 源码: src/exo/master/main.py -> _plan()
# ==========================================
class V2XClusterMonitor:
    def __init__(self, timeout_sec: float = 3.0):
        # 记录每辆车的最新状态和最后心跳时间
        self.node_status: dict[str, dict] = {}
        self.last_seen: dict[str, datetime] = {}
        self.timeout = timedelta(seconds=timeout_sec)

    async def network_listener(self, network_channel: asyncio.Queue):
        """模拟监听网络中发来的所有心跳包"""
        while True:
            pkt = await network_channel.get()
            node_id = pkt["node_id"]
            
            # 记录最新状态与心跳时间
            self.node_status[node_id] = pkt
            self.last_seen[node_id] = pkt["timestamp"]

    async def watchdog_loop(self, check_interval_sec: float = 1.0):
        """后台看门狗：揪出那些超时没发心跳的设备"""
        print(f"[Master] 看门狗启动，超时剔除阈值设为 {self.timeout.total_seconds()} 秒...")
        while True:
            now = datetime.now(timezone.utc)
            dead_nodes = []

            # 遍历所有已知车辆，检查心跳是否超时
            for node_id, last_time in list(self.last_seen.items()):
                if now - last_time > self.timeout:
                    dead_nodes.append(node_id)

            # 剔除掉线车辆
            for node_id in dead_nodes:
                print(f"   🚨 [Master 警告] 车辆 {node_id} 失去联络超过 {self.timeout.total_seconds()} 秒！已从拓扑图中剔除！")
                # 清理状态
                del self.last_seen[node_id]
                del self.node_status[node_id]
                
                # ---> 在这里，你可以调用之前写的 ModelPartitioner 进行算力重组！<---

            # 打印当前健康的集群算力
            self._print_cluster_status()
            await asyncio.sleep(check_interval_sec)

    def _print_cluster_status(self):
        if not self.node_status:
            print("[Master 状态] 当前无可用车辆。")
            return
        status_strs = []
        for nid, info in self.node_status.items():
            gpu_str = f"GPU余:{info['gpu_free_gb']}G" if info['has_gpu'] else "无GPU"
            status_strs.append(f"[{nid} (RAM余:{info['ram_free_gb']}G, {gpu_str})]")
        print(f"[Master 状态] 当前在线算力节点: {' | '.join(status_strs)}")


# ==========================================
# 3. 本地模拟运行流程
# ==========================================
async def main():
    # 模拟物理网络总线
    network_channel = asyncio.Queue()

    # 初始化两辆车 (Worker) 和一个协调中心 (Master)
    car_a = V2XNodeWorker("Car_A_001", network_channel)
    car_b = V2XNodeWorker("Car_B_002", network_channel)
    
    # 车端对延迟要求高，我们把超时阈值设为极短的 3 秒 (exo 默认是 30 秒)
    master = V2XClusterMonitor(timeout_sec=3.0)

    # 启动异步任务
    tasks = [
        asyncio.create_task(master.network_listener(network_channel)),
        asyncio.create_task(master.watchdog_loop(check_interval_sec=1.0)),
        asyncio.create_task(car_a.run_heartbeat_loop(interval_sec=1.0)),
        asyncio.create_task(car_b.run_heartbeat_loop(interval_sec=1.5)),
    ]

    # 让集群平稳运行 4 秒，你能看到资源持续上报
    await asyncio.sleep(4)

    # 模拟 Car B 突然掉线 (开进隧道断网)
    car_b.simulate_crash()

    # 再观察 5 秒。看门狗会在第 3 秒左右准时发现 Car B 掉线并将其剔除
    await asyncio.sleep(5)

    # 退出所有任务
    for t in tasks:
        t.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 