import rustworkx as rx

class V2XTopologyManager:
    """管理车端 P2P 物理网络的拓扑图"""
    def __init__(self):
        self.graph = rx.PyDiGraph()
        self.node_to_idx = {}  # NodeID (如车牌/IP) 到 图索引的映射
        self.idx_to_node = {}
        
    def add_node(self, node_id: str):
        if node_id not in self.node_to_idx:
            idx = self.graph.add_node(node_id)
            self.node_to_idx[node_id] = idx
            self.idx_to_node[idx] = node_id

    def remove_node(self, node_id: str):
        """节点掉线时的拓扑自愈"""
        if node_id in self.node_to_idx:
            idx = self.node_to_idx[node_id]
            self.graph.remove_node(idx)
            del self.node_to_idx[node_id]
            del self.idx_to_node[idx]

    def add_connection(self, source: str, sink: str, weight: float = 1.0):
        """添加车辆间的网络连接 (weight 可以是延迟或距离)"""
        self.add_node(source)
        self.add_node(sink)
        src_idx = self.node_to_idx[source]
        sink_idx = self.node_to_idx[sink]
        # 避免重复添加边
        if not self.graph.has_edge(src_idx, sink_idx):
            self.graph.add_edge(src_idx, sink_idx, weight)

    def get_all_cycles(self) -> list[list[str]]:
        """寻找所有可能的闭环（流水线路径）"""
        cycle_idxs = rx.simple_cycles(self.graph)
        cycles = []
        for cycle_idx in cycle_idxs:
            cycles.append([self.idx_to_node[idx] for idx in cycle_idx])
        # 加入单节点自身（单车独立推理的情况）
        for node_id in self.node_to_idx.keys():
            cycles.append([node_id])
        return cycles

class ModelPartitioner:
    """大模型算力切分引擎"""
    
    @staticmethod
    def allocate_layers_proportionally(total_layers: int, memory_fractions: list[float]) -> list[int]:
        """
        核心算法：根据可用显存比例，为管道中的每个节点分配模型层数
        (完美还原 exo 的 placement_utils 逻辑)
        """
        n = len(memory_fractions)
        if n == 0 or total_layers < n:
            raise ValueError(f"无法将 {total_layers} 层分配给 {n} 个节点")

        # 1. 基础按比例分配（向下取整）
        raw = [f * total_layers for f in memory_fractions]
        result = [int(r) for r in raw]
        
        # 2. 处理余数：优先把剩余的层分给因取整而损失比例最大的节点
        by_remainder = sorted(range(n), key=lambda i: raw[i] - result[i], reverse=True)
        for i in range(total_layers - sum(result)):
            result[by_remainder[i]] += 1

        # 3. 兜底保障：确保每个节点至少分到 1 层
        for i in range(n):
            if result[i] == 0:
                max_idx = max(range(n), key=lambda j: result[j])
                result[max_idx] -= 1
                result[i] = 1

        return result

    @staticmethod
    def find_best_pipeline(
        topology: V2XTopologyManager, 
        node_memory_gb: dict[str, float], 
        model_layers: int, 
        model_required_gb: float
    ) -> dict[str, dict]:
        """寻找最优计算环路，并生成切分方案"""
        cycles = topology.get_all_cycles()
        
        valid_cycles = []
        for cycle in cycles:
            # 检查这个环路的总显存是否大于模型需求
            total_mem = sum(node_memory_gb.get(n, 0) for n in cycle)
            if total_mem >= model_required_gb:
                valid_cycles.append((cycle, total_mem))
                
        if not valid_cycles:
            raise ValueError("当前车联网内可用算力不足，无法运行该模型！")

        # 选出节点最少、总内存最大的最优环路
        min_nodes = min(len(c) for c, _ in valid_cycles)
        best_cycle_info = max(
            [c for c in valid_cycles if len(c[0]) == min_nodes],
            key=lambda x: x[1] # 比较 total_mem
        )
        best_cycle = best_cycle_info[0]
        total_mem = best_cycle_info[1]

        # 计算比例并切分层数
        fractions = [node_memory_gb.get(n, 0) / total_mem for n in best_cycle]
        allocations = ModelPartitioner.allocate_layers_proportionally(model_layers, fractions)

        # 生成具体下发给车机的路由表与层数分配
        routing_plan = {}
        current_layer = 0
        for i, node in enumerate(best_cycle):
            next_node = best_cycle[(i + 1) % len(best_cycle)] if len(best_cycle) > 1 else node
            layers_for_this_node = allocations[i]
            
            routing_plan[node] = {
                "rank": i,
                "start_layer": current_layer,
                "end_layer": current_layer + layers_for_this_node,
                "send_output_to": next_node  # 告诉这台车算完后把张量发给谁
            }
            current_layer += layers_for_this_node
            
        return routing_plan