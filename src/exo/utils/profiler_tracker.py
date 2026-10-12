import atexit
import contextlib
import json
import os
import sys
import tempfile
import time
from typing import Any, ClassVar, Self

_DEFAULT_REPORT_FILE = "exo_inference_profile_report.json"


class ExoProfileTracker:
    _instance: ClassVar[Self | None] = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_tracker()
        return cls._instance

    def _init_tracker(self) -> None:
        self.output_filepath: str = os.getenv("EXO_REPORT_FILE", _DEFAULT_REPORT_FILE)
        self.device_rank: int = int(os.getenv("EXO_DEVICE_RANK", "0"))
        self.model_id: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"

        self.is_warming_up: bool = False
        self._exported: bool = False

        # 统计数据结构
        self.prompt_text: str = ""
        self.prompt_tokens: int = 0
        self.output_text: str = ""
        self.generated_tokens: int = 0

        # 预填充阶段
        self.prefill_duration_ms: float = 0.0

        # 逐 Token 耗时统计 (ms)
        self.token_step_latencies: list[float] = []
        self.forward_comm_times: list[float] = []  # 发送/接收 Hidden State 耗时
        self.backward_comm_times: list[float] = []  # Ring/Token 同步通信耗时
        self.compute_times: list[float] = []  # 本地 Transformer 计算耗时
        self.sampling_times: list[float] = []  # LM Head / 采样耗时

        self.start_generation_time: float = 0.0
        self.end_generation_time: float = 0.0

        # 注册退出钩子，保证进程正常退出或被中断时保存
        atexit.register(self.export_report_to_json)

    def set_device_and_model(self, rank: int, model_id: str) -> None:
        self.device_rank = rank
        self.model_id = model_id

    def set_warmup_mode(self, is_warmup: bool) -> None:
        self.is_warming_up = is_warmup

    def reset(self) -> None:
        self.prompt_text = ""
        self.prompt_tokens = 0
        self.output_text = ""
        self.generated_tokens = 0
        self.prefill_duration_ms = 0.0
        self.token_step_latencies.clear()
        self.forward_comm_times.clear()
        self.backward_comm_times.clear()
        self.compute_times.clear()
        self.sampling_times.clear()
        self.start_generation_time = 0.0
        self.end_generation_time = 0.0
        self._exported = False

    def set_prompt(self, prompt: str, token_count: int, is_warmup: bool = False) -> None:
        if self.is_warming_up or is_warmup or "Prompt to warm up" in prompt:
            return

        self.reset()
        self.prompt_text = prompt
        self.prompt_tokens = token_count
        self.start_generation_time = time.perf_counter()

    def record_prefill(self, duration_ms: float) -> None:
        if self.is_warming_up:
            return
        self.prefill_duration_ms = duration_ms

    def record_step(
        self,
        compute_ms: float,
        comm_ms: float,
        step_total_ms: float,
        sampling_ms: float = 0.0,
        ring_comm_ms: float = 0.0,
    ) -> None:
        if self.is_warming_up:
            return

        self.generated_tokens += 1
        self.compute_times.append(compute_ms)
        self.forward_comm_times.append(comm_ms)
        self.sampling_times.append(sampling_ms)
        self.backward_comm_times.append(ring_comm_ms)
        self.token_step_latencies.append(step_total_ms)
        self.end_generation_time = time.perf_counter()

        tok_speed = 1000.0 / step_total_ms if step_total_ms > 0 else 0.0
        with contextlib.suppress(Exception):
            print(
                f"[EXO-Profile] Token #{self.generated_tokens:04d} | "
                f"本步总耗时: {step_total_ms:.2f}ms ({tok_speed:.1f} tok/s) | "
                f"计算耗时: {compute_ms:.2f}ms | 通信耗时: {comm_ms:.2f}ms",
                flush=True,
            )

    def append_output_chunk(self, text: str) -> None:
        if self.is_warming_up:
            return
        self.output_text += text

    def export_report_to_json(self) -> None:
        # 如果当前进程没有任何推理数据，严禁写出空报告或覆盖有效报告
        if self.generated_tokens == 0 and not self.prompt_text:
            return

        if self._exported:
            return
        self._exported = True

        total_time_s = (
            (self.end_generation_time - self.start_generation_time)
            if self.end_generation_time > self.start_generation_time
            else 0.0
        )
        avg_token_latency_ms = (
            (sum(self.token_step_latencies) / len(self.token_step_latencies))
            if self.token_step_latencies
            else 0.0
        )
        overall_tps = (
            (self.generated_tokens / total_time_s)
            if total_time_s > 0
            else (1000.0 / avg_token_latency_ms if avg_token_latency_ms > 0 else 0.0)
        )

        avg_compute = sum(self.compute_times) / max(1, len(self.compute_times))
        avg_fwd_comm = sum(self.forward_comm_times) / max(
            1, len(self.forward_comm_times)
        )
        avg_bwd_comm = sum(self.backward_comm_times) / max(
            1, len(self.backward_comm_times)
        )
        avg_sampling = sum(self.sampling_times) / max(1, len(self.sampling_times))
        identified_ms = avg_compute + avg_fwd_comm + avg_bwd_comm + avg_sampling
        bubble_overhead = max(0.0, avg_token_latency_ms - identified_ms)

        # 统计分位数
        sorted_latencies = sorted(self.token_step_latencies)
        min_lat = sorted_latencies[0] if sorted_latencies else 0.0
        max_lat = sorted_latencies[-1] if sorted_latencies else 0.0
        p50_lat = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_lat = sorted_latencies[p95_index] if sorted_latencies else 0.0

        report: dict[str, Any] = {
            "report_name": "EXO Distributed LLM Inference Profiling Report",
            "model_id": self.model_id,
            "device_rank": self.device_rank,
            "prompt_statistics": {
                "prompt_content": self.prompt_text,
                "input_tokens_count": self.prompt_tokens,
                "prefill_latency_ms": round(self.prefill_duration_ms, 2),
            },
            "generation_statistics": {
                "output_content": self.output_text,
                "generated_tokens_count": self.generated_tokens,
                "total_generation_time_seconds": round(total_time_s, 3),
                "average_latency_per_token_ms": round(avg_token_latency_ms, 2),
                "tokens_per_second": round(overall_tps, 2),
            },
            "detailed_latencies_breakdown_ms": {
                "avg_per_token_compute_ms": round(avg_compute, 2),
                "avg_per_token_forward_comm_ms": round(avg_fwd_comm, 2),
                "avg_per_token_backward_comm_ms": round(avg_bwd_comm, 2),
                "avg_per_token_sampling_ms": round(avg_sampling, 2),
                "pipeline_bubble_and_overhead_ms": round(bubble_overhead, 2),
            },
            "latency_percentiles_ms": {
                "min_ms": round(min_lat, 2),
                "max_ms": round(max_lat, 2),
                "p50_ms": round(p50_lat, 2),
                "p95_ms": round(p95_lat, 2),
            },
        }

        try:
            target_path = os.path.abspath(self.output_filepath)
            target_dir = os.path.dirname(target_path) or "."
            os.makedirs(target_dir, exist_ok=True)

            # 原子性写入：先写入临时文件，再原子替换，防止中断导致空文件或损坏
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target_dir,
                delete=False,
            ) as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                temp_file_name = f.name
            os.replace(temp_file_name, target_path)

            with contextlib.suppress(Exception):
                print(
                    f"\n[EXO-Profile] 已成功将推理性能报告导出至: {target_path}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as e:
            with contextlib.suppress(Exception):
                print(f"\n[EXO-Profile] 导出报告失败: {e}", file=sys.stderr, flush=True)


profiler = ExoProfileTracker()
