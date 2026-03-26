"""Capture model card and server metrics from an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import requests

from pawbench.types import ModelCard


def capture_model_card(endpoint: str) -> ModelCard:
    """Capture model + serving config for reproducible comparisons."""
    card = ModelCard()

    try:
        data = requests.get(f"{endpoint}/v1/models", timeout=5).json()
        card.model_name = data["data"][0]["id"]
    except Exception:
        pass

    # vLLM serve args from process list
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True, timeout=5)
        for line in out.split("\n"):
            if "vllm serve" in line or "vllm.entrypoints" in line:
                card.serving["cmdline"] = line.strip()
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.startswith("--") and i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                        card.serving[p.lstrip("-").replace("-", "_")] = parts[i + 1]
                    elif p.startswith("--") and "=" in p:
                        k, v = p.lstrip("-").split("=", 1)
                        card.serving[k.replace("-", "_")] = v
                break
    except Exception:
        pass

    # Model config.json
    model_path = card.serving.get("model", "")
    if not model_path and "cmdline" in card.serving:
        parts = card.serving["cmdline"].split("serve ")
        if len(parts) > 1:
            model_path = parts[1].split(" ")[0]
    if model_path:
        try:
            import pathlib

            config_path = pathlib.Path(model_path) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    mc = json.load(f)
                card.model_config = {
                    "architectures": mc.get("architectures", []),
                    "model_type": mc.get("model_type", ""),
                    "hidden_size": mc.get("hidden_size", 0),
                    "num_hidden_layers": mc.get("num_hidden_layers", 0),
                    "num_attention_heads": mc.get("num_attention_heads", 0),
                    "num_key_value_heads": mc.get("num_key_value_heads", 0),
                    "head_dim": mc.get("head_dim", 0),
                    "num_experts": mc.get("num_experts", mc.get("num_local_experts", 0)),
                    "num_experts_per_tok": mc.get("num_experts_per_tok", mc.get("num_selected_experts", 0)),
                    "vocab_size": mc.get("vocab_size", 0),
                    "quantization_config": mc.get("quantization_config", {}),
                }
        except Exception:
            pass

    card.tuning = {
        "gpu_memory_utilization": card.serving.get("gpu_memory_utilization", "?"),
        "max_model_len": card.serving.get("max_model_len", "?"),
        "max_num_seqs": card.serving.get("max_num_seqs", "?"),
        "max_num_batched_tokens": card.serving.get("max_num_batched_tokens", "?"),
        "kv_cache_dtype": card.serving.get("kv_cache_dtype", "?"),
        "speculative_config": card.serving.get("speculative_config", "none"),
    }

    # GPU info
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        card.gpu = {"name": parts[0], "driver": parts[1], "temp_c": parts[2], "power_w": parts[3]}
    except Exception:
        pass

    # Memory
    try:
        out = subprocess.check_output(["free", "-g"], text=True, timeout=5)
        for line in out.split("\n"):
            if line.startswith("Mem:"):
                parts = line.split()
                card.memory = {"total_gb": int(parts[1]), "used_gb": int(parts[2])}
    except Exception:
        pass

    return card


def scrape_server_metrics(endpoint: str) -> dict[str, Any]:
    """Scrape vLLM prometheus metrics relevant to benchmarking."""
    metrics: dict[str, Any] = {}
    try:
        text = requests.get(f"{endpoint}/metrics", timeout=5).text
        for line in text.split("\n"):
            if line.startswith("#"):
                continue
            for key in [
                "vllm:gpu_cache_usage_perc",
                "vllm:gpu_prefix_cache_hit_rate",
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
                "vllm:spec_decode_num_accepted_tokens_total",
                "vllm:spec_decode_num_draft_tokens_total",
                "vllm:avg_generation_throughput_toks_per_s",
            ]:
                if line.startswith(key):
                    try:
                        val = float(line.split()[-1])
                        clean_key = key.replace("vllm:", "")
                        metrics[clean_key] = val
                    except (ValueError, IndexError):
                        pass
    except Exception:
        pass

    # Compute acceptance rate if speculative metrics exist
    accepted = metrics.get("spec_decode_num_accepted_tokens_total", 0)
    drafted = metrics.get("spec_decode_num_draft_tokens_total", 0)
    if drafted > 0:
        metrics["spec_acceptance_rate"] = accepted / drafted

    return metrics
