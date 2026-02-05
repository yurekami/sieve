import sys
import subprocess
import time
import requests
import json
import os
from typing import Optional, List


def start_vllm_server(
    model_to_serve_name: str,
    served_model_name: str,
    max_model_len: int = 8192,
    tensor_parallel_size: int = 2,
    data_parallel_size: int = 1,
    max_logprobs: int = 20,
    port: int = 8000,
    cuda_devices: Optional[List[int]] = None,
) -> subprocess.Popen:
    """
    Start a vLLM server for the specified model.

    Args:
        model_to_serve_name: Name/path of the model to serve
        served_model_name: Name to serve the model as
        max_model_len: Maximum model length
        tensor_parallel_size: Number of GPUs for tensor parallelism
        data_parallel_size: Number of data parallel replicas (vLLM native DP)
        max_logprobs: Maximum number of logprobs to return
        port: Port number for the server
        cuda_devices: List of GPU indices to use (e.g., [0,1,2,3] or [4,5,6,7])

    Returns:
        Process object for the server
    """
    # Set up environment with CUDA_VISIBLE_DEVICES if specified
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in cuda_devices)
        print(f"Starting vLLM server on GPUs: {cuda_devices}")

    # Build command arguments
    cmd_args = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_to_serve_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--data-parallel-size",
        str(data_parallel_size),
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        str(max_model_len),
        "--trust-remote-code",
        "--max-logprobs",
        str(max_logprobs),
    ]

    # fmt: off
    vllm_server_process = subprocess.Popen(
        cmd_args,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )
    # fmt: on

    # Wait for the vLLM server to be ready
    print(f"Waiting for vLLM server on port {port} to initialize...")
    server_ready = False
    max_retries = 30
    retry_interval = 10

    for attempt in range(max_retries):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/v1/models")
            if response.status_code == 200:
                models = response.json()
                print(
                    f"vLLM server on port {port} is ready! Available models: {json.dumps(models, indent=2)}"
                )
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass

        print(
            f"Waiting for vLLM server on port {port} to initialize (attempt {attempt + 1}/{max_retries})..."
        )
        time.sleep(retry_interval)

    if not server_ready:
        print(
            f"Error: vLLM server on port {port} did not start successfully after maximum retries"
        )
        vllm_server_process.terminate()
        sys.exit(1)

    return vllm_server_process
