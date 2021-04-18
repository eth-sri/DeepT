import subprocess
import os


def get_gpu_memory_usage() -> int:
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory_usage = [int(x) for x in result.strip().split('\n')]
    current_gpu = int(os.getenv('CUDA_VISIBLE_DEVICES', 0))
    gpu_memory = gpu_memory_usage[current_gpu]
    return gpu_memory
