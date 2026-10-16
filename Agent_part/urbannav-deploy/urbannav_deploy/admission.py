"""Read-only resource admission before claiming isolated inference measurements."""
import csv
import io
import subprocess


def evaluate_gpu_ownership(gpu_csv, process_csv, visible_device):
    """Require exactly one compute process on the service's physical GPU.

    Args:
        gpu_csv: nvidia-smi rows: index, uuid, memory.used, utilization.gpu.
        process_csv: nvidia-smi rows: gpu_uuid, pid, used_memory.
        visible_device: Single numeric CUDA_VISIBLE_DEVICES from service health.
    Returns:
        Auditable snapshot with an admission decision and matching processes.
    Raises:
        ValueError: Device mapping cannot be established unambiguously.
    """
    if not str(visible_device).isdigit():
        raise ValueError('explicit physical GPU index required for admission')
    devices = [list(map(str.strip, row)) for row in csv.reader(io.StringIO(gpu_csv)) if row]
    matches = [row for row in devices if row[0] == str(visible_device)]
    if len(matches) != 1 or len(matches[0]) != 4:
        raise ValueError('cannot map service to one physical GPU')
    device = matches[0]
    processes = [list(map(str.strip, row)) for row in csv.reader(io.StringIO(process_csv))
                 if row and row[0].strip() == device[1]]
    return {'gpu_index': device[0], 'gpu_uuid': device[1], 'memory_used': device[2],
            'utilization': device[3], 'compute_processes': processes,
            'admitted': len(processes) == 1,
            'reason': 'single_compute_process' if len(processes) == 1 else 'shared_or_unidentified_gpu'}


def inspect_service_gpu(health):
    """Read GPU/process inventory once; never stop or reconfigure another process."""
    def query(fields):
        return subprocess.check_output(['nvidia-smi', fields, '--format=csv,noheader'],
                                       text=True, timeout=10)
    return evaluate_gpu_ownership(
        query('--query-gpu=index,uuid,memory.used,utilization.gpu'),
        query('--query-compute-apps=gpu_uuid,pid,used_memory'), health['cuda_visible_devices'])
