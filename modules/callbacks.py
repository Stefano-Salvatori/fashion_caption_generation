# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GPU Stats Monitor
=================

Monitor and logs GPU stats during training.

"""

import os
import shutil
import subprocess
from typing import Dict, List, Tuple

import torch
from transformers import TrainerControl
from transformers.trainer_callback import TrainerCallback, TrainingArguments, TrainerState


class GPUStatsMonitor(TrainerCallback):
    """
    Automatically monitors and logs GPU stats during training stage. ``GPUStatsMonitor``
    is a callback and in order to use it you need to assign a logger in the ``Trainer``

    Raises:
        ValueError:
            If NVIDIA driver is not installed, not running on GPUs, or ``Trainer`` has no logger.

    GPU stats are mainly based on `nvidia-smi --query-gpu` command. The description of the queries is as follows:

    - **fan.speed** - The fan speed value is the percent of maximum speed that the device's fan is currently
      intended to run at. It ranges from 0 to 100 %. Note: The reported speed is the intended fan speed.
      If the fan is physically blocked and unable to spin, this output will not match the actual fan speed.
      Many parts do not report fan speeds because they rely on cooling via fans in the surrounding enclosure.
    - **memory.used** - Total memory allocated by active contexts.
    - **memory.free** - Total free memory.
    - **utilization.gpu** - Percent of time over the past sample period during which one or more kernels was
      executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product.
    - **utilization.memory** - Percent of time over the past sample period during which global (device) memory was
      being read or written. The sample period may be between 1 second and 1/6 second depending on the product.
    - **temperature.gpu** - Core GPU temperature, in degrees C.
    - **temperature.memory** - HBM memory temperature, in degrees C.

    """

    def __init__(self):
        super().__init__()

        if shutil.which("nvidia-smi") is None:
            raise ValueError("Cannot use GPUStatsMonitor callback because NVIDIA driver is not installed.")

        self._gpu_ids = self._get_gpu_ids()

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict = None, **kwargs
    ):
        if state.is_local_process_zero:
            gpu_stat_keys = self._get_gpu_stat_keys() + self._get_gpu_device_stat_keys()
            gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
            gpu_logs = self._parse_gpu_stats(self._gpu_ids, gpu_stats, gpu_stat_keys)
            logs.update(gpu_logs)

    @staticmethod
    def _get_gpu_ids() -> List[str]:
        """Get the unmasked real GPU IDs."""
        # All devices if `CUDA_VISIBLE_DEVICES` unset
        default = ",".join(str(i) for i in range(torch.cuda.device_count()))
        cuda_visible_devices: List[str] = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
        return [c.strip() for c in cuda_visible_devices]

    def _get_gpu_stats(self, queries: List[str]) -> List[List[float]]:
        if not queries:
            return []

        """Run nvidia-smi to get the gpu stats"""
        gpu_query = ",".join(queries)
        format = "csv,nounits,noheader"
        gpu_ids = ",".join(self._gpu_ids)
        result = subprocess.run(
            [
                # it's ok to supress the warning here since we ensure nvidia-smi exists during init
                shutil.which("nvidia-smi"),  # type: ignore
                f"--query-gpu={gpu_query}",
                f"--format={format}",
                f"--id={gpu_ids}",
            ],
            encoding="utf-8",
            capture_output=True,
            check=True,
        )

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except ValueError:
                return 0.0

        stats = [[_to_float(x) for x in s.split(", ")] for s in result.stdout.strip().split(os.linesep)]
        return stats

    @staticmethod
    def _parse_gpu_stats(
        device_ids: List[int], stats: List[List[float]], keys: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Parse the gpu stats into a loggable dict."""
        logs = {}
        for i, device_id in enumerate(device_ids):
            for j, (x, unit) in enumerate(keys):
                logs[f"device_id: {device_id}/{x} ({unit})"] = stats[i][j]
        return logs

    def _get_gpu_stat_keys(self) -> List[Tuple[str, str]]:
        """Get the GPU stats keys."""
        return [("memory.used", "MB"), ("memory.free", "MB"), ("utilization.memory", "%"), ("utilization.gpu", "%")]

    def _get_gpu_device_stat_keys(self) -> List[Tuple[str, str]]:
        """Get the device stats keys."""
        return [("fan.speed", "%"), ("temperature.gpu", "°C"), ("temperature.memory", "°C")]
