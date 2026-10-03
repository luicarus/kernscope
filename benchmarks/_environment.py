import csv
import subprocess


GPU_FIELDS = ("gpu", "driver", "pstate", "sm_clock_mhz", "memory_clock_mhz", "power_w", "temperature_c")


def gpu_state():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ["unknown"] * len(GPU_FIELDS)
    if result.returncode:
        return ["unknown"] * len(GPU_FIELDS)
    return [value.strip() for value in next(csv.reader(result.stdout.splitlines()))]
