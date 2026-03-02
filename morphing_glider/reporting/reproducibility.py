import time
import sys
import platform
import hashlib
import json
from typing import Dict, Any

import numpy as np
import torch
import gymnasium as gym
import stable_baselines3 as sb3
import matplotlib as mpl

from morphing_glider.config import DEVICE, HYPERPARAMETER_REGISTRY


class ReproducibilityReport:
    """Collects and reports all information needed for reproducibility.

    Args:
        None.

    Returns:
        Dict via generate(), also saves JSON.

    References:
        [PINEAU_2021] Improving Reproducibility in ML Research.
    """

    @staticmethod
    def generate() -> Dict[str, Any]:
        """Generate reproducibility report.

        Returns:
            Dict with library versions, hardware info, hyperparameters, timestamp.
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "gymnasium_version": gym.__version__,
            "sb3_version": sb3.__version__,
        }
        try:
            import scipy; report["scipy_version"] = scipy.__version__
        except Exception: report["scipy_version"] = "N/A"
        try:
            report["matplotlib_version"] = mpl.__version__
        except Exception: pass
        try:
            if torch.cuda.is_available():
                report["gpu"] = torch.cuda.get_device_name(0)
            elif torch.backends.mps.is_available():
                report["gpu"] = "Apple Silicon (MPS)"
            else:
                report["gpu"] = "CPU only"
        except Exception:
            report["gpu"] = "unknown"
        report["device_used"] = str(DEVICE)
        report["hyperparameters"] = dict(HYPERPARAMETER_REGISTRY)
        # SHA-256 of this file
        try:
            with open(__file__, "rb") as f:
                report["source_sha256"] = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            report["source_sha256"] = "N/A (interactive)"
        return report

    @staticmethod
    def save_and_print(path: str = "reproducibility_report.json") -> Dict[str, Any]:
        report = ReproducibilityReport.generate()
        print("\n" + "="*80)
        print("REPRODUCIBILITY REPORT")
        print("="*80)
        for k, v in report.items():
            if k == "hyperparameters":
                print(f"  {k}: ({len(v)} entries)")
            else:
                print(f"  {k}: {v}")
        try:
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Saved: {path}")
        except Exception as e:
            print(f"Save failed: {e!r}")
        return report
