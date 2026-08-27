import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.eval import audit_historical_vg_run as audit


class AuditHistoricalVGRunTest(unittest.TestCase):
    def test_validate_checkpoint_accepts_explicit_unfrozen_fuser(self):
        state = {
            "position_net.gat_layers.0.weight": torch.zeros(1),
            "position_net.graph_gate": torch.zeros(1),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            torch.save(
                {
                    "iters": 1000,
                    "model_trainable": state,
                    "trainable_names": sorted(state),
                    "config_dict": {
                        "freeze_fuser": False,
                        "freeze_position_base": True,
                        "save_trainable_only": True,
                    },
                },
                checkpoint_path,
            )
            with mock.patch.object(
                audit,
                "validate_checkpoint_trainable_manifest",
                return_value=None,
            ), mock.patch.object(
                audit,
                "validate_grounding_state",
                return_value=(state, {"loaded": len(state)}),
            ):
                report = audit.validate_checkpoint(
                    checkpoint_path,
                    expected_iters=1000,
                    model_state=state,
                    expected_freeze_fuser=False,
                )

        self.assertEqual(report["iterations"], 1000)
        self.assertEqual(report["strictly_compatible_tensor_count"], 2)


if __name__ == "__main__":
    unittest.main()
