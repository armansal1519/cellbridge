import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import cellbridge_example


def test_seeded_example_matches_expected_output():
    expected = json.loads((ROOT / "examples" / "expected_output.json").read_text())
    actual = cellbridge_example.run()
    for key in ("predict", "contributions", "low", "high"):
        assert actual[key] == expected[key]
