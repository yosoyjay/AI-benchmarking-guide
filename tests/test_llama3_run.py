"""Tests for LLAMA3 pretraining benchmark helpers."""

from benchmarks.nvidia.llama3_run import compute_steady_state, parse_llama3_output

# Realistic NeMo training log lines
SAMPLE_LOG = """\
[INFO] Initializing model...
[INFO] global_step: 1 epoch: 0 reduced_train_loss: 11.234 lr: 0.0001 train_step_timing in s: 5.432
[INFO] global_step: 2 epoch: 0 reduced_train_loss: 10.567 lr: 0.0001 train_step_timing in s: 4.321
[INFO] Some unrelated log line
[INFO] global_step: 3 epoch: 0 reduced_train_loss: 9.876 lr: 0.0001 train_step_timing in s: 4.111
"""


class TestParseLlama3Output:
    def test_extracts_matching_lines(self) -> None:
        steps, losses, times = parse_llama3_output(SAMPLE_LOG)
        assert steps == [1, 2, 3]
        assert losses == [11.234, 10.567, 9.876]
        assert times == [5.432, 4.321, 4.111]

    def test_empty_input(self) -> None:
        steps, losses, times = parse_llama3_output("")
        assert steps == []
        assert losses == []
        assert times == []

    def test_no_matching_lines(self) -> None:
        text = "no relevant data here\njust some random text\n"
        steps, losses, times = parse_llama3_output(text)
        assert steps == []
        assert losses == []
        assert times == []

    def test_partial_match_skipped(self) -> None:
        # Line has global_step and reduced_train_loss but no train_step_timing
        text = "[INFO] global_step: 5 epoch: 0 reduced_train_loss: 8.0\n"
        steps, losses, times = parse_llama3_output(text)
        assert steps == []


class TestComputeSteadyState:
    def test_constant_array(self) -> None:
        arr = [1.0] * 30
        idx, val = compute_steady_state(arr, window_size=10, std_thresh=0.1, min_windows=3)
        assert idx is not None
        assert idx == 0
        assert val is not None
        assert abs(val - 1.0) < 1e-6

    def test_noisy_then_stable(self) -> None:
        # First 15 values are noisy, then 30 constant values
        noisy = [float(i * 10) for i in range(15)]
        stable = [5.0] * 30
        arr = noisy + stable
        idx, val = compute_steady_state(arr, window_size=10, std_thresh=0.1, min_windows=3)
        assert idx is not None
        # Steady state should be detected within the stable region
        assert idx >= 15
        assert val is not None
        assert abs(val - 5.0) < 1e-6

    def test_all_noisy_returns_none(self) -> None:
        # Each value differs significantly from its neighbours
        arr = [float(i * 100) for i in range(30)]
        idx, val = compute_steady_state(arr, window_size=10, std_thresh=0.01, min_windows=3)
        assert idx is None
        assert val is None

    def test_short_array_returns_none(self) -> None:
        arr = [1.0, 2.0, 3.0]
        idx, val = compute_steady_state(arr, window_size=10, std_thresh=0.1, min_windows=3)
        assert idx is None
        assert val is None

    def test_empty_array_returns_none(self) -> None:
        idx, val = compute_steady_state([], window_size=10, std_thresh=0.1, min_windows=3)
        assert idx is None
        assert val is None
