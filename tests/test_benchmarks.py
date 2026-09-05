import json
from sglkit.benchmarks import BenchmarkResult


def test_benchmark_save(tmp_path):
    result = BenchmarkResult(
        experiment_id="demo",
        model_name="scalar_monopole",
        seed=1,
        metrics={"mse": 0.1},
    )
    path = result.save_json(tmp_path / "result.json")
    with path.open() as handle:
        data = json.load(handle)
    assert data["benchmark_family"] == "solar_gravitational_lens"
