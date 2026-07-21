import torch

from koopman_sbi.config import load_experiment_config
from koopman_sbi.experiments.pipeline import _build_benchmark_suite_specs


def test_benchmark_suite_reuses_single_flow_model_instance(tiny_config_path):
    config = load_experiment_config(str(tiny_config_path))
    device = torch.device("cpu")

    class FakeModel:
        pass

    flow_model = FakeModel()
    koopman_model = FakeModel()
    npe_model = FakeModel()
    cmpe_model = FakeModel()

    from koopman_sbi.experiments import pipeline as pipeline_module

    original_flow_load = pipeline_module.ConditionalFlowMatching.load
    original_koopman_load = pipeline_module.KoopmanFlow.load
    original_npe_load = pipeline_module.NormalizingFlowNPE.load
    original_cmpe_load = pipeline_module.ConsistencyModelPosteriorEstimator.load
    original_summary = pipeline_module._read_run_summary_from_checkpoint
    try:
        pipeline_module.ConditionalFlowMatching.load = classmethod(lambda cls, path, device: flow_model)
        pipeline_module.KoopmanFlow.load = classmethod(lambda cls, path, device: koopman_model)
        pipeline_module.NormalizingFlowNPE.load = classmethod(lambda cls, path, device: npe_model)
        pipeline_module.ConsistencyModelPosteriorEstimator.load = classmethod(
            lambda cls, path, device: cmpe_model
        )
        pipeline_module._read_run_summary_from_checkpoint = lambda path: {"best_checkpoint_path": str(path)}

        specs = _build_benchmark_suite_specs(
            config=config,
            device=device,
            flow_checkpoint=tiny_config_path,
            koopman_checkpoint=tiny_config_path,
            npe_checkpoint=tiny_config_path,
            cmpe_checkpoint=tiny_config_path,
        )
    finally:
        pipeline_module.ConditionalFlowMatching.load = original_flow_load
        pipeline_module.KoopmanFlow.load = original_koopman_load
        pipeline_module.NormalizingFlowNPE.load = original_npe_load
        pipeline_module.ConsistencyModelPosteriorEstimator.load = original_cmpe_load
        pipeline_module._read_run_summary_from_checkpoint = original_summary

    flow_specs = [spec for spec in specs if spec.label.startswith("fmnpe_")]
    assert len(flow_specs) == 2
    assert all(spec.model is flow_model for spec in flow_specs)
    cmpe_specs = [spec for spec in specs if spec.label.startswith("cmpe_")]
    assert len(cmpe_specs) == 1
    assert cmpe_specs[0].model is cmpe_model
