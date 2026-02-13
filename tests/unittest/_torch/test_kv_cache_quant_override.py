import pytest
import torch

from tensorrt_llm._torch import model_config
from tensorrt_llm._torch.attention_backend import trtllm
from tensorrt_llm._torch.pyexecutor import model_loader
from tensorrt_llm.models import modeling_utils


def _make_model_config(kv_cache_quant_algo):
    return model_config.ModelConfig(quant_config=modeling_utils.QuantConfig(
        kv_cache_quant_algo=kv_cache_quant_algo))


def test_validate_and_set_kv_cache_quant_auto_uses_checkpoint():
    model_config_inst = _make_model_config(modeling_utils.QuantAlgo.FP8)
    model_loader.validate_and_set_kv_cache_quant(model_config_inst, "auto")
    assert model_config_inst.quant_config.kv_cache_quant_algo == modeling_utils.QuantAlgo.FP8


def test_validate_and_set_kv_cache_quant_explicit_dtype_overrides():
    model_config_inst = _make_model_config(modeling_utils.QuantAlgo.FP8)
    model_loader.validate_and_set_kv_cache_quant(model_config_inst, "nvfp4")
    assert model_config_inst.quant_config.kv_cache_quant_algo == modeling_utils.QuantAlgo.NVFP4


def test_validate_and_set_kv_cache_quant_rejects_invalid_dtype():
    model_config_inst = _make_model_config(modeling_utils.QuantAlgo.FP8)
    with pytest.raises(ValueError, match="Accepted types are"):
        model_loader.validate_and_set_kv_cache_quant(model_config_inst,
                                                     "invalid_dtype")


def test_resolve_kv_scales_for_mode_fp8_uses_scalar_override():
    kv_scale_orig_quant = torch.tensor([1.0], dtype=torch.float32)
    kv_scale_quant_orig = torch.tensor([1.0], dtype=torch.float32)
    kv_scales_sf = torch.tensor([0.5], dtype=torch.float32)
    kv_scales_sf_inv = torch.tensor([2.0], dtype=torch.float32)

    resolved = trtllm.TrtllmAttention._resolve_kv_scales_for_mode(
        True,
        False,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        kv_scales_sf,
        kv_scales_sf_inv,
    )

    assert torch.allclose(resolved[0], kv_scales_sf_inv)
    assert torch.allclose(resolved[1], kv_scales_sf)
    assert resolved[2] is None
    assert resolved[3] is None


def test_resolve_kv_scales_for_mode_fp8_accepts_kv_pair():
    kv_scale_orig_quant = torch.tensor([1.0], dtype=torch.float32)
    kv_scale_quant_orig = torch.tensor([1.0], dtype=torch.float32)
    kv_scales_sf = torch.tensor([0.25, 0.5], dtype=torch.float32)
    kv_scales_sf_inv = torch.tensor([4.0, 2.0], dtype=torch.float32)

    resolved = trtllm.TrtllmAttention._resolve_kv_scales_for_mode(
        True,
        False,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        kv_scales_sf,
        kv_scales_sf_inv,
    )

    assert torch.allclose(resolved[0], torch.tensor([2.0], dtype=torch.float32))
    assert torch.allclose(resolved[1], torch.tensor([0.5], dtype=torch.float32))
    assert resolved[2] is None
    assert resolved[3] is None


def test_resolve_kv_scales_for_mode_fp8_accepts_qkv_scale_triplet():
    kv_scale_orig_quant = torch.tensor([1.0], dtype=torch.float32)
    kv_scale_quant_orig = torch.tensor([1.0], dtype=torch.float32)
    kv_scales_sf = torch.tensor([1.0, 0.25, 0.5], dtype=torch.float32)
    kv_scales_sf_inv = torch.tensor([1.0, 4.0, 2.0], dtype=torch.float32)

    resolved = trtllm.TrtllmAttention._resolve_kv_scales_for_mode(
        True,
        False,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        kv_scales_sf,
        kv_scales_sf_inv,
    )

    assert torch.allclose(resolved[0], torch.tensor([2.0], dtype=torch.float32))
    assert torch.allclose(resolved[1], torch.tensor([0.5], dtype=torch.float32))
    assert resolved[2] is None
    assert resolved[3] is None


def test_resolve_kv_scales_for_mode_nvfp4_requires_pair_shape():
    kv_scale_orig_quant = torch.tensor([1.0], dtype=torch.float32)
    kv_scale_quant_orig = torch.tensor([1.0], dtype=torch.float32)
    kv_scales_sf = torch.tensor([0.5], dtype=torch.float32)
    kv_scales_sf_inv = torch.tensor([2.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="shape \\(2\\)"):
        trtllm.TrtllmAttention._resolve_kv_scales_for_mode(
            False,
            True,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            kv_scales_sf,
            kv_scales_sf_inv,
        )


def test_resolve_kv_scales_for_mode_nvfp4_accepts_kv_pair():
    kv_scale_orig_quant = torch.tensor([1.0], dtype=torch.float32)
    kv_scale_quant_orig = torch.tensor([1.0], dtype=torch.float32)
    kv_scales_sf = torch.tensor([0.25, 0.5], dtype=torch.float32)
    kv_scales_sf_inv = torch.tensor([4.0, 2.0], dtype=torch.float32)

    resolved = trtllm.TrtllmAttention._resolve_kv_scales_for_mode(
        False,
        True,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        kv_scales_sf,
        kv_scales_sf_inv,
    )

    assert torch.allclose(resolved[2], torch.tensor([0.25, 0.5], dtype=torch.float32))
    assert torch.allclose(resolved[3], torch.tensor([4.0, 2.0], dtype=torch.float32))


def test_resolve_kv_scales_for_mode_nvfp4_accepts_qkv_scale_triplet():
    kv_scale_orig_quant = torch.tensor([1.0], dtype=torch.float32)
    kv_scale_quant_orig = torch.tensor([1.0], dtype=torch.float32)
    kv_scales_sf = torch.tensor([1.0, 0.25, 0.5], dtype=torch.float32)
    kv_scales_sf_inv = torch.tensor([1.0, 4.0, 2.0], dtype=torch.float32)

    resolved = trtllm.TrtllmAttention._resolve_kv_scales_for_mode(
        False,
        True,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        kv_scales_sf,
        kv_scales_sf_inv,
    )

    assert torch.allclose(resolved[2], torch.tensor([0.25, 0.5], dtype=torch.float32))
    assert torch.allclose(resolved[3], torch.tensor([4.0, 2.0], dtype=torch.float32))


def test_resolve_kv_scales_for_mode_fp8_falls_back_to_defaults():
    kv_scale_orig_quant = torch.tensor([0.25], dtype=torch.float32)
    kv_scale_quant_orig = torch.tensor([4.0], dtype=torch.float32)

    resolved = trtllm.TrtllmAttention._resolve_kv_scales_for_mode(
        True,
        False,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        None,
        None,
    )

    assert torch.allclose(resolved[0], kv_scale_orig_quant)
    assert torch.allclose(resolved[1], kv_scale_quant_orig)
    assert resolved[2] is None
    assert resolved[3] is None
