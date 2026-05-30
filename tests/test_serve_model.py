from __future__ import annotations

from training import serve_model


def test_resolve_adapter_path_prefers_existing_default_candidate(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path
    default_dir = repo_root / "models" / "bu-30b-lora"
    legacy_dir = repo_root / "models" / "bu-30b-login-500-lora"
    legacy_dir.mkdir(parents=True)

    monkeypatch.setattr(serve_model, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(serve_model, "_DEFAULT_ADAPTER_PATH", str(default_dir))
    monkeypatch.setattr(serve_model, "DEFAULT_ADAPTER_PATH", str(default_dir))
    monkeypatch.setattr(serve_model, "DEFAULT_ADAPTER_CANDIDATES", (default_dir, legacy_dir))

    resolved = serve_model.resolve_adapter_path(str(default_dir))

    assert resolved == legacy_dir.resolve()


def test_build_preflight_report_surfaces_missing_runtime_and_adapter_files(tmp_path, monkeypatch) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    monkeypatch.setattr(serve_model, "missing_runtime_dependencies", lambda backend: ["torch", "transformers"])

    report = serve_model.build_preflight_report(backend="hf", adapter_path=adapter_dir)

    assert report["adapter_exists"] is True
    assert report["missing_adapter_files"] == ["adapter_config.json", "adapter_model.safetensors"]
    assert report["missing_runtime_dependencies"] == ["torch", "transformers"]
    assert isinstance(report["default_adapter_candidates"], list)


def test_resolve_adapter_path_keeps_explicit_non_default_path(tmp_path) -> None:
    explicit = tmp_path / "custom-adapter"

    resolved = serve_model.resolve_adapter_path(str(explicit))

    assert resolved == explicit.resolve()
