import os

import pytest

from orchestrator import storage


@pytest.fixture()
def fake_env(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT", "http://pi00.lan:30900")
    monkeypatch.setenv("S3_ACCESS_KEY", "AKIATEST")
    monkeypatch.setenv("S3_SECRET_KEY", "secretkey")
    monkeypatch.setenv("S3_BUCKET", "quadra-ml")


def test_presign_shape(fake_env):
    url = storage.presign("GET", "datasets/queer/abc/data.jsonl", expires=3600)
    assert url.startswith("http://pi00.lan:30900/quadra-ml/datasets/queer/abc/data.jsonl?")
    for token in ("X-Amz-Algorithm=AWS4-HMAC-SHA256", "X-Amz-Credential=AKIATEST",
                  "X-Amz-Signature=", "X-Amz-Expires=3600",
                  "X-Amz-SignedHeaders=host"):
        assert token in url, token


def test_mint_job_urls_keys(fake_env):
    urls = storage.mint_job_urls("run-1", dataset_key="datasets/queer/hash/data.jsonl")
    assert set(urls) == {"payload_get", "dataset_get", "artifacts_put",
                         "manifest_put", "log_put"}
    assert "runs/run-1/payload.json" in urls["payload_get"]
    assert "runs/run-1/artifacts.tar.gz" in urls["artifacts_put"]
    assert urls["dataset_get"]  # non-empty when key given


def test_missing_creds_raises(monkeypatch):
    for k in ("S3_ACCESS_KEY", "S3_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(storage.StorageError):
        storage.presign("GET", "x")


def test_vol_cfg_off_without_env(monkeypatch):
    for k in ("RUNPOD_VOLUME_ID", "RUNPOD_S3_ACCESS_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert storage.vol_cfg() is None


def test_vol_cfg_reads_runpod_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_VOLUME_ID", "vol1")
    monkeypatch.setenv("RUNPOD_S3_ACCESS_KEY", "user_x")
    monkeypatch.setenv("RUNPOD_S3_SECRET_KEY", "rps_y")
    monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api-eu-ro-1.runpod.io/")
    c = storage.vol_cfg()
    assert c["bucket"] == "vol1" and c["endpoint"] == "https://s3api-eu-ro-1.runpod.io"
    assert c["dc"] == "EU-RO-1" and c["mount"] == "/data"


def test_sha256_file(tmp_path):
    import hashlib

    p = tmp_path / "a.jsonl"
    p.write_bytes(b"hello\n")
    assert storage.sha256_file(p) == hashlib.sha256(b"hello\n").hexdigest()
