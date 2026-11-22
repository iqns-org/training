import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gpuai_deploy_reports_gpu_template_or_api_failure(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_gpu = fake_bin / "gpu"
    fake_gpu.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "if [ \"$1\" = \"keys\" ] && [ \"$2\" = \"list\" ]; then\n"
        "  echo \"abc123 $(hostname)\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"templates\" ] && [ \"$2\" = \"list\" ]; then\n"
        "  echo '[]'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"deploy\" ]; then\n"
        "  echo 'template not found or account mismatch' >&2\n"
        "  exit 1\n"
        "fi\n"
        "echo 'unexpected command: $*' >&2\n"
        "exit 0\n"
    )
    fake_gpu.chmod(0o755)

    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    result = subprocess.run(
        ["bash", "bin/gpuai-deploy", "-q", "queer"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    combined = f"{result.stdout}\n{result.stderr}".lower()
    assert "template" in combined
    assert "gpu deploy" in combined or "gpu template" in combined


def test_extract_gpu_json_payload_ignores_polling_prefix():
    payload = (
        "Submitted: deploy operation 93ae52be-aade-499b-ba36-398e306edf51 (state=pending) — polling...\n"
        '{\n'
        '  "instance": {\n'
        '    "id": "gpu-abc",\n'
        '    "status": "allocating"\n'
        '  },\n'
        '  "app_url": "https://example.test"\n'
        '}\n'
    )

    script = (
        "source ./bin/gpuai-deploy; "
        "payload=$(cat <<'EOF'\n"
        f"{payload}"
        "EOF\n)\n"
        "printf '%s' \"$payload\" | extract_gpu_json_payload | jq -r '.instance.id'"
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-lc", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gpu-abc"


def test_instance_is_active_rejects_error_state():
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-lc",
            "source ./bin/gpuai-deploy; if instance_is_active 'error'; then echo active; else echo inactive; fi",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "inactive"
