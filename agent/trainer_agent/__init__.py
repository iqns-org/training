"""trainer_agent — in-pod control plane (SPEC §5).

Lives inside the generic ``trainer`` image. v0 slice: health/info + a single
active job running an arbitrary command with log capture, cancellation and
progress scraping. Presigned transfer + artifact gate/manifest land with P1.
"""

# 0.5.1: co-tenant pair mode (payload configs[] -> N job yamls, attested
# per-config), gate PATTERN entries (model*.safetensors shards + per-slug
# trees), and the DEADMAN RE-ARM (the line 1d52eae swallowed was
# unreachable ever since — repo 86d5e9a; this image is where the fleet
# actually gets it back). Deployed 0.5.0 images train with the watchdog
# DISARMED: this bump is the release that fixes safety, not just features.
__version__ = "0.5.2"
