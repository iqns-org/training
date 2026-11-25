"""trainer_agent — in-pod control plane (SPEC §5).

Lives inside the generic ``trainer`` image. v0 slice: health/info + a single
active job running an arbitrary command with log capture, cancellation and
progress scraping. Presigned transfer + artifact gate/manifest land with P1.
"""

__version__ = "0.3.0"
