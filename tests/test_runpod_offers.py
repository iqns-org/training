from orchestrator.models import Offer
from orchestrator.providers.runpod import map_offers

# Trimmed real catalog entries (captured live 2026-11-22).
FIXTURE = [
    {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "memory": 24,
     "availability": "MEDIUM", "community": True, "secure": True,
     "price": {"community": 0.34, "secure": 0.89, "serverless": 1.34}},
    {"id": "NVIDIA RTX A5000", "name": "RTX A5000", "memory": 24,
     "availability": "NONE", "community": True, "secure": True,
     "price": {"community": 0.16, "secure": 0.27}},
    {"id": "NVIDIA A40", "name": "A40", "memory": 48,
     "availability": "MEDIUM", "community": True, "secure": True,
     "price": {"community": 0.35, "secure": 0.59}},
    {"id": "AMD Instinct MI300X OAM", "name": "MI300X", "memory": 192,
     "availability": "LOW", "community": False, "secure": True,
     "price": {"community": 0.5, "secure": 2.39}},
    {"id": "NVIDIA A100-SXM4-80GB", "name": "A100 SXM", "memory": 80,
     "availability": "HIGH", "community": True, "secure": True,
     "price": {"secure": 2.19}},          # no community price → community skipped
]


def test_offers_filter_and_sort():
    offers = map_offers(FIXTURE)
    ids = [(o.gpu_id, o.secure) for o in offers]
    # A5000 dropped (availability NONE); MI300X community dropped (flag False);
    # A100 community dropped (no price).
    assert ("NVIDIA RTX A5000", False) not in ids
    assert ("AMD Instinct MI300X OAM", False) not in ids
    assert ("NVIDIA A100-SXM4-80GB", False) not in ids
    assert ("NVIDIA RTX A5000", True) not in ids       # NONE drops both tiers
    # cheapest first
    prices = [o.price_hr for o in offers]
    assert prices == sorted(prices)
    assert all(isinstance(o, Offer) for o in offers)


def test_offer_fields():
    o = map_offers(FIXTURE, ["4090"], "community")[0]
    assert (o.gpu_id, o.vram_gb, o.price_hr, o.secure, o.availability) == (
        "NVIDIA GeForce RTX 4090", 24, 0.34, False, "MEDIUM")
