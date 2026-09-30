"""Generate clearly labeled synthetic shipment data for local testing."""
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

def generate(n: int = 5500) -> pd.DataFrame:
    hubs = ["HUB_US", "HUB_EU", "HUB_APAC", "HUB_LATAM"]
    services = ["Express_Worldwide", "Same_Day", "Economy_Select"]

    hub = RNG.choice(hubs, n, p=[.42, .30, .20, .08])
    service = RNG.choice(services, n, p=[.55, .20, .25])
    weather = RNG.integers(0, 6, n)
    traffic = RNG.integers(0, 11, n)
    transit = np.maximum(RNG.normal(18, 6, n), 1)
    distance = np.maximum(RNG.normal(850, 350, n), 10)
    weight = np.maximum(RNG.lognormal(2.0, .6, n), .1)

    logit = (
        -3.2
        + 0.34 * weather
        + 0.16 * traffic
        + 0.055 * transit
        + 0.00035 * distance
        + 0.015 * weight
        + (service == "Economy_Select") * 0.65
        + (hub == "HUB_LATAM") * 0.35
    )
    probability = 1 / (1 + np.exp(-logit))
    delayed = RNG.binomial(1, probability)

    return pd.DataFrame({
        "shipment_id": [f"DEMO-{i:06d}" for i in range(n)],
        "customer_id": [f"CUST-{RNG.integers(100000, 999999)}" for _ in range(n)],
        "hub_location": hub,
        "service_level": service,
        "transit_duration": transit.round(2),
        "weather_severity": weather,
        "traffic_density": traffic,
        "route_distance": distance.round(2),
        "package_weight": weight.round(2),
        "delayed": delayed,
    })

if __name__ == "__main__":
    out = Path("data/raw/demo_shipments.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(out, index=False)
    print(f"Wrote {out} (SYNTHETIC DEMO DATA ONLY)")
