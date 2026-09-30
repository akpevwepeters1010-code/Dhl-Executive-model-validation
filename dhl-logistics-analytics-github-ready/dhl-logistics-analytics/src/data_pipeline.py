"""Data preparation utilities based on the capstone architecture."""
import hashlib
import pandas as pd

def anonymize(value: str, salt: str) -> str:
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()

def prepare(df: pd.DataFrame, salt: str) -> pd.DataFrame:
    out = df.copy()
    if "customer_id" in out:
        out["customer_id"] = out["customer_id"].map(lambda x: anonymize(x, salt))
    numeric = [
        "transit_duration", "weather_severity", "traffic_density",
        "route_distance", "package_weight"
    ]
    for col in numeric:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(out[col].median())
            lo, hi = out[col].quantile([0.01, 0.99])
            out[col] = out[col].clip(lo, hi)
    return out
