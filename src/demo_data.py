"""
demo_data.py
Generate a realistic synthetic CVE dataset that mirrors NVD API 2.0 output.
Used for CI, offline testing, and demos without network access.
The synthetic CVSS base scores are computed with the real CVSS v3.1 formula
so that Linear Regression has a genuine signal to learn from.
"""

import json
import math
import random
from pathlib import Path
from typing import Optional
from src.config import RAW_DATA_DIR

RANDOM_SEED = 42

ATTACK_VECTORS = ["NETWORK", "ADJACENT", "LOCAL", "PHYSICAL"]
ATTACK_COMPLEXITIES = ["LOW", "HIGH"]
PRIVILEGES_REQUIRED = ["NONE", "LOW", "HIGH"]
USER_INTERACTIONS = ["NONE", "REQUIRED"]
SCOPES = ["UNCHANGED", "CHANGED"]
IMPACTS = ["NONE", "LOW", "HIGH"]

# Realistic CWE distribution (top weaknesses from NVD historical data)
CWE_POOL = [
    ("CWE-79", 0.12),   # XSS
    ("CWE-89", 0.09),   # SQL injection
    ("CWE-20", 0.07),   # improper input validation
    ("CWE-125", 0.06),  # out-of-bounds read
    ("CWE-787", 0.06),  # out-of-bounds write
    ("CWE-416", 0.05),  # use after free
    ("CWE-22", 0.04),   # path traversal
    ("CWE-78", 0.04),   # OS command injection
    ("CWE-190", 0.03),  # integer overflow
    ("CWE-476", 0.03),  # NULL pointer dereference
    ("CWE-200", 0.03),  # info disclosure
    ("CWE-352", 0.03),  # CSRF
    ("CWE-119", 0.03),  # buffer errors
    ("CWE-502", 0.02),  # deserialization
    ("CWE-284", 0.02),  # improper access control
    ("CWE-918", 0.02),  # SSRF
    ("CWE-601", 0.02),  # open redirect
    ("CWE-434", 0.02),  # unrestricted upload
    ("CWE-611", 0.02),  # XXE
    ("CWE-362", 0.02),  # race condition
    ("CWE-OTHER", 0.18),
]

_CWES, _CWE_PROBS = zip(*CWE_POOL)

def _cvss31_base_score(av, ac, pr, ui, s, c, i, a) -> float:
    """
    Exact CVSS v3.1 base score formula.
    Reference: https://www.first.org/cvss/specification-document
    """
    AV = {"NETWORK": 0.85, "ADJACENT": 0.62, "LOCAL": 0.55, "PHYSICAL": 0.20}[av]
    AC = {"LOW": 0.77, "HIGH": 0.44}[ac]
    PR_unchanged = {"NONE": 0.85, "LOW": 0.62, "HIGH": 0.27}[pr]
    PR_changed = {"NONE": 0.85, "LOW": 0.68, "HIGH": 0.50}[pr]
    UI = {"NONE": 0.85, "REQUIRED": 0.62}[ui]
    S_flag = s == "CHANGED"
    PR_val = PR_changed if S_flag else PR_unchanged

    C_val = {"NONE": 0.00, "LOW": 0.22, "HIGH": 0.56}[c]
    I_val = {"NONE": 0.00, "LOW": 0.22, "HIGH": 0.56}[i]
    A_val = {"NONE": 0.00, "LOW": 0.22, "HIGH": 0.56}[a]

    ISS = 1 - (1 - C_val) * (1 - I_val) * (1 - A_val)
    if S_flag:
        impact = 7.52 * (ISS - 0.029) - 3.25 * ((ISS - 0.02) ** 15)
    else:
        impact = 6.42 * ISS

    exploitability = 8.22 * AV * AC * PR_val * UI

    if impact <= 0:
        return 0.0

    if S_flag:
        base = min(1.08 * (impact + exploitability), 10)
    else:
        base = min(impact + exploitability, 10)

    # Round up to 1 decimal
    return math.ceil(base * 10) / 10

def _severity_from_score(score: float) -> str:
    if score >= 9.0:
        return "CRITICAL"
    elif score >= 7.0:
        return "HIGH"
    elif score >= 4.0:
        return "MEDIUM"
    else:
        return "LOW"

def generate_demo_data(n: int = 30_000, output_path: Optional[Path] = None) -> list:
    """
    Create *n* synthetic CVE records in NVD API 2.0 format.
    CVSS scores are computed with the real formula, so the signal is real.
    """
    rng = random.Random(RANDOM_SEED)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or RAW_DATA_DIR / "cves.json"

    records = []
    for idx in range(n):
        year = rng.randint(2015, 2025)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)

        av = rng.choices(ATTACK_VECTORS, weights=[0.55, 0.12, 0.27, 0.06])[0]
        ac = rng.choices(ATTACK_COMPLEXITIES, weights=[0.70, 0.30])[0]
        pr = rng.choices(PRIVILEGES_REQUIRED, weights=[0.50, 0.30, 0.20])[0]
        ui = rng.choices(USER_INTERACTIONS, weights=[0.65, 0.35])[0]
        scope = rng.choices(SCOPES, weights=[0.80, 0.20])[0]
        conf = rng.choices(IMPACTS, weights=[0.25, 0.20, 0.55])[0]
        integ = rng.choices(IMPACTS, weights=[0.25, 0.20, 0.55])[0]
        avail = rng.choices(IMPACTS, weights=[0.25, 0.20, 0.55])[0]

        base_score = _cvss31_base_score(av, ac, pr, ui, scope, conf, integ, avail)
        severity = _severity_from_score(base_score)

        iss = 1 - (
            (1 - {"NONE": 0.00, "LOW": 0.22, "HIGH": 0.56}[conf])
            * (1 - {"NONE": 0.00, "LOW": 0.22, "HIGH": 0.56}[integ])
            * (1 - {"NONE": 0.00, "LOW": 0.22, "HIGH": 0.56}[avail])
        )
        impact_score = round(min(6.42 * iss, 10.0), 2)
        exploitability_score = round(
            8.22
            * {"NETWORK": 0.85, "ADJACENT": 0.62, "LOCAL": 0.55, "PHYSICAL": 0.20}[av]
            * {"LOW": 0.77, "HIGH": 0.44}[ac]
            * {"NONE": 0.85, "LOW": 0.62, "HIGH": 0.27}[pr]
            * {"NONE": 0.85, "REQUIRED": 0.62}[ui],
            2,
        )

        cwe = rng.choices(_CWES, weights=_CWE_PROBS)[0]

        cve_id = f"CVE-{year}-{10000 + idx:05d}"
        pub_date = f"{year}-{month:02d}-{day:02d}T00:00:00.000"

        record = {
            "cve": {
                "id": cve_id,
                "published": pub_date,
                "weaknesses": [
                    {
                        "description": [{"lang": "en", "value": cwe}]
                    }
                ],
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "exploitabilityScore": exploitability_score,
                            "impactScore": impact_score,
                            "cvssData": {
                                "attackVector": av,
                                "attackComplexity": ac,
                                "privilegesRequired": pr,
                                "userInteraction": ui,
                                "scope": scope,
                                "confidentialityImpact": conf,
                                "integrityImpact": integ,
                                "availabilityImpact": avail,
                                "baseScore": base_score,
                                "baseSeverity": severity,
                            },
                        }
                    ]
                },
            }
        }
        records.append(record)

    with open(output_path, "w") as fh:
        json.dump(records, fh)

    print(f"[demo] Generated {len(records)} synthetic CVE records -> {output_path}")
    return records
