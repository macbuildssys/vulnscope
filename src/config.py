# Global configuration and constants for VulnScope.

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_RESULTS_PER_PAGE = 2000
NVD_RATE_LIMIT_SLEEP = 6.5        # seconds between pages (no key: 5 req/30s)
NVD_RATE_LIMIT_SLEEP_KEYED = 0.7  # seconds with API key (50 req/30s)

NVD_DEFAULT_START_YEAR = 1990
NVD_DEFAULT_END_YEAR   = 2026

CVSS_ATTACK_VECTOR = {
    "NETWORK": 3,
    "ADJACENT": 2,
    "LOCAL": 1,
    "PHYSICAL": 0,
    "ADJACENT_NETWORK": 2,  # legacy alias
}

CVSS_ATTACK_COMPLEXITY = {"LOW": 1, "HIGH": 0}
CVSS_PRIVILEGES_REQUIRED = {"NONE": 2, "LOW": 1, "HIGH": 0}
CVSS_USER_INTERACTION = {"NONE": 1, "REQUIRED": 0}
CVSS_SCOPE = {"UNCHANGED": 0, "CHANGED": 1}
CVSS_IMPACT = {"NONE": 0, "LOW": 1, "HIGH": 2}

SEVERITY_TO_INT = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
INT_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_INT.items()}

CRITICAL_THRESHOLD = 9.0    # baseScore >= this => Critical
TOP_N_CWE = 25              # how many distinct CWEs to keep; rest -> CWE-OTHER
RANDOM_STATE = 42
TEST_SIZE = 0.20

"""
 Supply-chain vulnerability CWE definitions
 These are the weakness categories most commonly propagated through the software
 supply chain: third-party library memory bugs, deserialization, code-execution
 via dependencies, and data-handling flaws in upstream parsers.
"""

SUPPLY_CHAIN_CWES = {
    # Memory safety. C/C++ libraries propagated through dependency trees
    "CWE-787",  # Out-of-bounds Write
    "CWE-125",  # Out-of-bounds Read
    "CWE-416",  # Use After Free
    "CWE-190",  # Integer Overflow
    "CWE-119",  # Improper Restriction of Operations within Buffer
    "CWE-476",  # NULL Pointer Dereference
    # Code execution through third-party components
    "CWE-78",   # OS Command Injection
    "CWE-502",  # Deserialization of Untrusted Data
    # Data handling in upstream parsers / HTTP clients
    "CWE-611",  # XML External Entity (XXE)
    "CWE-918",  # Server-Side Request Forgery (SSRF)
    "CWE-22",   # Path Traversal
    # Access control defects in shared components
    "CWE-284",  # Improper Access Control
    "CWE-200",  # Information Exposure
    # Web-component injection (XSS / SQLi propagated via dependencies)
    "CWE-79",   # Cross-Site Scripting
    "CWE-89",   # SQL Injection
    # Input validation in third-party parsers
    "CWE-20",   # Improper Input Validation
    "CWE-362",  # Race Condition in Shared Libraries
}

# Human-readable grouping for charts
SUPPLY_CHAIN_CATEGORIES = {
    "Memory Safety":          {"CWE-787", "CWE-125", "CWE-416", "CWE-190", "CWE-119", "CWE-476"},
    "Code Execution":         {"CWE-78",  "CWE-502"},
    "Parser / Protocol Bugs": {"CWE-611", "CWE-918", "CWE-22"},
    "Access Control":         {"CWE-284", "CWE-200"},
    "Injection":              {"CWE-79",  "CWE-89",  "CWE-20", "CWE-362"},
}
