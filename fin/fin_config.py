"""
Configuration for ReflectGuard-Fin evaluation pipeline.
"""

import os
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
FIN_DIR          = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR         = os.path.dirname(FIN_DIR)
FIN_BENCHMARK_PATH = os.path.join(FIN_DIR, "fin_benchmark.jsonl")
FIN_RESULTS_DIR  = os.path.join(ROOT_DIR, "results")

# ── Finance Harm Taxonomy ──────────────────────────────────────────────────────
# 7 categories derived from FinSafetyBench (arXiv:2605.00706) subcategories
# and TRIDENT (arXiv:2507.21134) CFA Institute ethics framework.
FIN_CATEGORIES = {
    "FIN-1": {
        "label": "Financial Fraud & Scams",
        "description": "Phishing, Ponzi schemes, investment fraud, advance-fee fraud",
    },
    "FIN-2": {
        "label": "Market Manipulation & Insider Trading",
        "description": "Pump-and-dump, spoofing, front-running, material non-public info",
    },
    "FIN-3": {
        "label": "Money Laundering & Structuring",
        "description": "Placement, layering, smurfing, trade-based laundering",
    },
    "FIN-4": {
        "label": "Unauthorized Investment Advice",
        "description": "Unlicensed advisory, suitability bypass, fiduciary duty violations",
    },
    "FIN-5": {
        "label": "Identity Theft & Account Fraud",
        "description": "Account takeover, synthetic identity, credential stuffing",
    },
    "FIN-6": {
        "label": "Tax Evasion & Regulatory Evasion",
        "description": "Offshore accounts, unreported income, AML/KYC evasion",
    },
    "FIN-7": {
        "label": "Predatory Lending & Financial Misconduct",
        "description": "Loan sharking, mis-selling, churning, undisclosed conflicts",
    },
}

FIN_CATEGORY_CODES = list(FIN_CATEGORIES.keys())

# ── Keyword → category mapping (for auto-labelling FinSafetyBench subcategories) ──
# Each entry: list of substrings (lowercase) → category code
SUBCATEGORY_KEYWORD_MAP = [
    (["fraud", "scam", "phishing", "ponzi", "pyramid", "advance fee", "wire fraud"], "FIN-1"),
    (["market manipul", "insider", "pump", "dump", "spoofing", "front.run", "wash trad"], "FIN-2"),
    (["launder", "structur", "smurfing", "placement", "layering"], "FIN-3"),
    (["investment advice", "unlicensed", "suitability", "fiduciary"], "FIN-4"),
    (["identity theft", "account takeover", "synthetic identity", "credential"], "FIN-5"),
    (["tax evasi", "offshore", "unreported", "aml", "kyc", "regulatory evasi"], "FIN-6"),
    (["predatory lend", "loan shark", "mis.sell", "churning", "conflict of interest"], "FIN-7"),
]


def map_subcategory_to_fin(subcategory: str) -> str:
    """Map a FinSafetyBench/TRIDENT subcategory string to our FIN-N code.
    Returns 'FIN-UNK' if no keyword matches."""
    s = subcategory.lower()
    import re
    for keywords, code in SUBCATEGORY_KEYWORD_MAP:
        for kw in keywords:
            if re.search(kw, s):
                return code
    return "FIN-UNK"
