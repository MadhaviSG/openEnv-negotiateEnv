"""Load negotiation scenarios from HuggingFace SyntheticSaasDataset.

The dataset is an Excel file: mayukareddy/SyntheticSaasDataset
File: saas_buyer_synthetic_200.xlsx

Actual columns:
    id, company_size, seat_count, saas_product, vendor,
    list_price, competitor_price, Budget,
    vendor_floor_price_hidden, contract_length_months, urgency

Falls back to the built-in SCENARIOS list if the dataset is unavailable.

Usage:
    from negotiate_env.dataset_loader import load_scenarios
    scenarios = load_scenarios()          # HF dataset or fallback
    scenarios = load_scenarios(hf=False)  # built-in only
"""

from __future__ import annotations

import random
from typing import Any

from negotiate_env.scenarios import SCENARIOS as _BUILTIN_SCENARIOS

_HF_REPO = "mayukareddy/SyntheticSaasDataset"
_HF_FILE = "saas_buyer_synthetic_200.xlsx"


def _xlsx_row_to_scenario(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a row from the HF xlsx file to the internal scenario schema."""
    list_price = float(row.get("list_price") or 100.0)
    floor_price = float(row.get("vendor_floor_price_hidden") or list_price * 0.75)
    competitor_price = float(row.get("competitor_price") or list_price * 0.85)
    seat_count = int(row.get("seat_count") or 50)
    # contract_length_months → years
    contract_months = float(row.get("contract_length_months") or 12)
    contract_years = max(1.0, round(contract_months / 12, 1))
    budget = float(row.get("Budget") or list_price * seat_count * contract_years * 0.9)
    urgency = str(row.get("urgency") or "medium").lower()
    vendor = str(row.get("vendor") or "SaaS Vendor")
    product = str(row.get("saas_product") or "Enterprise Software")
    company_size = str(row.get("company_size") or "mid-market")
    row_id = row.get("id", random.randint(1000, 9999))

    # Agent budget ceiling: slightly above competitor price but below list
    agent_max_price = round(min(competitor_price * 1.05, list_price * 0.95), 2)
    agent_max_length = min(3.0, contract_years + 1)
    agent_max_cap = 6.0

    vendor_preferred_length = max(1.0, min(3.0, contract_years))

    drift_turn_map = {"high": 2, "medium": 3, "low": 5}
    drift_turn = drift_turn_map.get(urgency, 3)

    return {
        "id": f"hf_{row_id}",
        "product": f"{vendor} {product}, {seat_count} seats",
        "context": (
            f"You are the procurement manager for a {company_size} company. "
            f"You are negotiating a {product} contract with {vendor} for {seat_count} seats "
            f"over {int(contract_years)} year(s). Your total budget is ${budget:,.0f}. "
            f"A competitor offers the same capability at ${competitor_price:.2f}/seat/month."
        ),
        "agent_max_price": agent_max_price,
        "agent_max_length": agent_max_length,
        "agent_max_cap": agent_max_cap,
        "vendor_list_price": list_price,
        "vendor_floor_price": floor_price,
        "vendor_preferred_length": vendor_preferred_length,
        "vendor_max_cap": 8.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            f"{product} for {seat_count} seats is ${list_price:.2f} per seat per month. "
            f"Our standard is a {int(vendor_preferred_length)}-year agreement. "
            "What timeline are you working with?"
        ),
        "opponent_strategy": random.choice(
            ["cooperative", "concession_trader", "hardball", "urgency"]
        ),
        "drift_event": _drift_event(urgency),
        "drift_turn": drift_turn,
        "source": _HF_REPO,
        # Extra fields kept for reference
        "seat_count": seat_count,
        "contract_length": contract_years,
        "budget": budget,
        "competitor_price": competitor_price,
    }


def _drift_event(urgency: str) -> str:
    events = {
        "high": "Board requires contract signed this quarter — timeline compressed.",
        "medium": "Budget review next month — CFO wants cost certainty.",
        "low": "No immediate pressure, but competitor pricing may change.",
    }
    return events.get(urgency, "Budget constraints tightened.")


def _load_from_hf(max_rows: int) -> list[dict[str, Any]]:
    """Download and parse the xlsx file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download  # type: ignore
    import openpyxl  # type: ignore

    path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=_HF_FILE,
        repo_type="dataset",
    )

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]

    scenarios = []
    for raw in rows[1 : max_rows + 1]:
        row = dict(zip(headers, raw))
        try:
            scenarios.append(_xlsx_row_to_scenario(row))
        except Exception:
            continue  # skip malformed rows

    wb.close()
    return scenarios


def load_scenarios(hf: bool = True, max_rows: int = 200) -> list[dict[str, Any]]:
    """Return a list of scenario dicts compatible with NegotiateEnvironment.

    Args:
        hf: If True, attempt to load from HuggingFace dataset first.
        max_rows: Maximum number of rows to load from the xlsx (default 200).

    Returns:
        List of scenario dicts. Falls back to built-in 20 scenarios on any error.
    """
    if hf:
        try:
            scenarios = _load_from_hf(max_rows)
            print(f"[dataset_loader] Loaded {len(scenarios)} scenarios from HuggingFace ({_HF_REPO}).")
            return scenarios
        except Exception as exc:
            print(f"[dataset_loader] HF load failed ({exc}). Using built-in scenarios.")

    print(f"[dataset_loader] Using {len(_BUILTIN_SCENARIOS)} built-in scenarios.")
    return list(_BUILTIN_SCENARIOS)
