"""
SentinelIQ — Synthetic Data Generator
======================================
Generates realistic transaction event data with intentionally injected fraud
patterns for ML training, graph feature engineering, and RAG knowledge base seeding.

Output:
    - data/synthetic/events.csv          (~10,000 transaction events)
    - data/synthetic/historical_cases.json (~200 historical case documents)

Usage:
    python scripts/generate_data.py --events 10000 --fraud-rate 0.015 --output data/synthetic/
"""

import random
import uuid
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

# ──────────────────────────────────────────────
# Seed everything for reproducibility
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ──────────────────────────────────────────────
# Constants — pools of realistic values
# ──────────────────────────────────────────────

# Countries we'll assign to accounts. Weighted so most accounts are from
# a few common countries (mirrors real-world distribution).
COUNTRIES = ["US", "UK", "CA", "DE", "FR", "AU", "IN", "BR", "NG", "JP"]
COUNTRY_WEIGHTS = [0.30, 0.15, 0.10, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06]

# Foreign countries used when simulating ATO IP mismatch.
# These are countries the attacker connects from (not the victim's home country).
FOREIGN_COUNTRIES = ["RU", "CN", "NG", "VN", "RO", "UA", "PH", "ID"]


# ──────────────────────────────────────────────
# Step 3: Account Profile Generator
# ──────────────────────────────────────────────

def generate_device_id() -> str:
    """Generate a unique device identifier."""
    return f"DEV-{uuid.uuid4().hex[:8].upper()}"


def generate_ip() -> str:
    """Generate a realistic-looking IPv4 address using Faker."""
    return fake.ipv4_public()


def generate_profiles(num_profiles: int) -> list[dict]:
    """
    Generate a pool of legitimate account profiles.

    Each profile represents a real user with consistent behavioural patterns.
    These baselines are critical — fraud detection works by spotting *deviations*
    from what's normal for each user.

    Args:
        num_profiles: Number of account profiles to generate.

    Returns:
        List of profile dicts, each containing:
            - account_id:              Unique account identifier
            - account_age_days:        How long the account has existed (30-1500 days)
            - home_country:            The country the account was registered in
            - primary_device_id:       The device they normally use
            - primary_ip:              The IP they normally connect from
            - avg_transaction_amount:  Their typical spending range (mean)
    """
    profiles = []

    for i in range(num_profiles):
        # Pick a home country using weighted distribution
        # (more accounts from US/UK, fewer from smaller markets)
        home_country = random.choices(COUNTRIES, weights=COUNTRY_WEIGHTS, k=1)[0]

        # Account age: 30-1500 days (no brand-new accounts in the legit pool)
        # New accounts are reserved for synthetic identity fraud injection
        account_age = random.randint(30, 1500)

        # Average transaction amount follows a log-normal-like distribution:
        #   - Most people spend $50-$200 per transaction (everyday purchases)
        #   - Some spend $200-$500 (higher-value users)
        #   - Very few spend $500+ regularly
        avg_amount = round(random.gauss(mu=150, sigma=80), 2)
        avg_amount = max(20.0, min(avg_amount, 600.0))  # Clamp to realistic range

        profile = {
            "account_id": f"ACC-{i:05d}",
            "account_age_days": account_age,
            "home_country": home_country,
            "primary_device_id": generate_device_id(),
            "primary_ip": generate_ip(),
            "avg_transaction_amount": avg_amount,
        }
        profiles.append(profile)

    return profiles


# ──────────────────────────────────────────────
# Step 4: Normal (Legitimate) Transaction Generator
# ──────────────────────────────────────────────

def generate_normal_transactions(
    profiles: list[dict],
    events_per_profile: tuple[int, int] = (5, 15),
) -> list[dict]:
    """
    Generate legitimate transactions from account profiles.

    Each transaction uses the profile's own device and IP (no mismatches),
    and the amount stays close to the profile's average spending. This is
    what 'normal' looks like — the ML model needs to see lots of this to
    learn the baseline before it can spot deviations.

    Args:
        profiles:           List of account profile dicts from generate_profiles().
        events_per_profile: (min, max) transactions to generate per profile.

    Returns:
        List of transaction event dicts (one dict = one row in events.csv).
    """
    transactions = []

    # Time window: transactions spread over the last 90 days
    now = datetime.now()
    window_start = now - timedelta(days=90)

    for profile in profiles:
        # Each account makes a random number of transactions
        num_events = random.randint(*events_per_profile)

        for _ in range(num_events):
            # ── Transaction amount ──
            # Draw from a gaussian centered on the profile's average.
            # sigma = 30% of the mean → natural spending variation.
            # Example: avg=$150 → most transactions between $105 and $195
            amount = round(
                random.gauss(
                    mu=profile["avg_transaction_amount"],
                    sigma=profile["avg_transaction_amount"] * 0.30,
                ),
                2,
            )
            amount = max(5.0, amount)  # Floor at $5 (no negatives)

            # ── Timestamp ──
            # Random point in the last 90 days
            random_offset = timedelta(
                seconds=random.randint(0, int((now - window_start).total_seconds()))
            )
            timestamp = window_start + random_offset

            # ── Build the transaction event ──
            event = {
                "event_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
                "account_id": profile["account_id"],
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_amount": amount,
                "account_age_days": profile["account_age_days"],
                "device_id": profile["primary_device_id"],       # Own device
                "ip_address": profile["primary_ip"],             # Own IP
                "ip_country_mismatch": 0,                        # No mismatch
                "device_change_count": 0,                        # No device changes
                "velocity_1hr": random.choices(                  # Low velocity
                    [0, 1, 2], weights=[0.5, 0.35, 0.15], k=1
                )[0],
                "avg_txn_amount_30d": profile["avg_transaction_amount"],
                "failed_login_count_24hr": random.choices(       # Rare failed logins
                    [0, 1], weights=[0.85, 0.15], k=1
                )[0],
                "is_fraud": 0,
                "fraud_type": None,
            }
            transactions.append(event)

    return transactions


# ──────────────────────────────────────────────
# Step 5: Account Takeover (ATO) Fraud Injection
# ──────────────────────────────────────────────

def generate_ato_fraud(
    profiles: list[dict],
    num_cases: int,
) -> list[dict]:
    """
    Simulate Account Takeover fraud on existing mature accounts.

    ATO pattern: a legitimate account gets compromised. The attacker logs in
    from a new device, a foreign IP, and hammers out high-value transactions
    as fast as possible before the real owner notices.

    We deliberately violate every "normal" baseline:
        - Device:    NEW (not the profile's primary)
        - IP:        FOREIGN country (ip_country_mismatch = 1)
        - Velocity:  HIGH (5-10 txns/hour — draining the account)
        - Amount:    SPIKED (3x-10x the profile's average)
        - Logins:    FAILED attempts before successful entry (brute force)

    Args:
        profiles:   List of account profile dicts.
        num_cases:  Number of ATO fraud events to generate.

    Returns:
        List of fraudulent transaction event dicts.
    """
    transactions = []

    # Only target mature accounts (age > 180 days)
    # New accounts can't be "taken over" — they have no history to deviate from
    mature_profiles = [p for p in profiles if p["account_age_days"] > 180]

    if not mature_profiles:
        raise ValueError("No mature profiles available for ATO fraud injection.")

    now = datetime.now()
    window_start = now - timedelta(days=90)

    for _ in range(num_cases):
        # Pick a random mature victim
        victim = random.choice(mature_profiles)

        # ── Spike the transaction amount ──
        # 3x to 10x the victim's normal spending
        # If they normally spend $150, ATO transactions are $450 – $1500
        multiplier = random.uniform(3.0, 10.0)
        amount = round(victim["avg_transaction_amount"] * multiplier, 2)

        # ── High velocity ──
        # The attacker is rushing — 5 to 10 transactions per hour
        velocity = random.randint(5, 10)

        # ── Failed logins ──
        # Brute force or credential stuffing leaves traces
        failed_logins = random.randint(3, 6)

        # ── Device changes ──
        # The attacker may have tried from multiple devices
        device_changes = random.randint(2, 5)

        # ── Timestamp ──
        random_offset = timedelta(
            seconds=random.randint(0, int((now - window_start).total_seconds()))
        )
        timestamp = window_start + random_offset

        event = {
            "event_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
            "account_id": victim["account_id"],
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_amount": amount,
            "account_age_days": victim["account_age_days"],
            "device_id": generate_device_id(),           # ← NEW device (not victim's)
            "ip_address": generate_ip(),                 # ← NEW IP (not victim's)
            "ip_country_mismatch": 1,                    # ← FOREIGN country
            "device_change_count": device_changes,       # ← Multiple device changes
            "velocity_1hr": velocity,                    # ← Rapid-fire transactions
            "avg_txn_amount_30d": victim["avg_transaction_amount"],  # History unchanged
            "failed_login_count_24hr": failed_logins,    # ← Brute force traces
            "is_fraud": 1,
            "fraud_type": "account_takeover",
        }
        transactions.append(event)

    return transactions


# ──────────────────────────────────────────────
# Step 6: Synthetic Identity Ring Fraud Injection
# ──────────────────────────────────────────────

def generate_synthetic_identity_fraud(
    num_rings: int = 6,
    accounts_per_ring: tuple[int, int] = (3, 5),
    txns_per_account: tuple[int, int] = (2, 4),
) -> list[dict]:
    """
    Simulate Synthetic Identity fraud rings.

    Pattern: Fraudsters create multiple brand-new fake accounts that all share
    the same physical device and/or IP address. Each individual transaction
    looks plausible on its own — the fraud is only visible when you see the
    *network* of shared infrastructure.

    This is WHY we build a NetworkX graph in Phase 2:
        - Account A uses DEV-SHARED-01
        - Account B uses DEV-SHARED-01  ← Same device!
        - Account C uses DEV-SHARED-01  ← Same device!
        → Graph centrality for DEV-SHARED-01 will be very high

    Args:
        num_rings:          Number of fraud rings to create.
        accounts_per_ring:  (min, max) fake accounts in each ring.
        txns_per_account:   (min, max) transactions per fake account.

    Returns:
        List of fraudulent transaction event dicts.
    """
    transactions = []

    now = datetime.now()
    window_start = now - timedelta(days=90)

    # Track the starting account number to avoid collisions with legit profiles
    ring_account_counter = 90000

    for ring_idx in range(num_rings):
        # ── Each ring shares ONE device and ONE IP ──
        # This is the critical graph signal
        shared_device = generate_device_id()
        shared_ip = generate_ip()

        # How many fake accounts in this ring
        num_accounts = random.randint(*accounts_per_ring)

        for acct_idx in range(num_accounts):
            ring_account_counter += 1
            fake_account_id = f"ACC-{ring_account_counter:05d}"

            # Brand new accounts (1-10 days old)
            account_age = random.randint(1, 10)

            # Fake average history (they have no real history)
            fake_avg = round(random.uniform(50.0, 150.0), 2)

            # Generate multiple transactions per fake account
            num_txns = random.randint(*txns_per_account)

            for _ in range(num_txns):
                # ── Amount: medium-high, testing limits ──
                # Not absurdly high (that would be too obvious)
                # but higher than average — probing credit limits
                amount = round(random.uniform(500.0, 2000.0), 2)

                # ── Moderate velocity ──
                # Not as frantic as ATO, but higher than normal
                velocity = random.randint(3, 5)

                # ── Timestamp ──
                random_offset = timedelta(
                    seconds=random.randint(
                        0, int((now - window_start).total_seconds())
                    )
                )
                timestamp = window_start + random_offset

                event = {
                    "event_id": f"TXN-{uuid.uuid4().hex[:10].upper()}",
                    "account_id": fake_account_id,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "transaction_amount": amount,
                    "account_age_days": account_age,        # ← Very new account
                    "device_id": shared_device,              # ← SHARED across ring
                    "ip_address": shared_ip,                 # ← SHARED across ring
                    "ip_country_mismatch": random.choice([0, 1]),  # Mixed
                    "device_change_count": 0,                # They stick to one device
                    "velocity_1hr": velocity,                # ← Moderate
                    "avg_txn_amount_30d": fake_avg,          # Fake history
                    "failed_login_count_24hr": random.randint(0, 1),  # Low
                    "is_fraud": 1,
                    "fraud_type": "synthetic_identity",
                }
                transactions.append(event)

    return transactions


# ──────────────────────────────────────────────
# Step 7: Combine & Export Dataset
# ──────────────────────────────────────────────

def build_dataset(
    target_events: int = 10000,
    fraud_rate: float = 0.015,
    output_dir: str = "data/synthetic",
):
    """
    Orchestrate the generation of normal traffic and fraud injections,
    combine into a single Pandas DataFrame, shuffle, and export.
    """
    import math

    print(f"Generating ~{target_events} events with a {fraud_rate:.1%} fraud rate...")

    # Calculate target counts
    target_fraud_txns = int(target_events * fraud_rate)
    target_normal_txns = target_events - target_fraud_txns

    # Assume each profile generates ~10 txns on average
    num_profiles = int(target_normal_txns / 10)

    # 1. Generate Legitimate Profiles
    profiles = generate_profiles(num_profiles)

    # 2. Generate Normal Transactions
    normal_txns = generate_normal_transactions(profiles, events_per_profile=(5, 15))

    # 3. Generate ATO Fraud
    # We want ATO to make up ~60% of the fraud cases
    ato_cases = int(target_fraud_txns * 0.6)
    ato_txns = generate_ato_fraud(profiles, num_cases=ato_cases)

    # 4. Generate Synthetic Identity Rings
    # We want Synth Rings to make up the remaining ~40% of fraud txns
    # Each ring uses ~12 txns total (3 accounts * 4 txns)
    synth_target_txns = target_fraud_txns - len(ato_txns)
    num_rings = max(1, int(synth_target_txns / 12))
    synth_txns = generate_synthetic_identity_fraud(
        num_rings=num_rings, accounts_per_ring=(3, 5), txns_per_account=(2, 4)
    )

    # 5. Combine and shuffle
    all_txns = normal_txns + ato_txns + synth_txns
    random.shuffle(all_txns)

    df = pd.DataFrame(all_txns)

    # 6. Sort chronologically for realistic time-series structure
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Export to CSV
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "events.csv"
    df.to_csv(csv_path, index=False)

    # Print summary statistics
    print("\n" + "=" * 50)
    print("DATASET GENERATION COMPLETE")
    print("=" * 50)
    print(f"File saved:           {csv_path}")
    print(f"Total events:         {len(df):,}")
    print(f"Unique accounts:      {df['account_id'].nunique():,}")
    
    fraud_df = df[df["is_fraud"] == 1]
    print(f"\nLegitimate events:    {len(df) - len(fraud_df):,} ({(len(df) - len(fraud_df)) / len(df):.2%})")
    print(f"Fraud events:         {len(fraud_df):,} ({len(fraud_df) / len(df):.2%})")
    
    ato = fraud_df[fraud_df["fraud_type"] == "account_takeover"]
    synth = fraud_df[fraud_df["fraud_type"] == "synthetic_identity"]
    print(f"  - ATO cases:          {len(ato):,}")
    print(f"  - Synthetic Identity: {len(synth):,}")
    print("=" * 50 + "\n")

    return df

# ──────────────────────────────────────────────
# Step 8: Generate Historical Cases for RAG
# ──────────────────────────────────────────────

def generate_historical_cases(
    num_cases: int = 200,
    output_dir: str = "data/synthetic"
):
    """
    Generate plain-text investigator reports of past fraud cases.
    
    This becomes the 'memory' for the LangGraph AI agent in Phase 4. When a new
    transaction is flagged, the agent searches ChromaDB for these historical
    documents to understand how humans resolved similar issues in the past.
    """
    cases = []
    
    fraud_types = [
        "account_takeover",
        "synthetic_identity",
        "card_testing",         # Extra type for broader knowledge base
        "first_party_fraud"     # Extra type for broader knowledge base
    ]
    
    # Pre-written templates so the language looks like a real human analyst wrote it
    templates = {
        "account_takeover": {
            "summary": "Account {acc} exhibited rapid transaction velocity ({vel} txns/hr) from an IP in {country} while the account is registered in US. Device ID was previously unseen.",
            "evidence": ["IP country mismatch", "New device detected", "Velocity significantly above baseline"],
            "action": "Block device. Freeze account. Customer contacted and password reset forced."
        },
        "synthetic_identity": {
            "summary": "Account {acc} was created {age} days ago. Investigation revealed the device ID is shared with {shared} other recently created accounts, indicating a synthetic identity ring.",
            "evidence": ["Device sharing across multiple accounts", "Low account age", "Coordinated transaction patterns"],
            "action": "Close account immediately. Blacklist device ID and IP address."
        },
        "card_testing": {
            "summary": "Account {acc} attempted {vel} micro-transactions ($1-3) in rapid succession. Multiple declined auths preceded a high-value transaction attempt.",
            "evidence": ["Micro-transaction pattern", "High decline rate", "Velocity spike"],
            "action": "Decline transaction. Send automated SMS verification to account holder."
        },
        "first_party_fraud": {
            "summary": "Account {acc} disputed a perfectly routed transaction shipped to their verified home address. Account has a history of similar disputes.",
            "evidence": ["Shipping address matches billing", "Historical dispute abuse", "No IP or device anomalies"],
            "action": "Deny dispute claim. Flag account for 'Friendly Fraud' review."
        }
    }

    print(f"Generating {num_cases} historical case investigation documents...")

    for i in range(num_cases):
        fraud_type = random.choice(fraud_types)
        template = templates[fraud_type]
        
        case_id = f"CASE-{random.randint(1000, 9999)}"
        acc_id = f"ACC-{random.randint(10000, 99999)}"
        
        # Fill placeholders with random but realistic numbers
        summary = template["summary"].format(
            acc=acc_id,
            vel=random.randint(4, 15),
            country=random.choice(FOREIGN_COUNTRIES),
            age=random.randint(1, 14),
            shared=random.randint(3, 8)
        )
        
        case_doc = {
            "case_id": case_id,
            "fraud_type": fraud_type,
            "summary": summary,
            "evidence": template["evidence"],
            "outcome": "confirmed_fraud",
            "recommended_action": template["action"]
        }
        cases.append(case_doc)
        
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "historical_cases.json"
    
    with open(json_path, "w") as f:
        json.dump(cases, f, indent=4)
        
    print(f"File saved:           {json_path}")
    print(f"Total cases:          {len(cases)}")
    return cases


# ──────────────────────────────────────────────
# Step 9: Wire It All Together (CLI)
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SentinelIQ Synthetic Data Generator")
    parser.add_argument("--events", type=int, default=10000, help="Number of transaction events to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.015, help="Percentage of events that are fraud (0.0 to 1.0)")
    parser.add_argument("--cases", type=int, default=200, help="Number of historical case reports to generate")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    
    args = parser.parse_args()
    
    print("\nStarting SentinelIQ Data Generation")
    print("=" * 50)
    
    # Run Step 7 (Transaction Dataset)
    build_dataset(
        target_events=args.events, 
        fraud_rate=args.fraud_rate, 
        output_dir=args.output
    )
    
    # Run Step 8 (Historical RAG Cases)
    generate_historical_cases(
        num_cases=args.cases, 
        output_dir=args.output
    )
    
    print("\nPhase 1 Data Generation is 100% complete! ✅\n")

if __name__ == "__main__":
    main()
