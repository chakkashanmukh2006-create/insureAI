"""
Insurance AI Intelligence System — Synthetic Dataset Generator
===============================================================
Generates two realistic CSV datasets:
  1. leads_dataset.csv   (1000 records)
  2. customers_dataset.csv (1000 records)

Targets are produced via logistic-probability functions so that
XGBoost (or any gradient-boosted model) can learn meaningful patterns.

Usage:
    python datasets/generate_datasets.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Output directory (same folder as this script)
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------
MALE_FIRST_NAMES = [
    "Rahul", "Amit", "Suresh", "Vikram", "Arjun", "Rohan", "Karan",
    "Aditya", "Sanjay", "Deepak", "Rajesh", "Nikhil", "Prashant",
    "Manish", "Varun", "Gaurav", "Ankur", "Ashish", "Sachin", "Vivek",
    "Harish", "Dinesh", "Mahesh", "Abhishek", "Ankit", "Kunal", "Neeraj",
    "Pankaj", "Ritesh", "Sandeep",
]

FEMALE_FIRST_NAMES = [
    "Priya", "Sneha", "Neha", "Anjali", "Pooja", "Swati", "Kavita",
    "Sunita", "Ritu", "Divya", "Meera", "Shalini", "Nisha", "Shruti",
    "Pallavi", "Aarti", "Komal", "Tanvi", "Isha", "Simran",
]

LAST_NAMES = [
    "Sharma", "Patel", "Gupta", "Singh", "Kumar", "Verma", "Joshi",
    "Mehta", "Shah", "Reddy", "Nair", "Iyer", "Pillai", "Mishra",
    "Tiwari", "Agarwal", "Malhotra", "Kapoor", "Bhatia", "Chopra",
    "Saxena", "Bansal", "Arora", "Sinha", "Chauhan", "Thakur", "Desai",
    "Kulkarni", "Patil", "Rao",
]

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Kochi",
    "Coimbatore", "Indore", "Bhopal",
]

OCCUPATIONS = [
    "Salaried", "Self-Employed", "Business", "Professional",
    "Government", "Student", "Retired",
]
OCCUPATION_WEIGHTS = [0.35, 0.15, 0.12, 0.13, 0.10, 0.08, 0.07]

LEAD_PRODUCTS = [
    "Term Life", "Health Insurance", "Motor Insurance",
    "Home Insurance", "Travel Insurance", "ULIP", "Endowment Plan",
]

CUSTOMER_PRODUCTS = [
    "Term Life", "Health Insurance", "Motor Insurance",
    "Home Insurance", "Travel Insurance", "ULIP", "Endowment Plan",
    "Critical Illness", "Personal Accident",
]

LEAD_SOURCES = [
    "Website", "Referral", "Social Media", "Email Campaign",
    "Google Ads", "Agent", "Walk-in",
]

# ---------------------------------------------------------------------------
# Income ranges by occupation (INR)
# ---------------------------------------------------------------------------
INCOME_RANGES = {
    "Salaried":      (300_000, 1_500_000),
    "Self-Employed":  (250_000, 2_000_000),
    "Business":       (500_000, 5_000_000),
    "Professional":   (400_000, 2_000_000),
    "Government":     (300_000, 1_200_000),
    "Student":        (200_000, 500_000),
    "Retired":        (200_000, 800_000),
}

# ---------------------------------------------------------------------------
# Premium ranges by policy type (INR)
# ---------------------------------------------------------------------------
PREMIUM_RANGES = {
    "Term Life":          (8_000, 50_000),
    "Health Insurance":   (10_000, 80_000),
    "Motor Insurance":    (5_000, 30_000),
    "Home Insurance":     (15_000, 100_000),
    "Travel Insurance":   (3_000, 15_000),
    "ULIP":               (30_000, 200_000),
    "Endowment Plan":     (20_000, 120_000),
    "Critical Illness":   (12_000, 60_000),
    "Personal Accident":  (4_000, 20_000),
}

# ---------------------------------------------------------------------------
# Feedback templates (40+ unique)
# ---------------------------------------------------------------------------
POSITIVE_FEEDBACK = [
    "Very satisfied with the claim process. Quick settlement and helpful staff.",
    "Excellent service. My agent is very responsive and helped me understand my policy.",
    "Great experience overall. The online portal is easy to use and renewal was seamless.",
    "Happy with the coverage. The premium is reasonable for the benefits provided.",
    "Claim was processed within 48 hours. Very impressed with the efficiency.",
    "Customer support resolved my query on the first call. Highly recommend.",
    "Smooth onboarding experience. The documentation was minimal and clear.",
    "Really appreciate the cashless hospital facility. It was stress-free.",
    "Agent is very knowledgeable. He explained all terms in simple language.",
    "Very professional service. I have already recommended this to my family.",
    "Online policy renewal took just 2 minutes. Very convenient.",
    "Good value for money. The add-ons are useful and reasonably priced.",
    "I had a minor accident and the motor claim was settled without hassle.",
    "Timely reminders for premium payment. Never missed a renewal because of this.",
    "The mobile app is well designed. I can track everything easily.",
]

NEGATIVE_FEEDBACK = [
    "Very disappointed with claim rejection. Filed three times still pending.",
    "Premium increased without notice. Support never picks up calls.",
    "Terrible experience with the cashless facility. Hospital said policy not valid.",
    "Claim settlement took over 6 months. Absolutely unacceptable service.",
    "Nobody responds to emails. I have been waiting for weeks for a callback.",
    "Hidden charges in the policy that were never disclosed by the agent.",
    "Support staff was rude and unhelpful. They kept transferring my call.",
    "Renewal process is unnecessarily complicated. The website keeps crashing.",
    "Policy terms changed at renewal without informing me. Very disappointed.",
    "I was mis-sold this policy. The returns are nothing like promised.",
    "Agent disappeared after selling the policy. No after-sales support at all.",
    "Filed a complaint 3 months ago. Still no resolution or update.",
    "Premium deducted twice from my account. No refund yet despite multiple calls.",
    "The worst customer service I have ever experienced. Will not renew.",
    "Waited 45 minutes on hold just to hear that my claim was rejected.",
]

NEUTRAL_FEEDBACK = [
    "Service is okay. Nothing special but gets the job done.",
    "Average experience. Some processes could be faster.",
    "Policy is decent for the price. No major complaints so far.",
    "The service is neither excellent nor bad. It is just fine.",
    "Haven't had to make a claim yet, so hard to judge fully.",
    "Renewal was straightforward but premium went up slightly.",
    "The agent is sometimes reachable, sometimes not. Mixed experience.",
    "Online portal works but the UI could be more intuitive.",
    "Coverage is standard. Nothing extraordinary about this plan.",
    "Adequate service. I wish the communication was more proactive.",
    "Documentation was a bit tedious but manageable overall.",
    "Claim process was slow but eventually resolved satisfactorily.",
]

NEGATIVE_KEYWORDS = [
    "disappointed", "rejected", "terrible", "unacceptable", "worst",
    "rude", "unhelpful", "complicated", "mis-sold", "disappeared",
    "no resolution", "no refund", "crashing", "hidden charges",
    "never picks up", "pending", "not valid", "no after-sales",
]


# ===================================================================
# Helper utilities
# ===================================================================

def _generate_name(gender: str) -> str:
    """Return a full name matching the given gender."""
    if gender == "Male":
        first = rng.choice(MALE_FIRST_NAMES)
    else:
        first = rng.choice(FEMALE_FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return f"{first} {last}"


def _generate_email(full_name: str, idx: int) -> str:
    """Generate a realistic email from a full name."""
    parts = full_name.lower().split()
    first, last = parts[0], parts[-1]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "rediffmail.com"]
    domain = rng.choice(domains)
    # Add a small numeric suffix to avoid collisions
    suffix = rng.choice(["", str(rng.integers(1, 99)), str(idx % 100)])
    return f"{first}.{last}{suffix}@{domain}"


def _generate_phone() -> str:
    """Generate a 10-digit Indian mobile number starting with 6/7/8/9."""
    first_digit = rng.choice([6, 7, 8, 9])
    remaining = "".join([str(rng.integers(0, 10)) for _ in range(9)])
    return f"{first_digit}{remaining}"


def _right_skewed_income(low: int, high: int) -> int:
    """Draw from a right-skewed distribution (log-normal mapped to range)."""
    # Log-normal produces a natural right skew
    raw = rng.lognormal(mean=0, sigma=0.7)
    # Normalize to [0, 1] — clip at practical bounds
    normalized = min(raw / 5.0, 1.0)
    value = low + normalized * (high - low)
    return int(round(value / 1000) * 1000)  # Round to nearest 1000


def _sigmoid(x: float) -> float:
    """Standard sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


# ===================================================================
# LEADS DATASET
# ===================================================================

def generate_leads(n: int = 1000, output_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic leads dataset with learnable conversion targets.

    The conversion_target is produced by a logistic probability function
    that combines multiple features, ensuring that a gradient-boosted
    model can discover meaningful feature–target relationships.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "leads_dataset.csv")

    records = []

    for i in range(n):
        lead_id = f"LEAD_{i + 1:04d}"

        # Gender (60/40 Male/Female)
        gender = "Male" if rng.random() < 0.60 else "Female"

        # Name
        full_name = _generate_name(gender)

        # Age — normal distribution centred at 35, clipped to [22, 65]
        age = int(np.clip(rng.normal(35, 8), 22, 65))

        # Occupation
        occupation = rng.choice(OCCUPATIONS, p=OCCUPATION_WEIGHTS)

        # Annual income — right-skewed within occupation band
        inc_low, inc_high = INCOME_RANGES[occupation]
        annual_income = _right_skewed_income(inc_low, inc_high)

        # City
        city = rng.choice(CITIES)

        # Existing policy (30% True)
        existing_policy = bool(rng.random() < 0.30)

        # Product interested
        product_interested = rng.choice(LEAD_PRODUCTS)

        # Engagement metrics (Poisson distributions)
        website_visits = int(np.clip(rng.poisson(3), 0, 20))
        email_opens = int(np.clip(rng.poisson(2), 0, 15))
        calls_answered = int(np.clip(rng.poisson(1), 0, 10))

        # Form submitted (20% True)
        form_submitted = bool(rng.random() < 0.20)

        # Last interaction days (uniform 1–90)
        last_interaction_days = int(rng.integers(1, 91))

        # Lead source
        lead_source = rng.choice(LEAD_SOURCES)

        # Contact details
        email = _generate_email(full_name, i)
        contact_number = _generate_phone()

        # ----- Conversion probability (logistic model) ----- #
        score = -2.0  # Bias — keeps base rate low

        # Income effect
        if annual_income > 1_500_000:
            score += 0.8
        elif annual_income > 800_000:
            score += 0.5

        # Existing policy
        if existing_policy:
            score += 0.6

        # Website visits
        if website_visits > 8:
            score += 0.7
        elif website_visits > 5:
            score += 0.4

        # Email opens
        if email_opens > 5:
            score += 0.5
        elif email_opens > 3:
            score += 0.3

        # Calls answered
        if calls_answered > 3:
            score += 0.6
        elif calls_answered > 2:
            score += 0.3

        # Form submitted
        if form_submitted:
            score += 0.7

        # Recency
        if last_interaction_days < 7:
            score += 0.5
        elif last_interaction_days < 15:
            score += 0.3
        elif last_interaction_days > 60:
            score -= 0.4

        # Lead source
        if lead_source in ["Referral", "Agent", "Walk-in"]:
            score += 0.5
        elif lead_source == "Google Ads":
            score += 0.1

        # Product interest
        if product_interested in ["Term Life", "Health Insurance"]:
            score += 0.4
        elif product_interested in ["ULIP", "Endowment Plan"]:
            score += 0.1

        # Age effect (prime earning years slightly more likely)
        if 28 <= age <= 45:
            score += 0.2

        # Add small noise to prevent perfect separability
        score += rng.normal(0, 0.3)

        probability = _sigmoid(score)
        conversion_target = int(rng.random() < probability)

        records.append({
            "lead_id": lead_id,
            "full_name": full_name,
            "age": age,
            "gender": gender,
            "occupation": occupation,
            "annual_income": annual_income,
            "city": city,
            "existing_policy": existing_policy,
            "product_interested": product_interested,
            "website_visits": website_visits,
            "email_opens": email_opens,
            "calls_answered": calls_answered,
            "form_submitted": form_submitted,
            "last_interaction_days": last_interaction_days,
            "lead_source": lead_source,
            "email": email,
            "contact_number": contact_number,
            "conversion_target": conversion_target,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    conv_rate = df["conversion_target"].mean() * 100
    print(f"✅ Leads dataset generated: {output_path}")
    print(f"   Records : {len(df)}")
    print(f"   Conversion rate: {conv_rate:.1f}%")

    return df


# ===================================================================
# CUSTOMERS DATASET
# ===================================================================

def generate_customers(n: int = 1000, output_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic customers dataset with learnable churn targets.

    The churn_target is produced by a logistic probability function so
    that tree-based models can discover realistic churn drivers.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "customers_dataset.csv")

    all_feedback = POSITIVE_FEEDBACK + NEGATIVE_FEEDBACK + NEUTRAL_FEEDBACK

    records = []

    for i in range(n):
        customer_id = f"CUST_{i + 1:04d}"

        # Gender (for name generation)
        gender = "Male" if rng.random() < 0.55 else "Female"
        name = _generate_name(gender)

        # Age — normal distribution centred at 40, clipped to [25, 70]
        age = int(np.clip(rng.normal(40, 10), 25, 70))

        # Policy type
        policy_type = rng.choice(CUSTOMER_PRODUCTS)

        # Premium amount — within policy-type band, slight right skew
        prem_low, prem_high = PREMIUM_RANGES[policy_type]
        premium_amount = _right_skewed_income(prem_low, prem_high)

        # Renewal history (0–10, geometric-ish)
        renewal_history = int(np.clip(rng.poisson(3), 0, 10))

        # Claim history (0–5)
        claim_history = int(np.clip(rng.poisson(0.8), 0, 5))

        # Complaint count (0–10, right-skewed)
        complaint_count = int(np.clip(rng.poisson(1.5), 0, 10))

        # Support tickets (0–8)
        support_tickets = int(np.clip(rng.poisson(1.2), 0, 8))

        # Feedback — bias towards positive for low-complaint customers,
        # negative for high-complaint customers
        if complaint_count >= 4:
            feedback_pool = NEGATIVE_FEEDBACK + NEUTRAL_FEEDBACK[:3]
        elif complaint_count <= 1:
            feedback_pool = POSITIVE_FEEDBACK + NEUTRAL_FEEDBACK[:5]
        else:
            feedback_pool = all_feedback
        feedback = rng.choice(feedback_pool)

        # Contact details
        email = _generate_email(name, i + 2000)
        contact_number = _generate_phone()

        # ----- Churn probability (logistic model) ----- #
        score = -1.8  # Bias — keeps base rate around 25-30%

        # Complaint count
        if complaint_count > 5:
            score += 1.0
        elif complaint_count > 3:
            score += 0.7

        # Support tickets
        if support_tickets > 5:
            score += 0.7
        elif support_tickets > 4:
            score += 0.4

        # Claims combined with complaints (frustrated claimants)
        if claim_history > 2 and complaint_count > 2:
            score += 0.8
        elif claim_history > 2:
            score += 0.3

        # Low renewal history → new customer, more likely to churn
        if renewal_history < 1:
            score += 0.7
        elif renewal_history < 2:
            score += 0.4

        # High tenure → sticky customer
        if renewal_history >= 5:
            score -= 0.5

        # Premium outlier — very high premium relative to policy type median
        prem_mid = (prem_low + prem_high) / 2
        if premium_amount > prem_mid * 1.4:
            score += 0.4

        # Feedback sentiment — simple keyword check
        feedback_lower = feedback.lower()
        neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in feedback_lower)
        if neg_hits >= 3:
            score += 0.8
        elif neg_hits >= 1:
            score += 0.4

        # Age effect — very young or very old slightly more likely to churn
        if age < 28 or age > 60:
            score += 0.2

        # Add small noise
        score += rng.normal(0, 0.3)

        probability = _sigmoid(score)
        churn_target = int(rng.random() < probability)

        records.append({
            "customer_id": customer_id,
            "name": name,
            "age": age,
            "policy_type": policy_type,
            "premium_amount": premium_amount,
            "renewal_history": renewal_history,
            "claim_history": claim_history,
            "complaint_count": complaint_count,
            "support_tickets": support_tickets,
            "feedback": feedback,
            "email": email,
            "contact_number": contact_number,
            "churn_target": churn_target,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    churn_rate = df["churn_target"].mean() * 100
    print(f"✅ Customers dataset generated: {output_path}")
    print(f"   Records : {len(df)}")
    print(f"   Churn rate: {churn_rate:.1f}%")

    return df


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Insurance AI — Synthetic Dataset Generator")
    print("=" * 60)
    print()

    generate_leads()
    print()
    generate_customers()

    print()
    print("✅ All datasets generated successfully!")
    print(f"   Output directory: {OUTPUT_DIR}")
