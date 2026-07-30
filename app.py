"""
Beginner Tax Desk & CA Assist: Small-business & Individual Tax Preparation Platform.

Calculates income tax strictly based on current Indian Income Tax rules for
Financial Year 2025-26 / Assessment Year 2026-27 under both the New Tax Regime
(u/s 115BAC) and the Old Tax Regime, featuring AI tools for Chartered Accountants.
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from html import escape
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Configuration & Constants (FY 2025-26 / AY 2026-27)
# =============================================================================

APP_TITLE = "Beginner Tax Desk & CA Assist"
APP_SUBTITLE = "Smart ITR preparation & CA AI workbench for FY 2025-26 (AY 2026-27)"
ASSESSMENT_YEAR = "AY 2026-27"
FINANCIAL_YEAR = "FY 2025-26"
FY_START_YEAR = "2025"

OFFICIAL_SOURCES = {
    "Income Tax Portal": "https://www.incometax.gov.in",
    "ITR-4 FAQ & Guidance": "https://www.incometax.gov.in/iec/foportal/help/e-filing-itr4-form-sugam-faq",
    "Business & Profession ITR Guide": "https://www.incometax.gov.in/iec/foportal/help/individual-business-profession",
    "ITR Downloads": "https://www.incometax.gov.in/iec/foportal/downloads/income-tax-returns",
}

# New Tax Regime Slabs (FY 2025-26 u/s 115BAC)
NEW_REGIME_SLABS: List[Tuple[float, float, float]] = [
    (0, 400_000, 0.00),
    (400_000, 800_000, 0.05),
    (800_000, 1_200_000, 0.10),
    (1_200_000, 1_600_000, 0.15),
    (1_600_000, 2_000_000, 0.20),
    (2_000_000, 2_400_000, 0.25),
    (2_400_000, np.inf, 0.30),
]

# Old Tax Regime Slabs (Below 60 Years)
OLD_REGIME_SLABS: List[Tuple[float, float, float]] = [
    (0, 250_000, 0.00),
    (250_000, 500_000, 0.05),
    (500_000, 1_000_000, 0.20),
    (1_000_000, np.inf, 0.30),
]

CATEGORY_RULES = {
    "Sales / Receipts": ["sale", "receipt", "upi cr", "neft cr", "credit", "invoice", "received", "payment in"],
    "Purchases": ["purchase", "supplier", "inventory", "stock", "raw material", "vendor"],
    "Rent": ["rent", "lease"],
    "Salary / Labour": ["salary", "wages", "labour", "payroll", "staff", "stipend"],
    "Travel": ["fuel", "petrol", "diesel", "travel", "cab", "hotel", "flight", "uber", "ola"],
    "Utilities": ["electricity", "water", "internet", "phone", "mobile", "broadband", "bescom"],
    "Marketing": ["ads", "advertising", "marketing", "meta", "google", "facebook"],
    "Bank / Finance": ["bank charge", "interest", "loan", "emi", "processing fee"],
    "Tax Payments": ["tds", "advance tax", "gst", "income tax", "challan"],
}

REQUIRED_CHECKLIST = [
    "PAN and Aadhaar linking status",
    "Bank account details & active IFSCs",
    "Form 16 (Part A & Part B) / Salary slips",
    "Annual Information Statement (AIS) / TIS verification",
    "Form 26AS tax credit reconciliation",
    "Bank statements for full FY 2025-26",
    "Proof of Chapter VI-A deductions (80C, 80D, 80CCD)",
    "HRA rent receipts, lease agreement & landlord PAN",
    "Sales invoices / GST returns reconciliation (for business)",
    "Advance tax and TDS payment challans",
]


@dataclass(frozen=True)
class TaxEstimate:
    gross_income: float
    deductions_applied: float
    taxable_income: float
    gross_tax: float
    rebate_87a: float
    cess: float
    net_tax: float
    balance_payable: float
    effective_rate: float
    regime_name: str


# =============================================================================
# Formatters & Helpers
# =============================================================================

def money(value: float) -> str:
    """Format Indian Rupee values."""
    try:
        return f"₹{value:,.0f}"
    except Exception:
        return "₹0"


def normalize_column_name(column: object) -> str:
    """Standardize column names to snake_case."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(column).strip().lower())
    return cleaned.strip("_")


def infer_category(description: str, amount: float) -> str:
    """Auto-categorize transactions using keyword matching."""
    text = str(description).lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in text for keyword in keywords):
            return category
    if amount > 0:
        return "Sales / Receipts"
    return "Other Expenses"


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel uploaded file."""
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return pd.DataFrame()
    except Exception as exc:
        st.warning(f"Could not read {uploaded_file.name}. Please verify format.")
        return pd.DataFrame()


def extract_pdf_preview(uploaded_file) -> str:
    """Extract a short preview from PDF files."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages[:2]:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages).strip()
        return text[:1_200] if text else "PDF uploaded. No selectable text found."
    except ModuleNotFoundError:
        return "PDF uploaded. Install pypdf to enable text preview."
    except Exception:
        return "PDF uploaded. Text preview unavailable."


def extract_pdf_full_text(uploaded_file, max_pages: int = 10, max_chars: int = 10_000) -> str:
    """Extract readable text from uploaded PDFs."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages).strip()
        return text[:max_chars]
    except Exception:
        return ""


def standardize_transactions(raw_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge and clean transaction tables."""
    try:
        frames = []
        for frame in raw_frames:
            if frame.empty:
                continue

            df = frame.copy()
            df.columns = [normalize_column_name(column) for column in df.columns]

            date_col = next((c for c in df.columns if c in {"date", "txn_date", "transaction_date"}), None)
            desc_col = next((c for c in df.columns if c in {"description", "particulars", "narration", "details"}), None)
            amount_col = next((c for c in df.columns if c in {"amount", "value", "transaction_amount"}), None)
            debit_col = next((c for c in df.columns if c in {"debit", "withdrawal", "withdrawals"}), None)
            credit_col = next((c for c in df.columns if c in {"credit", "deposit", "deposits"}), None)

            clean = pd.DataFrame(index=df.index)
            clean["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            clean["description"] = df[desc_col].astype(str) if desc_col else "Transaction"

            if amount_col:
                clean["amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
            elif debit_col or credit_col:
                debit = pd.to_numeric(df[debit_col], errors="coerce").fillna(0.0) if debit_col else 0.0
                credit = pd.to_numeric(df[credit_col], errors="coerce").fillna(0.0) if credit_col else 0.0
                clean["amount"] = credit - debit
            else:
                clean["amount"] = 0.0

            clean["type"] = np.where(clean["amount"] >= 0, "Income", "Expense")
            clean["category"] = [infer_category(d, a) for d, a in zip(clean["description"], clean["amount"])]
            clean["absolute_amount"] = clean["amount"].abs()
            frames.append(clean)

        if not frames:
            return make_sample_transactions()

        transactions = pd.concat(frames, ignore_index=True)
        transactions["date"] = transactions["date"].fillna(pd.Timestamp(f"{FY_START_YEAR}-04-01"))
        return transactions.sort_values("date").reset_index(drop=True)
    except Exception:
        return make_sample_transactions()


@st.cache_data(show_spinner=False)
def make_sample_transactions() -> pd.DataFrame:
    """Create sample business ledger for initial preview."""
    try:
        rng = np.random.default_rng(42)
        months = pd.date_range("2025-04-01", periods=12, freq="MS")
        rows = []
        for month in months:
            sales = rng.integers(110_000, 190_000)
            rows.extend([
                (month + pd.Timedelta(days=2), "Client UPI receipt", float(sales)),
                (month + pd.Timedelta(days=5), "Raw material purchase", -float(sales * 0.35)),
                (month + pd.Timedelta(days=10), "Commercial shop rent", -22_000.0),
                (month + pd.Timedelta(days=15), "Electricity and high-speed internet", -7_500.0),
                (month + pd.Timedelta(days=22), "Digital marketing & ads", -5_000.0),
            ])

        sample = pd.DataFrame(rows, columns=["date", "description", "amount"])
        sample["type"] = np.where(sample["amount"] >= 0, "Income", "Expense")
        sample["category"] = [infer_category(d, a) for d, a in zip(sample["description"], sample["amount"])]
        sample["absolute_amount"] = sample["amount"].abs()
        return sample
    except Exception:
        return pd.DataFrame(columns=["date", "description", "amount", "type", "category", "absolute_amount"])


# =============================================================================
# Legal Income Tax Calculation Engine (FY 2025-26 / AY 2026-27)
# =============================================================================

def calculate_new_regime_tax(gross_income: float, is_salaried: bool, prepaid_tax: float) -> TaxEstimate:
    """Calculate tax under NEW TAX REGIME (u/s 115BAC) for FY 2025-26 / AY 2026-27."""
    std_deduction = 75_000.0 if is_salaried else 0.0
    taxable_income = max(gross_income - std_deduction, 0.0)

    gross_tax = 0.0
    for lower, upper, rate in NEW_REGIME_SLABS:
        if taxable_income > lower:
            slice_amt = min(taxable_income, upper) - lower
            gross_tax += slice_amt * rate

    # Rebate u/s 87A: Income up to ₹12 Lakh gets rebate up to ₹60,000 (tax free)
    if taxable_income <= 1_200_000:
        rebate = min(gross_tax, 60_000.0)
    elif taxable_income > 1_200_000:
        # Marginal relief for income slightly exceeding ₹12,00,000
        excess_income = taxable_income - 1_200_000
        rebate = (gross_tax - excess_income) if gross_tax > excess_income else 0.0
    else:
        rebate = 0.0

    tax_after_rebate = max(gross_tax - rebate, 0.0)
    cess = tax_after_rebate * 0.04
    net_tax = tax_after_rebate + cess
    balance = max(net_tax - max(prepaid_tax, 0.0), 0.0)
    eff_rate = (net_tax / gross_income) if gross_income > 0 else 0.0

    return TaxEstimate(
        gross_income=gross_income,
        deductions_applied=std_deduction,
        taxable_income=taxable_income,
        gross_tax=gross_tax,
        rebate_87a=rebate,
        cess=cess,
        net_tax=net_tax,
        balance_payable=balance,
        effective_rate=eff_rate,
        regime_name="New Tax Regime (u/s 115BAC)",
    )


def calculate_old_regime_tax(
    gross_income: float,
    is_salaried: bool,
    sec_80c: float,
    sec_80d: float,
    sec_80ccd_1b: float,
    hra_exemption: float,
    other_deductions: float,
    prepaid_tax: float,
) -> TaxEstimate:
    """Calculate tax under OLD TAX REGIME for FY 2025-26 / AY 2026-27."""
    std_deduction = 50_000.0 if is_salaried else 0.0
    total_80c = min(max(sec_80c, 0.0), 150_000.0)
    total_80ccd = min(max(sec_80ccd_1b, 0.0), 50_000.0)
    total_80d = min(max(sec_80d, 0.0), 100_000.0)

    total_deductions = std_deduction + total_80c + total_80d + total_80ccd + max(hra_exemption, 0.0) + max(other_deductions, 0.0)
    taxable_income = max(gross_income - total_deductions, 0.0)

    gross_tax = 0.0
    for lower, upper, rate in OLD_REGIME_SLABS:
        if taxable_income > lower:
            slice_amt = min(taxable_income, upper) - lower
            gross_tax += slice_amt * rate

    # Rebate u/s 87A under Old Regime: Income up to ₹5 Lakh gets rebate up to ₹12,500
    if taxable_income <= 500_000:
        rebate = min(gross_tax, 12_500.0)
    else:
        rebate = 0.0

    tax_after_rebate = max(gross_tax - rebate, 0.0)
    cess = tax_after_rebate * 0.04
    net_tax = tax_after_rebate + cess
    balance = max(net_tax - max(prepaid_tax, 0.0), 0.0)
    eff_rate = (net_tax / gross_income) if gross_income > 0 else 0.0

    return TaxEstimate(
        gross_income=gross_income,
        deductions_applied=total_deductions,
        taxable_income=taxable_income,
        gross_tax=gross_tax,
        rebate_87a=rebate,
        cess=cess,
        net_tax=net_tax,
        balance_payable=balance,
        effective_rate=eff_rate,
        regime_name="Old Tax Regime",
    )


def estimate_presumptive_income(
    digital_receipts: float,
    cash_receipts: float,
    professional_receipts: float,
    mode: str,
) -> Tuple[float, List[str]]:
    """Calculate 44AD / 44ADA Presumptive Income."""
    notes = []
    business_turnover = digital_receipts + cash_receipts

    if mode == "Business 44AD":
        income = (digital_receipts * 0.06) + (cash_receipts * 0.08)
        if business_turnover > 30_000_000 and cash_receipts <= business_turnover * 0.05:
            notes.append("Turnover exceeds Section 44AD enhanced threshold of ₹3 Crore.")
        elif business_turnover > 20_000_000 and cash_receipts > business_turnover * 0.05:
            notes.append("Cash receipts exceed 5%; standard 44AD limit is ₹2 Crore.")
        return income, notes

    if mode == "Profession 44ADA":
        income = professional_receipts * 0.50
        if professional_receipts > 7_500_000:
            notes.append("Professional receipts exceed Section 44ADA limit of ₹75 Lakh.")
        return income, notes

    return 0.0, notes


# =============================================================================
# Google Gemini 2.5 Flash Free AI Helpers (`google-genai` SDK)
# =============================================================================

def get_gemini_client():
    """Initialize free Gemini AI client."""
    api_key = st.secrets.get("GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ModuleNotFoundError:
        return None


def ai_parse_form16_or_document(doc_text: str) -> Tuple[Optional[Dict[str, float]], str]:
    """Parse Form 16 / Salary Slip / Document using Gemini 2.5 Flash."""
    if not doc_text.strip():
        return None, "No readable text found in uploaded document."

    client = get_gemini_client()
    if client is None:
        return None, "GEMINI_API_KEY is not configured in Streamlit Secrets."

    try:
        from google.genai import types

        prompt = f"""
        You are an Indian Income Tax document extractor. Parse this text from a Form 16 / Salary Slip / AIS.
        Return ONLY a JSON object with these exact numeric keys:
        - "gross_salary": gross salary or income (numeric)
        - "standard_deduction": standard deduction mentioned if any (numeric)
        - "hra_exemption": HRA exemption u/s 10(13A) if any (numeric)
        - "sec_80c": Section 80C deductions (PPF, EPF, ELSS, LIC, tuition fee) (numeric)
        - "sec_80d": Section 80D health insurance (numeric)
        - "tds_deducted": Total TDS / Tax deducted at source (numeric)
        - "employer_name": name of company or employer (string)

        Document Text:
        {doc_text[:8000]}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        parsed = json.loads(response.text)
        data = {
            "gross_salary": float(parsed.get("gross_salary", 0) or 0),
            "standard_deduction": float(parsed.get("standard_deduction", 0) or 0),
            "hra_exemption": float(parsed.get("hra_exemption", 0) or 0),
            "sec_80c": float(parsed.get("sec_80c", 0) or 0),
            "sec_80d": float(parsed.get("sec_80d", 0) or 0),
            "tds_deducted": float(parsed.get("tds_deducted", 0) or 0),
            "employer_name": str(parsed.get("employer_name", "") or "Detected Employer"),
        }
        return data, "Extracted successfully via Gemini AI."
    except Exception as exc:
        return None, f"Document AI extraction error: {type(exc).__name__}"


def ai_recommend_regime(new_est: TaxEstimate, old_est: TaxEstimate, gross_salary: float) -> str:
    """Generate intelligent regime comparison advisory using Gemini."""
    client = get_gemini_client()
    diff = abs(new_est.net_tax - old_est.net_tax)
    better = "New Tax Regime" if new_est.net_tax < old_est.net_tax else "Old Tax Regime"

    if client is None:
        return f"**Recommendation:** **{better}** saves you {money(diff)} in net tax for FY 2025-26."

    try:
        prompt = f"""
        You are a senior Chartered Accountant in India reviewing a client's tax figures for FY 2025-26 (AY 2026-27).
        Gross Income: {money(gross_salary)}
        New Regime Tax Liability: {money(new_est.net_tax)} (Deductions: {money(new_est.deductions_applied)})
        Old Regime Tax Liability: {money(old_est.net_tax)} (Deductions: {money(old_est.deductions_applied)})

        Explain clearly in 3-4 bullet points:
        1. Which regime is better and the exact net savings ({money(diff)}).
        2. Key reason for the difference (e.g. 87A rebate limit of ₹12L in New Regime vs 80C/80D in Old).
        3. Practical action advice for the CA or client before filing ITR.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text or f"Recommend {better} with savings of {money(diff)}."
    except Exception:
        return f"**Recommendation:** **{better}** saves {money(diff)}."


def ai_audit_ledger(transactions: pd.DataFrame, query: str) -> str:
    """CA AI Audit Assistant to detect anomalies and cash ratios."""
    if transactions.empty:
        return "No transactions loaded to analyze."

    client = get_gemini_client()
    if client is None:
        return "Configure GEMINI_API_KEY in Streamlit Secrets to enable CA Audit Assistant."

    try:
        by_cat = transactions.groupby(["type", "category"])["absolute_amount"].sum().round(0).astype(int).to_dict()
        monthly = transactions.copy()
        monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").astype(str)
        m_summary = monthly.groupby(["month", "type"])["absolute_amount"].sum().round(0).astype(int).to_dict()

        summary_data = {
            "total_records": len(transactions),
            "category_totals": by_cat,
            "monthly_breakdown": m_summary,
        }

        prompt = f"""
        You are an expert CA tax auditor. Review this summarized client ledger data:
        {json.dumps(summary_data, indent=2)}

        CA's Audit Query: {query}

        Provide concise, high-value tax audit observations (3-5 sentences). Highlight any unusual spikes, cash transactions, or potential disallowance risks u/s 40A(3) or GST mismatches.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text or "No audit findings returned."
    except Exception as exc:
        return f"Audit Error: {type(exc).__name__}"


def ai_generate_client_memo(
    profile: Dict[str, object],
    new_est: TaxEstimate,
    old_est: TaxEstimate,
) -> str:
    """Generate a formal CA Client Advisory Memo."""
    client = get_gemini_client()
    if client is None:
        return "Gemini API Key required to generate formal memo."

    try:
        prompt = f"""
        Draft a brief, professional tax advisory memo from a Chartered Accountant to their client:
        Client Name: {profile.get('taxpayer_name', 'Valued Client')}
        Business/Entity: {profile.get('business_type', 'Individual')}
        Assessment Year: AY 2026-27 (FY 2025-26)

        Figures:
        - Gross Income: {money(new_est.gross_income)}
        - New Regime Tax: {money(new_est.net_tax)}
        - Old Regime Tax: {money(old_est.net_tax)}
        - Recommended Regime: {"New Regime" if new_est.net_tax <= old_est.net_tax else "Old Regime"}

        Include:
        - Formal greeting & subject line
        - Executive summary of tax liability
        - Clear recommendation & action items before filing deadline
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text or "Memo draft unavailable."
    except Exception as exc:
        return f"Memo Error: {type(exc).__name__}"


# =============================================================================
# UI Setup & Sidebar
# =============================================================================

def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}

        .stApp {
            background: linear-gradient(115deg, rgba(3, 7, 18, .96), rgba(8, 19, 36, .96)),
                        linear-gradient(135deg, #050816 0%, #071629 38%, #0b1020 62%, #10251f 100%);
            color: #eef7ff;
        }

        .hero {
            padding: 1.45rem;
            background: linear-gradient(145deg, rgba(15, 23, 42, .82), rgba(8, 19, 36, .70));
            border: 1px solid rgba(148, 163, 184, .22);
            border-radius: 10px;
            margin-bottom: 1rem;
        }

        .info-box, .success-box, .warn-box {
            padding: 0.85rem 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .info-box { background: rgba(59, 130, 246, 0.18); color: #dbeafe; }
        .success-box { background: rgba(34, 197, 94, 0.18); color: #dcfce7; }
        .warn-box { background: rgba(245, 158, 11, 0.18); color: #fef3c7; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, object]:
    with st.sidebar:
        st.title("🧾 Tax Profile & Setup")
        st.caption(f"Legal Rules for {FINANCIAL_YEAR} → {ASSESSMENT_YEAR}")

        taxpayer_name = st.text_input("Taxpayer / Business Name", value="Rajesh Kumar", key="sb_name")
        is_salaried = st.checkbox("Is Salaried Individual?", value=True)

        st.divider()
        st.subheader("1. AI Document Parser")
        st.caption("Upload Form 16, Salary Slip, or AIS (PDF):")
        doc_file = st.file_uploader("Form 16 / Document PDF", type=["pdf"], key="sb_doc")

        if doc_file is not None and st.button("Parse Document with Gemini AI", key="btn_parse_doc"):
            text = extract_pdf_full_text(doc_file)
            with st.spinner("Extracting tax figures..."):
                parsed, msg = ai_parse_form16_or_document(text)
            if parsed:
                st.session_state["in_gross_salary"] = parsed["gross_salary"]
                st.session_state["in_80c"] = parsed["sec_80c"]
                st.session_state["in_80d"] = parsed["sec_80d"]
                st.session_state["in_hra"] = parsed["hra_exemption"]
                st.session_state["in_prepaid_tax"] = parsed["tds_deducted"]
                st.success(f"{msg} Employer: {parsed['employer_name']}")
            else:
                st.warning(msg)

        st.divider()
        st.subheader("2. Income & Deductions")
        gross_salary = st.number_input(
            "Gross Annual Income / Salary (₹)", min_value=0.0, value=1250000.0, step=25000.0, key="in_gross_salary"
        )
        sec_80c = st.number_input("Section 80C Deductions (Max ₹1.5L)", min_value=0.0, value=150000.0, step=10000.0, key="in_80c")
        sec_80d = st.number_input("Section 80D Health Insurance (₹)", min_value=0.0, value=25000.0, step=5000.0, key="in_80d")
        sec_80ccd = st.number_input("Section 80CCD(1B) NPS (Max ₹50K)", min_value=0.0, value=50000.0, step=5000.0, key="in_80ccd")
        hra_exemption = st.number_input("HRA Exemption u/s 10(13A) (₹)", min_value=0.0, value=0.0, step=10000.0, key="in_hra")
        other_deductions = st.number_input("Other Deductions (80E, 80G, etc.)", min_value=0.0, value=0.0, step=5000.0, key="in_other_ded")

        prepaid_tax = st.number_input("TDS + Advance Tax Paid (₹)", min_value=0.0, value=45000.0, step=5000.0, key="in_prepaid_tax")

        st.divider()
        st.subheader("3. Business / Presumptive")
        presumptive_mode = st.selectbox("Presumptive Section", ["Business 44AD", "Profession 44ADA"])
        digital_receipts = st.number_input("Digital Receipts (₹)", min_value=0.0, value=0.0, step=50000.0)
        cash_receipts = st.number_input("Cash Receipts (₹)", min_value=0.0, value=0.0, step=10000.0)
        professional_receipts = st.number_input("Professional Receipts (₹)", min_value=0.0, value=0.0, step=50000.0)

    return {
        "taxpayer_name": taxpayer_name,
        "is_salaried": is_salaried,
        "gross_salary": gross_salary,
        "sec_80c": sec_80c,
        "sec_80d": sec_80d,
        "sec_80ccd": sec_80ccd,
        "hra_exemption": hra_exemption,
        "other_deductions": other_deductions,
        "prepaid_tax": prepaid_tax,
        "presumptive_mode": presumptive_mode,
        "digital_receipts": digital_receipts,
        "cash_receipts": cash_receipts,
        "professional_receipts": professional_receipts,
    }


# =============================================================================
# Visualizations
# =============================================================================

def build_regime_comparison_chart(new_est: TaxEstimate, old_est: TaxEstimate) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                name="New Regime (u/s 115BAC)",
                x=["Gross Income", "Deductions", "Taxable Income", "Net Tax Liability"],
                y=[new_est.gross_income, new_est.deductions_applied, new_est.taxable_income, new_est.net_tax],
                marker_color="#2dd4bf",
            ),
            go.Bar(
                name="Old Tax Regime",
                x=["Gross Income", "Deductions", "Taxable Income", "Net Tax Liability"],
                y=[old_est.gross_income, old_est.deductions_applied, old_est.taxable_income, old_est.net_tax],
                marker_color="#60a5fa",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#eaf7ff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# =============================================================================
# Main Application Flow
# =============================================================================

def main() -> None:
    configure_page()
    profile = render_sidebar()

    st.markdown(
        f"""
        <div class="hero">
            <h1 style="margin:0; font-size:2.2rem;">{APP_TITLE}</h1>
            <p style="margin-top:0.4rem; color:#b9c7d8;">{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "1. Document & Ledger Upload",
        "2. New vs Old Regime Analysis",
        "3. CA AI Workbench & Audit",
        "4. Client Checklist & Memo",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Document & Ledger Upload
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("1. Attach Financial Statements & Data")
        st.caption("Upload CSV / Excel bank statements, sales registers, or PDF reports:")

        uploads = st.file_uploader(
            "Drop bank statements, ledger spreadsheets, or PDF tax records",
            type=["csv", "xlsx", "xls", "pdf"],
            accept_multiple_files=True,
        )

        table_frames = []
        pdf_previews = []
        for uploaded in uploads or []:
            if uploaded.name.lower().endswith(".pdf"):
                pdf_previews.append(f"**{uploaded.name}**\n\n{extract_pdf_preview(uploaded)}")
            else:
                table_frames.append(read_uploaded_table(uploaded))

        transactions = standardize_transactions(table_frames)

        income = float(transactions.loc[transactions["type"] == "Income", "amount"].sum())
        expenses = float(transactions.loc[transactions["type"] == "Expense", "absolute_amount"].sum())
        book_profit = max(income - expenses, 0.0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Detected Business Income", money(income))
        m2.metric("Detected Expenses", money(expenses))
        m3.metric("Computed Net Book Profit", money(book_profit))

        with st.expander("View Processed Ledger Transactions Table"):
            st.dataframe(transactions, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 2: Legal Tax Calculation & Regime Comparison
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("2. Tax Calculation Engine (FY 2025-26 / AY 2026-27)")

        presumptive_inc, p_notes = estimate_presumptive_income(
            float(profile["digital_receipts"]),
            float(profile["cash_receipts"]),
            float(profile["professional_receipts"]),
            str(profile["presumptive_mode"]),
        )

        total_gross = float(profile["gross_salary"]) + presumptive_inc
        prepaid = float(profile["prepaid_tax"])

        # Execute Legal Tax Calculations
        new_regime_est = calculate_new_regime_tax(
            gross_income=total_gross,
            is_salaried=bool(profile["is_salaried"]),
            prepaid_tax=prepaid,
        )

        old_regime_est = calculate_old_regime_tax(
            gross_income=total_gross,
            is_salaried=bool(profile["is_salaried"]),
            sec_80c=float(profile["sec_80c"]),
            sec_80d=float(profile["sec_80d"]),
            sec_80ccd_1b=float(profile["sec_80ccd"]),
            hra_exemption=float(profile["hra_exemption"]),
            other_deductions=float(profile["other_deductions"]),
            prepaid_tax=prepaid,
        )

        for note in p_notes:
            st.markdown(f'<div class="warn-box">{note}</div>', unsafe_allow_html=True)

        # High-level Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Gross Income", money(total_gross))
        col2.metric("New Regime Net Tax", money(new_regime_est.net_tax), delta=f"Effective: {new_regime_est.effective_rate:.1%}")
        col3.metric("Old Regime Net Tax", money(old_regime_est.net_tax), delta=f"Effective: {old_regime_est.effective_rate:.1%}")
        
        tax_diff = abs(new_regime_est.net_tax - old_regime_est.net_tax)
        best_regime = "New Regime" if new_regime_est.net_tax <= old_regime_est.net_tax else "Old Regime"
        col4.metric("Recommended Option", best_regime, delta=f"Savings: {money(tax_diff)}")

        st.plotly_chart(build_regime_comparison_chart(new_regime_est, old_regime_est), use_container_width=True)

        # AI Regime Advisory Note
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### Gemini AI Tax Advisor Recommendation")
        st.markdown(ai_recommend_regime(new_regime_est, old_regime_est, total_gross))
        st.markdown("</div>", unsafe_allow_html=True)

        # Detailed Breakdown Table
        st.subheader("Detailed Line-by-Line Comparison")
        comp_df = pd.DataFrame(
            [
                ("Gross Annual Income", money(new_regime_est.gross_income), money(old_regime_est.gross_income)),
                ("Standard Deduction", money(new_regime_est.deductions_applied), money(50000.0 if profile["is_salaried"] else 0.0)),
                ("Chapter VI-A (80C, 80D, 80CCD, HRA)", "Not Allowed", money(old_regime_est.deductions_applied - (50000.0 if profile["is_salaried"] else 0.0))),
                ("Net Taxable Income", money(new_regime_est.taxable_income), money(old_regime_est.taxable_income)),
                ("Gross Slab Tax", money(new_regime_est.gross_tax), money(old_regime_est.gross_tax)),
                ("Section 87A Rebate", money(-new_regime_est.rebate_87a), money(-old_regime_est.rebate_87a)),
                ("Health & Education Cess (4%)", money(new_regime_est.cess), money(old_regime_est.cess)),
                ("Total Net Tax Liability", money(new_regime_est.net_tax), money(old_regime_est.net_tax)),
                ("TDS / Advance Tax Paid", money(-prepaid), money(-prepaid)),
                ("Final Balance Payable / (Refund)", money(new_regime_est.balance_payable), money(old_regime_est.balance_payable)),
            ],
            columns=["Tax Calculation Head", "New Tax Regime (u/s 115BAC)", "Old Tax Regime"],
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 3: CA AI Workbench & Audit Tools
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("3. CA AI Workbench & Audit Assistant")
        st.caption("Free AI tools to automate ledger auditing, compliance checks, and query explanations.")

        audit_query = st.text_input(
            "Ask CA Audit Assistant about this ledger data",
            placeholder="e.g., Any high cash withdrawals or disallowance risks u/s 40A(3)?",
            key="audit_query_input",
        )

        if st.button("Run AI Audit Inspection", key="btn_audit"):
            with st.spinner("Analyzing ledger patterns with Gemini 2.5 Flash..."):
                result = ai_audit_ledger(transactions, audit_query)
            st.markdown(f'<div class="info-box">{escape(result)}</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 4: Checklist & Client Advisory Memo
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("4. Filing Checklist & Client Handoff")

        checked_items = [item for item in REQUIRED_CHECKLIST if st.checkbox(item, key=f"chk_{item}")]
        st.progress(len(checked_items) / len(REQUIRED_CHECKLIST))

        st.divider()
        st.subheader("Generate CA Client Advisory Memo")
        if st.button("Draft Client Advisory Memo", key="btn_memo"):
            with st.spinner("Drafting memo with Gemini AI..."):
                memo = ai_generate_client_memo(profile, new_regime_est, old_regime_est)
            st.text_area("Generated Memo Draft", value=memo, height=280)

        packet_data = {
            "generated_on": date.today().isoformat(),
            "assessment_year": ASSESSMENT_YEAR,
            "financial_year": FINANCIAL_YEAR,
            "profile": profile,
            "new_regime_tax": new_regime_est.net_tax,
            "old_regime_tax": old_regime_est.net_tax,
            "checklist": checked_items,
        }

        st.download_button(
            "Download Handoff JSON for Records",
            data=json.dumps(packet_data, indent=2),
            file_name=f"tax_summary_{profile['taxpayer_name']}.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
