"""
Beginner Tax Desk: small-business tax and ITR preparation assistant.

This Streamlit app helps entry-level Indian businesses organize documents,
summarize income and expenses, estimate tax under the new regime for AY 2026-27,
and prepare a simple ITR filing checklist.
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
# Configuration
# =============================================================================

APP_TITLE = "Beginner Tax Desk"
APP_SUBTITLE = "Easy ITR preparation for entry-level businesses"
ASSESSMENT_YEAR = "AY 2026-27"
FINANCIAL_YEAR = "FY 2025-26"
FY_START_YEAR = re.search(r"\d{4}", FINANCIAL_YEAR).group()  # "2025"

OFFICIAL_SOURCES = {
    "ITR-4 FAQ": "https://www.incometax.gov.in/iec/foportal/help/e-filing-itr4-form-sugam-faq",
    "Business / Profession ITR Guidance": "https://www.incometax.gov.in/iec/foportal/help/individual-business-profession",
    "ITR Downloads": "https://www.incometax.gov.in/iec/foportal/downloads/income-tax-returns",
}

NEW_REGIME_SLABS: List[Tuple[float, float, float]] = [
    (0, 400_000, 0.00),
    (400_000, 800_000, 0.05),
    (800_000, 1_200_000, 0.10),
    (1_200_000, 1_600_000, 0.15),
    (1_600_000, 2_000_000, 0.20),
    (2_000_000, 2_400_000, 0.25),
    (2_400_000, np.inf, 0.30),
]

CATEGORY_RULES = {
    "Sales / Receipts": ["sale", "receipt", "upi cr", "neft cr", "credit", "invoice", "received"],
    "Purchases": ["purchase", "supplier", "inventory", "stock", "raw material"],
    "Rent": ["rent", "lease"],
    "Salary / Labour": ["salary", "wages", "labour", "payroll", "staff"],
    "Travel": ["fuel", "petrol", "diesel", "travel", "cab", "hotel"],
    "Utilities": ["electricity", "water", "internet", "phone", "mobile", "broadband"],
    "Marketing": ["ads", "advertising", "marketing", "meta", "google"],
    "Bank / Finance": ["bank charge", "interest", "loan", "emi", "processing fee"],
    "Tax Payments": ["tds", "advance tax", "gst", "income tax", "challan"],
}

REQUIRED_CHECKLIST = [
    "PAN and Aadhaar details",
    "Bank account details and IFSC",
    "Bank statements for the full financial year",
    "Sales invoices or receipt register",
    "Purchase and expense bills",
    "GST returns, if registered",
    "Form 26AS / AIS / TIS reconciliation",
    "TDS certificates and advance-tax challans",
    "Loan, rent, salary, and major expense proofs",
]


@dataclass(frozen=True)
class TaxEstimate:
    taxable_income: float
    gross_tax: float
    rebate_87a: float
    cess: float
    net_tax: float
    balance_payable: float
    effective_rate: float


# =============================================================================
# Data Loading and Cleanup
# =============================================================================

def money(value: float) -> str:
    """Format Indian Rupee values for readable dashboard display."""
    try:
        return f"₹{value:,.0f}"
    except Exception:
        return "₹0"


def normalize_column_name(column: object) -> str:
    """Convert uploaded statement headers into predictable snake-case names."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(column).strip().lower())
    return cleaned.strip("_")


def infer_category(description: str, amount: float) -> str:
    """Classify a transaction using simple beginner-friendly keyword rules."""
    text = str(description).lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in text for keyword in keywords):
            return category
    if amount > 0:
        return "Sales / Receipts"
    return "Other Expenses"


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel transaction data from the uploaded file."""
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return pd.DataFrame()
    except Exception as exc:
        st.warning(f"Could not read {uploaded_file.name}. Please check the file format.")
        st.exception(exc)
        return pd.DataFrame()


def extract_pdf_preview(uploaded_file) -> str:
    """Extract a short text preview from PDFs when pypdf is available."""
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
        return "PDF uploaded. Text preview could not be extracted."


def extract_pdf_full_text(uploaded_file, max_pages: int = 8, max_chars: int = 8_000) -> str:
    """Extract complete text from a PDF for structured data extraction (e.g. Form 16)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages).strip()
        return text[:max_chars]
    except ModuleNotFoundError:
        return ""
    except Exception:
        return ""


def standardize_transactions(raw_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge and standardize uploaded transaction tables."""
    try:
        frames = []
        for frame in raw_frames:
            if frame.empty:
                continue

            df = frame.copy()
            df.columns = [normalize_column_name(column) for column in df.columns]

            date_col = next((c for c in df.columns if c in {"date", "txn_date", "transaction_date"}), None)
            desc_col = next(
                (c for c in df.columns if c in {"description", "particulars", "narration", "details"}),
                None,
            )
            amount_col = next((c for c in df.columns if c in {"amount", "value", "transaction_amount"}), None)
            debit_col = next((c for c in df.columns if c in {"debit", "withdrawal", "withdrawals"}), None)
            credit_col = next((c for c in df.columns if c in {"credit", "deposit", "deposits"}), None)

            clean = pd.DataFrame(index=df.index)
            clean["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
            clean["description"] = df[desc_col].astype(str) if desc_col else "Uploaded transaction"

            if amount_col:
                clean["amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
            elif debit_col or credit_col:
                debit = pd.to_numeric(df[debit_col], errors="coerce").fillna(0.0) if debit_col else 0.0
                credit = pd.to_numeric(df[credit_col], errors="coerce").fillna(0.0) if credit_col else 0.0
                clean["amount"] = credit - debit
            else:
                clean["amount"] = 0.0

            clean["type"] = np.where(clean["amount"] >= 0, "Income", "Expense")
            clean["category"] = [
                infer_category(description, amount)
                for description, amount in zip(clean["description"], clean["amount"])
            ]
            clean["absolute_amount"] = clean["amount"].abs()
            frames.append(clean)

        if not frames:
            return make_sample_transactions()

        transactions = pd.concat(frames, ignore_index=True)
        transactions["date"] = transactions["date"].fillna(pd.Timestamp(f"{FY_START_YEAR}-04-01"))
        return transactions.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        st.error("Uploaded transactions could not be cleaned.")
        st.exception(exc)
        return make_sample_transactions()


@st.cache_data(show_spinner=False)
def make_sample_transactions() -> pd.DataFrame:
    """Create a sample ledger for first-time users."""
    try:
        rng = np.random.default_rng(7)
        months = pd.date_range("2025-04-01", periods=12, freq="MS")
        rows = []
        for month in months:
            sales = rng.integers(85_000, 165_000)
            rows.extend(
                [
                    (month + pd.Timedelta(days=3), "UPI sales receipts", float(sales)),
                    (month + pd.Timedelta(days=7), "Supplier inventory purchase", -float(sales * 0.32)),
                    (month + pd.Timedelta(days=12), "Shop rent", -18_000.0),
                    (month + pd.Timedelta(days=18), "Electricity and internet", -6_500.0),
                    (month + pd.Timedelta(days=24), "Local advertising", -4_500.0),
                ]
            )

        sample = pd.DataFrame(rows, columns=["date", "description", "amount"])
        sample["type"] = np.where(sample["amount"] >= 0, "Income", "Expense")
        sample["category"] = [
            infer_category(description, amount)
            for description, amount in zip(sample["description"], sample["amount"])
        ]
        sample["absolute_amount"] = sample["amount"].abs()
        return sample
    except Exception as exc:
        st.error("Sample transactions could not be generated.")
        st.exception(exc)
        return pd.DataFrame(columns=["date", "description", "amount", "type", "category", "absolute_amount"])


# =============================================================================
# Tax Engine
# =============================================================================

def slab_tax_new_regime(taxable_income: float) -> float:
    """Calculate Indian individual tax under the new regime slabs for AY 2026-27."""
    try:
        tax = 0.0
        income = max(float(taxable_income), 0.0)
        for lower, upper, rate in NEW_REGIME_SLABS:
            if income > lower:
                taxable_slice = min(income, upper) - lower
                tax += taxable_slice * rate
        return max(tax, 0.0)
    except Exception:
        return 0.0


def calculate_tax_estimate(
    taxable_income: float,
    tds_and_advance_tax: float,
    enable_87a_rebate: bool,
) -> TaxEstimate:
    """Estimate net tax payable after rebate, cess, and prepaid taxes."""
    try:
        gross_tax = slab_tax_new_regime(taxable_income)

        if enable_87a_rebate and taxable_income <= 1_200_000:
            rebate = min(gross_tax, 60_000.0)
        elif enable_87a_rebate and taxable_income > 1_200_000:
            excess_income = taxable_income - 1_200_000
            rebate = (gross_tax - excess_income) if gross_tax > excess_income else 0.0
        else:
            rebate = 0.0

        tax_after_rebate = max(gross_tax - rebate, 0.0)
        cess = tax_after_rebate * 0.04
        net_tax = tax_after_rebate + cess
        balance = max(net_tax - max(tds_and_advance_tax, 0.0), 0.0)
        effective_rate = (net_tax / taxable_income) if taxable_income > 0 else 0.0
        return TaxEstimate(taxable_income, gross_tax, rebate, cess, net_tax, balance, effective_rate)
    except Exception as exc:
        st.error("Tax estimate could not be calculated.")
        st.exception(exc)
        return TaxEstimate(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def estimate_presumptive_income(
    digital_receipts: float,
    cash_receipts: float,
    professional_receipts: float,
    mode: str,
) -> Tuple[float, List[str]]:
    """Estimate presumptive business/professional income and threshold warnings."""
    notes = []
    try:
        business_turnover = digital_receipts + cash_receipts
        if mode == "Business 44AD":
            income = (digital_receipts * 0.06) + (cash_receipts * 0.08)
            if business_turnover > 30_000_000 and cash_receipts <= business_turnover * 0.05:
                notes.append("Turnover is above the 44AD enhanced threshold of Rs. 3 crore.")
            elif business_turnover > 20_000_000 and cash_receipts > business_turnover * 0.05:
                notes.append("Cash receipts appear above 5%; standard 44AD threshold may be Rs. 2 crore.")
            return income, notes

        if mode == "Profession 44ADA":
            income = professional_receipts * 0.50
            if professional_receipts > 7_500_000:
                notes.append("Professional receipts are above the enhanced 44ADA threshold of Rs. 75 lakh.")
            return income, notes

        return 0.0, notes
    except Exception as exc:
        st.error("Presumptive income could not be estimated.")
        st.exception(exc)
        return 0.0, notes


# =============================================================================
# FREE GEMINI API INTEGRATION
# =============================================================================

def get_gemini_client():
    """Initializes Google Gemini client using free tier key from secrets/environment."""
    api_key = st.secrets.get("GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ModuleNotFoundError:
        return None


def extract_form16_data(pdf_text: str) -> Tuple[Optional[Dict[str, float]], str]:
    """Uses Gemini 2.5 Flash (Free) to extract Form 16 details."""
    if not pdf_text.strip():
        return None, "Could not read any text from this PDF. It may be a scanned image."

    client = get_gemini_client()
    if client is None:
        return None, (
            "Form 16 auto-fill isn't set up yet. Add GEMINI_API_KEY under "
            "Streamlit Cloud's App settings -> Secrets, and add \"google-genai\" to requirements.txt."
        )

    try:
        from google.genai import types

        prompt = f"""
        Extract key figures from this Indian Form 16 text.
        Return ONLY a JSON object with these exact keys:
        - "gross_salary": gross salary or income under head salaries (numeric)
        - "standard_deduction": standard deduction under u/s 16(ia) if mentioned, else 0 (numeric)
        - "tds_deducted": total tax deducted at source / TDS (numeric)
        - "other_deductions": total Chapter VI-A deductions (80C, 80D, etc.) (numeric)
        - "employer_name": name of the employer / company (string)

        Form 16 text:
        {pdf_text[:8000]}
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        parsed = json.loads(response.text)
        data = {
            "gross_salary": float(parsed.get("gross_salary", 0) or 0),
            "standard_deduction": float(parsed.get("standard_deduction", 0) or 0),
            "tds_deducted": float(parsed.get("tds_deducted", 0) or 0),
            "other_deductions": float(parsed.get("other_deductions", 0) or 0),
            "employer_name": str(parsed.get("employer_name", "") or ""),
        }
        return data, "Extracted successfully. Please review the figures below."
    except json.JSONDecodeError:
        return None, "The response could not be parsed as valid JSON figures. Please enter details manually."
    except Exception as exc:
        return None, f"Could not process Form 16 right now ({type(exc).__name__}). Please enter manually."


def summarize_ledger_for_ca(transactions: pd.DataFrame, question: str) -> str:
    """Let a CA ask questions about a larger ledger using Gemini 2.5 Flash."""
    question = (question or "").strip()
    if not question:
        return "Type a question about this ledger first, e.g. \"What looks unusual here?\""
    if len(question) > 400:
        return "Please shorten your question to a single specific topic."
    if transactions.empty:
        return "No transaction data is loaded yet."

    client = get_gemini_client()
    if client is None:
        return "Gemini API key is missing. Add GEMINI_API_KEY in Streamlit Secrets."

    try:
        by_category = (
            transactions.groupby(["type", "category"])["absolute_amount"].sum().round(0).astype(int).to_dict()
        )
        monthly = transactions.copy()
        monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").astype(str)
        monthly_totals = monthly.groupby(["month", "type"])["absolute_amount"].sum().round(0).astype(int).to_dict()

        summary_text = (
            f"Total transactions: {len(transactions)}\n"
            f"By type & category: {by_category}\n"
            f"By month & type: {monthly_totals}\n"
        )

        prompt = f"""
        You are a data assistant helping a Chartered Accountant review a summary of a business ledger.
        Answer the CA's question using ONLY the aggregated figures provided below.
        Point out patterns or anomalies. Keep answers under 5 sentences.

        Ledger Summary:
        {summary_text}

        CA's Question: {question}
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text or "No analysis generated."
    except Exception as exc:
        return f"Ledger analysis error: ({type(exc).__name__})"


COMMON_JARGON_QUESTIONS = [
    "What is the 87A rebate?",
    "What does presumptive income mean?",
    "Why is there a Health and Education cess?",
    "Should I file ITR-3 or ITR-4?",
]


def explain_tax_term(question: str, context: Dict[str, str]) -> str:
    """Explain a tax term in plain language grounded in the client's figures."""
    question = (question or "").strip()
    if not question:
        return "Type a question first, e.g. \"What does 87A rebate mean for me?\""
    if len(question) > 300:
        return "Please shorten your question."

    client = get_gemini_client()
    if client is None:
        return "Gemini API key not configured. Please add GEMINI_API_KEY to Secrets manager."

    try:
        context_text = "\n".join(f"- {label}: {value}" for label, value in context.items())
        prompt = f"""
        You explain Indian income tax terms simply to a small business client.
        Rely strictly on these on-screen figures without inventing new numbers:
        {context_text}

        Question: {question}
        Answer in 3-4 short, simple sentences.
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text or "No explanation returned."
    except Exception as exc:
        return f"Could not fetch explanation ({type(exc).__name__})."


# =============================================================================
# UI Styling
# =============================================================================

def configure_page() -> None:
    """Configure page and apply interface styles."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
        .stAppDeployButton {display: none;}
        a[href*="github.com"] {display: none !important;}
        .viewerBadge_container__1QSob, .viewerBadge_link__1S137, .styles_viewerBadge__1yB5_, .stAppToolbar {display: none !important;}

        .stApp {
            background: linear-gradient(115deg, rgba(3, 7, 18, .96), rgba(8, 19, 36, .96)),
                        linear-gradient(135deg, #050816 0%, #071629 38%, #0b1020 62%, #10251f 100%);
            color: #eef7ff;
        }

        .hero {
            padding: 1.45rem;
            background: linear-gradient(145deg, rgba(15, 23, 42, .76), rgba(8, 19, 36, .62));
            border: 1px solid rgba(148, 163, 184, .18);
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .step-card {
            border: 1px solid rgba(148, 163, 184, .18);
            border-radius: 8px;
            background: rgba(15, 23, 42, .76);
            padding: 1rem;
            margin-bottom: 0.5rem;
        }

        .info-box, .success-box, .warn-box {
            padding: 0.8rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .info-box { background: rgba(59, 130, 246, 0.2); }
        .success-box { background: rgba(34, 197, 94, 0.2); }
        .warn-box { background: rgba(245, 158, 11, 0.2); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Sidebar and Inputs
# =============================================================================

def render_sidebar() -> Dict[str, object]:
    """Collect taxpayer profile and tax inputs."""
    with st.sidebar:
        st.title("🧾 Tax Setup")
        st.caption(f"{FINANCIAL_YEAR} → {ASSESSMENT_YEAR}")

        taxpayer_name = st.text_input("Business or taxpayer name", value="My Small Business")
        business_type = st.selectbox(
            "What best describes you?",
            ["Small trader / shop", "Freelancer / professional", "Service business", "Other"],
        )
        itr_path = st.radio(
            "Which filing path do you want to prepare for?",
            ["ITR-4 Sugam: presumptive income", "ITR-3: regular books"],
            help="ITR-4 is used by eligible small taxpayers under 44AD/44ADA.",
        )

        st.divider()
        st.subheader("Form 16 auto-fill")
        st.caption("Salaried individuals: upload Form 16 to auto-fill salary and TDS figures below.")
        form16_file = st.file_uploader("Upload Form 16 (PDF)", type=["pdf"], key="form16_upload")

        if form16_file is not None and st.button("Extract from Form 16", key="form16_extract_btn"):
            pdf_text = extract_pdf_full_text(form16_file)
            with st.spinner("Reading Form 16 with Gemini AI..."):
                data, message = extract_form16_data(pdf_text)
            if data:
                st.session_state["in_other_income"] = max(data["gross_salary"] - data["standard_deduction"], 0.0)
                st.session_state["in_deductions"] = max(data["other_deductions"], 0.0)
                st.session_state["in_prepaid_tax"] = max(data["tds_deducted"], 0.0)
                st.success(
                    f"{message} Employer: {data['employer_name'] or 'Not detected'} | "
                    f"Gross salary: {money(data['gross_salary'])} | TDS: {money(data['tds_deducted'])}"
                )
            else:
                st.warning(message)

        st.divider()
        st.subheader("Quick Money Inputs")
        other_income = st.number_input(
            "Other income, if any", min_value=0.0, value=0.0, step=5_000.0, key="in_other_income"
        )
        deductions = st.number_input(
            "Basic deductions you want to track", min_value=0.0, value=0.0, step=5_000.0, key="in_deductions"
        )
        prepaid_tax = st.number_input(
            "TDS + advance tax already paid", min_value=0.0, value=0.0, step=5_000.0, key="in_prepaid_tax"
        )
        enable_rebate = st.checkbox("Apply 87A rebate check", value=True)

        st.divider()
        st.subheader("Presumptive helper")
        presumptive_mode = st.selectbox("Section helper", ["Business 44AD", "Profession 44ADA"])
        digital_receipts = st.number_input("Digital business receipts", min_value=0.0, value=0.0, step=10_000.0)
        cash_receipts = st.number_input("Cash business receipts", min_value=0.0, value=0.0, step=10_000.0)
        professional_receipts = st.number_input("Professional receipts", min_value=0.0, value=0.0, step=10_000.0)

    return {
        "taxpayer_name": taxpayer_name,
        "business_type": business_type,
        "itr_path": itr_path,
        "other_income": other_income,
        "deductions": deductions,
        "prepaid_tax": prepaid_tax,
        "enable_rebate": enable_rebate,
        "presumptive_mode": presumptive_mode,
        "digital_receipts": digital_receipts,
        "cash_receipts": cash_receipts,
        "professional_receipts": professional_receipts,
    }


# =============================================================================
# Charts and Download Helpers
# =============================================================================

def build_category_chart(transactions: pd.DataFrame) -> go.Figure:
    """Show expense categories as a donut chart."""
    expenses = transactions[transactions["type"] == "Expense"].copy()
    if expenses.empty:
        expenses = pd.DataFrame({"category": ["No expenses"], "absolute_amount": [1.0]})

    grouped = expenses.groupby("category", as_index=False)["absolute_amount"].sum()
    fig = px.pie(grouped, values="absolute_amount", names="category", hole=0.55)
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#eaf7ff"), showlegend=False)
    return fig


def build_monthly_chart(transactions: pd.DataFrame) -> go.Figure:
    """Show monthly income and expenses."""
    df = transactions.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly = df.pivot_table(index="month", columns="type", values="amount", aggfunc="sum", fill_value=0)
    monthly["Income"] = monthly.get("Income", 0)
    monthly["Expense"] = monthly.get("Expense", 0).abs()
    monthly = monthly.reset_index()

    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["Income"], name="Income", marker_color="#22c55e")
    fig.add_bar(x=monthly["month"], y=monthly["Expense"], name="Expenses", marker_color="#f59e0b")
    fig.update_layout(height=300, barmode="group", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#eaf7ff"))
    return fig


def build_tax_waterfall(estimate: TaxEstimate, prepaid_tax: float) -> go.Figure:
    """Visualize tax calculation."""
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Gross tax", "87A rebate", "Cess", "Prepaid tax", "Balance Payable"],
            y=[
                estimate.gross_tax,
                -estimate.rebate_87a,
                estimate.cess,
                -min(prepaid_tax, estimate.net_tax),
                estimate.balance_payable,
            ],
        )
    )
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#eaf7ff"))
    return fig


def make_download_packet(profile: Dict[str, object], transactions: pd.DataFrame, regular_profit: float, presumptive_income: float, estimate: TaxEstimate) -> str:
    """Create downloadable JSON packet."""
    packet = {
        "generated_on": date.today().isoformat(),
        "assessment_year": ASSESSMENT_YEAR,
        "profile": profile,
        "summary": {
            "regular_profit": float(regular_profit),
            "presumptive_income": float(presumptive_income),
            "taxable_income": float(estimate.taxable_income),
            "net_tax": float(estimate.net_tax),
            "balance_payable": float(estimate.balance_payable),
        },
        "checklist": REQUIRED_CHECKLIST,
    }
    return json.dumps(packet, indent=2)


# =============================================================================
# Main Workflow
# =============================================================================

def render_ca_assistant(profile: Dict[str, object], estimate: TaxEstimate, presumptive_income: float, regular_profit: float, transactions: pd.DataFrame) -> None:
    """Render Gemini AI Assistant tab for CAs and clients."""
    st.markdown("#### CA Assist: Plain-language Explainer")
    st.caption("Ask questions about tax concepts or figures computed on screen.")

    context = {
        "Filing path": str(profile.get("itr_path", "")),
        "Taxable income": money(estimate.taxable_income),
        "Gross tax": money(estimate.gross_tax),
        "87A rebate": money(estimate.rebate_87a),
        "Cess": money(estimate.cess),
        "Net tax": money(estimate.net_tax),
        "Prepaid tax": money(float(profile.get("prepaid_tax", 0.0))),
        "Balance payable": money(estimate.balance_payable),
    }

    quick_cols = st.columns(len(COMMON_JARGON_QUESTIONS))
    quick_pick = None
    for col, term in zip(quick_cols, COMMON_JARGON_QUESTIONS):
        with col:
            if st.button(term, key=f"quick_{term}", use_container_width=True):
                quick_pick = term

    typed_question = st.text_input("Or ask your own tax question", placeholder="e.g. Why is 87A rebate applied?", key="ca_assist_question")
    ask_clicked = st.button("Explain", key="ca_assist_ask")

    active_question = quick_pick or (typed_question if ask_clicked else None)
    if active_question:
        with st.spinner("Asking Gemini..."):
            answer = explain_tax_term(active_question, context)
        st.markdown(f'<div class="info-box">{escape(answer)}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Ledger Data Analytics (Gemini AI)")
    st.caption("Analyze larger company datasets by category-level totals.")
    ledger_question = st.text_input("Ask about this uploaded ledger", placeholder="e.g. Which expense category is highest?", key="ca_ledger_question")
    if st.button("Analyze Ledger", key="ca_ledger_ask") and ledger_question:
        with st.spinner("Analyzing ledger with Gemini..."):
            ans = summarize_ledger_for_ca(transactions, ledger_question)
        st.markdown(f'<div class="info-box">{escape(ans)}</div>', unsafe_allow_html=True)


def main() -> None:
    configure_page()
    profile = render_sidebar()

    st.markdown(f'<div class="hero"><h1>{APP_TITLE}</h1><p>{APP_SUBTITLE}</p></div>', unsafe_allow_html=True)

    tabs = st.tabs(["1. Attach Documents", "2. Review Summary", "3. Tax Estimate", "4. Checklist & Handoff"])

    with tabs[0]:
        st.subheader("1. Attach your records")
        uploads = st.file_uploader("Upload bank statements or CSV/Excel files", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True)
        table_frames = []
        pdf_previews = []
        for uploaded in uploads or []:
            if uploaded.name.lower().endswith(".pdf"):
                pdf_previews.append(f"**{uploaded.name}**\n\n{extract_pdf_preview(uploaded)}")
            else:
                table_frames.append(read_uploaded_table(uploaded))
        transactions = standardize_transactions(table_frames)
        st.success(f"Loaded {len(transactions)} transaction records.")

    with tabs[1]:
        st.subheader("2. Review Summary")
        income = float(transactions.loc[transactions["type"] == "Income", "amount"].sum())
        expenses = float(transactions.loc[transactions["type"] == "Expense", "absolute_amount"].sum())
        regular_profit = max(income - expenses, 0.0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Business Receipts", money(income))
        m2.metric("Expenses", money(expenses))
        m3.metric("Estimated Profit", money(regular_profit))

        c1, c2 = st.columns(2)
        c1.plotly_chart(build_monthly_chart(transactions), use_container_width=True)
        c2.plotly_chart(build_category_chart(transactions), use_container_width=True)

    with tabs[2]:
        st.subheader("3. Tax Estimate")
        presumptive_income, notes = estimate_presumptive_income(
            float(profile["digital_receipts"]),
            float(profile["cash_receipts"]),
            float(profile["professional_receipts"]),
            str(profile["presumptive_mode"]),
        )

        business_income = presumptive_income if str(profile["itr_path"]).startswith("ITR-4") and presumptive_income > 0 else regular_profit
        taxable_income = max(business_income + float(profile["other_income"]) - float(profile["deductions"]), 0.0)
        estimate = calculate_tax_estimate(taxable_income, float(profile["prepaid_tax"]), bool(profile["enable_rebate"]))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Business Income", money(business_income))
        m2.metric("Taxable Income", money(estimate.taxable_income))
        m3.metric("Estimated Tax", money(estimate.net_tax))
        m4.metric("Balance Payable", money(estimate.balance_payable))

        st.plotly_chart(build_tax_waterfall(estimate, float(profile["prepaid_tax"])), use_container_width=True)
        st.divider()
        render_ca_assistant(profile, estimate, presumptive_income, regular_profit, transactions)

    with tabs[3]:
        st.subheader("4. Checklist & Handoff")
        done = [item for item in REQUIRED_CHECKLIST if st.checkbox(item)]
        st.progress(len(done) / len(REQUIRED_CHECKLIST))

        packet = make_download_packet(profile, transactions, regular_profit, presumptive_income, estimate)
        st.download_button("Download Handoff JSON", data=packet, file_name="tax_summary.json", mime="application/json")


if __name__ == "__main__":
    main()
