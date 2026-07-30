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

            clean = pd.DataFrame()
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
        transactions["date"] = transactions["date"].fillna(pd.Timestamp(f"{FINANCIAL_YEAR[:4]}-04-01"))
        return transactions.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        st.error("Uploaded transactions could not be cleaned.")
        st.exception(exc)
        return make_sample_transactions()


@st.cache_data(show_spinner=False)
def make_sample_transactions() -> pd.DataFrame:
    """Create a friendly sample ledger so first-time users see the workflow."""

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
            # Full rebate: taxable income at or below Rs 12L is tax-free (up to Rs 60,000 rebate).
            rebate = min(gross_tax, 60_000.0)
        elif enable_87a_rebate and taxable_income > 1_200_000:
            # Marginal relief: tax on income just above Rs 12L cannot exceed the excess
            # income itself, until that cap exceeds the normal slab tax (~Rs 12.75L breakeven).
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
# UI Styling
# =============================================================================

def configure_page() -> None:
    """Configure page and apply a bright, beginner-friendly interface."""

    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        /* Hide Streamlit Community Cloud's GitHub badge, hamburger menu, and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
        .stAppDeployButton {display: none;}
        a[href*="github.com"] {display: none !important;}
        .viewerBadge_container__1QSob,
        .viewerBadge_link__1S137,
        .styles_viewerBadge__1yB5_,
        .stAppToolbar {display: none !important;}

        .stApp {
            background:
                radial-gradient(circle at 9% 8%, rgba(70, 167, 255, 0.18), transparent 28%),
                radial-gradient(circle at 90% 16%, rgba(255, 185, 91, 0.20), transparent 27%),
                linear-gradient(135deg, #f6fbff 0%, #eef7f1 48%, #fff7ec 100%);
            color: #142033;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.74);
            border-right: 1px solid rgba(20, 32, 51, 0.10);
            backdrop-filter: blur(16px);
        }

        .hero {
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(20, 32, 51, 0.10);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.70);
            box-shadow: 0 22px 60px rgba(29, 76, 112, 0.12);
            backdrop-filter: blur(18px);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0 0 .25rem 0;
            color: #142033;
            font-size: clamp(2rem, 4vw, 3.8rem);
            letter-spacing: 0;
        }

        .hero p {
            margin: 0;
            color: #4c5b6d;
            font-size: 1.04rem;
            line-height: 1.55;
        }

        .step-card {
            border: 1px solid rgba(20, 32, 51, 0.10);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.68);
            padding: 1rem;
            box-shadow: 0 16px 40px rgba(29, 76, 112, 0.10);
            min-height: 132px;
        }

        .step-card span {
            color: #637083;
            text-transform: uppercase;
            font-size: .78rem;
            font-weight: 700;
        }

        .step-card strong {
            display: block;
            margin-top: .35rem;
            color: #142033;
            font-size: 1.25rem;
        }

        .success-box, .warn-box, .info-box {
            border-radius: 8px;
            padding: .9rem 1rem;
            margin: .6rem 0;
            border: 1px solid rgba(20, 32, 51, 0.10);
        }

        .success-box { background: rgba(34, 197, 94, .15); }
        .warn-box { background: rgba(245, 158, 11, .18); }
        .info-box { background: rgba(59, 130, 246, .14); }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(20, 32, 51, 0.08);
            border-radius: 8px;
            padding: .8rem;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(20, 32, 51, 0.10);
        }

        @keyframes pageGlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes riseIn {
            from { opacity: 0; transform: translateY(18px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes softFloat {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }

        @keyframes typing {
            from { width: 0; }
            to { width: 100%; }
        }

        @keyframes blinkCaret {
            0%, 45% { border-color: #2563eb; }
            46%, 100% { border-color: transparent; }
        }

        @keyframes shimmer {
            from { transform: translateX(-120%) skewX(-18deg); }
            to { transform: translateX(140%) skewX(-18deg); }
        }

        .stApp {
            background:
                linear-gradient(115deg, rgba(255,255,255,.86), rgba(236,247,255,.72)),
                repeating-linear-gradient(90deg, rgba(37,99,235,.055) 0 1px, transparent 1px 72px),
                repeating-linear-gradient(0deg, rgba(20,184,166,.050) 0 1px, transparent 1px 72px),
                linear-gradient(135deg, #f7fbff, #edf8f1, #fff5e8, #eef4ff);
            background-size: auto, auto, auto, 260% 260%;
            animation: pageGlow 18s ease infinite;
        }

        .block-container {
            padding-top: 1.25rem;
            animation: riseIn .55s ease-out both;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.78);
            box-shadow: 18px 0 50px rgba(29, 76, 112, .08);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label {
            color: #142033 !important;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.45rem 1.55rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,.86), rgba(255,255,255,.62)),
                linear-gradient(120deg, rgba(37,99,235,.12), rgba(20,184,166,.10), rgba(249,115,22,.12));
            animation: riseIn .7s ease-out both;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            width: 42%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,.58), transparent);
            animation: shimmer 5.5s ease-in-out infinite;
            pointer-events: none;
        }

        .hero h1,
        .hero p,
        .typing-wrap {
            position: relative;
            z-index: 1;
        }

        .typing-wrap {
            margin-top: .85rem;
            width: min(760px, 100%);
        }

        .typing-line {
            display: inline-block;
            max-width: 100%;
            overflow: hidden;
            white-space: nowrap;
            border-right: 3px solid #2563eb;
            color: #1d4ed8;
            font-weight: 800;
            animation: typing 3.6s steps(68, end) .35s both, blinkCaret .9s step-end infinite;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .85rem;
            margin: .85rem 0 1rem 0;
        }

        .step-card {
            position: relative;
            overflow: hidden;
            min-height: 126px;
            transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
            animation: riseIn .65s ease-out both;
        }

        .step-card:hover {
            transform: translateY(-5px);
            border-color: rgba(37, 99, 235, 0.26);
            box-shadow: 0 24px 58px rgba(29, 76, 112, 0.18);
        }

        .step-card:nth-child(2) { animation-delay: .08s; }
        .step-card:nth-child(3) { animation-delay: .16s; }
        .step-card:nth-child(4) { animation-delay: .24s; }

        .step-icon {
            display: inline-grid;
            place-items: center;
            width: 38px;
            height: 38px;
            border-radius: 8px;
            color: #ffffff;
            font-weight: 900;
            background: linear-gradient(135deg, #2563eb, #14b8a6);
            box-shadow: 0 12px 24px rgba(37,99,235,.22);
            animation: softFloat 4s ease-in-out infinite;
        }

        .step-card span {
            display: block;
            margin-top: .75rem;
        }

        .step-card strong {
            font-size: 1.12rem;
            line-height: 1.25;
        }

        .step-card p {
            margin: .35rem 0 0 0;
            color: #5e6c80;
            font-size: .92rem;
            line-height: 1.35;
        }

        .success-box, .warn-box, .info-box {
            animation: riseIn .45s ease-out both;
        }

        div[data-testid="stMetric"] {
            box-shadow: 0 14px 35px rgba(29, 76, 112, .09);
            transition: transform .2s ease, box-shadow .2s ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 48px rgba(29, 76, 112, .15);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .45rem;
            background: rgba(255,255,255,.52);
            padding: .45rem;
            border-radius: 8px;
            border: 1px solid rgba(20, 32, 51, 0.09);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: .6rem .9rem;
            transition: background .2s ease, transform .2s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(37,99,235,.16), rgba(20,184,166,.16));
            color: #1d4ed8;
            transform: translateY(-1px);
        }

        .stButton button,
        .stDownloadButton button {
            border-radius: 8px;
            border: 0;
            background: linear-gradient(135deg, #2563eb, #14b8a6);
            color: white;
            font-weight: 800;
            box-shadow: 0 16px 32px rgba(37,99,235,.22);
            transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            transform: translateY(-2px);
            filter: brightness(1.04);
            box-shadow: 0 20px 42px rgba(37,99,235,.28);
        }

        section[data-testid="stFileUploaderDropzone"] {
            border-radius: 8px;
            border: 1px dashed rgba(37,99,235,.34);
            background: rgba(255,255,255,.62);
            transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
        }

        section[data-testid="stFileUploaderDropzone"]:hover {
            transform: translateY(-3px);
            border-color: rgba(20,184,166,.62);
            box-shadow: 0 18px 38px rgba(20,184,166,.12);
        }

        @media (max-width: 900px) {
            .workflow-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .typing-line { white-space: normal; border-right: 0; animation: riseIn .7s ease-out both; }
        }

        @media (max-width: 560px) {
            .workflow-grid { grid-template-columns: 1fr; }
        }

        @keyframes liquidSweep {
            0% { transform: translateX(-18%) translateY(-6%) rotate(0deg); opacity: .68; }
            50% { transform: translateX(8%) translateY(4%) rotate(1deg); opacity: .92; }
            100% { transform: translateX(-18%) translateY(-6%) rotate(0deg); opacity: .68; }
        }

        @keyframes scanLines {
            from { background-position: 0 0; }
            to { background-position: 0 96px; }
        }

        @keyframes quickPop {
            from { opacity: 0; transform: translateY(10px) scale(.985); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        @keyframes activePulse {
            0%, 100% { box-shadow: 0 0 0 rgba(45, 212, 191, 0); }
            50% { box-shadow: 0 0 26px rgba(45, 212, 191, .24); }
        }

        .stApp {
            color: #eef7ff;
            background:
                linear-gradient(115deg, rgba(3, 7, 18, .96), rgba(8, 19, 36, .96)),
                repeating-linear-gradient(90deg, rgba(45,212,191,.045) 0 1px, transparent 1px 82px),
                repeating-linear-gradient(0deg, rgba(96,165,250,.040) 0 1px, transparent 1px 82px),
                linear-gradient(135deg, #050816 0%, #071629 38%, #0b1020 62%, #10251f 100%);
            background-size: auto, auto, auto, 220% 220%;
            animation: pageGlow 12s ease infinite;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: -18% -22% auto -22%;
            height: 42vh;
            pointer-events: none;
            z-index: 0;
            background:
                linear-gradient(105deg, transparent 0%, rgba(45,212,191,.18) 18%, transparent 34%),
                linear-gradient(84deg, transparent 24%, rgba(96,165,250,.22) 48%, transparent 68%),
                linear-gradient(122deg, transparent 42%, rgba(250,204,21,.10) 57%, transparent 73%);
            filter: blur(18px);
            animation: liquidSweep 8s cubic-bezier(.22, 1, .36, 1) infinite;
        }

        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background:
                linear-gradient(rgba(255,255,255,.030) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.022) 1px, transparent 1px);
            background-size: 96px 96px;
            mask-image: linear-gradient(to bottom, rgba(0,0,0,.75), transparent 70%);
            animation: scanLines 7s linear infinite;
        }

        .block-container,
        [data-testid="stSidebar"] {
            position: relative;
            z-index: 1;
        }

        .block-container {
            animation: quickPop .42s cubic-bezier(.2, .9, .2, 1) both;
        }

        h1, h2, h3, h4, h5, h6,
        p, label, span, div, .stMarkdown,
        [data-testid="stMarkdownContainer"] {
            color: #eef7ff;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(8, 19, 36, .86), rgba(3, 7, 18, .88)),
                linear-gradient(135deg, rgba(45,212,191,.09), rgba(96,165,250,.07));
            border-right: 1px solid rgba(148, 163, 184, .18);
            box-shadow: 18px 0 60px rgba(0, 0, 0, .26);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #eaf7ff !important;
        }

        .hero,
        .step-card,
        div[data-testid="stMetric"],
        .success-box,
        .warn-box,
        .info-box,
        .stTabs [data-baseweb="tab-list"] {
            background:
                linear-gradient(145deg, rgba(15, 23, 42, .76), rgba(8, 19, 36, .62)),
                linear-gradient(115deg, rgba(45,212,191,.09), rgba(96,165,250,.08), rgba(250,204,21,.045));
            border: 1px solid rgba(148, 163, 184, .18);
            box-shadow: 0 24px 70px rgba(0, 0, 0, .28);
            backdrop-filter: blur(20px) saturate(1.18);
        }

        .hero {
            min-height: 190px;
            animation: quickPop .42s cubic-bezier(.2, .9, .2, 1) both, activePulse 3.2s ease-in-out infinite;
        }

        .hero h1 {
            color: #f8fbff;
            text-shadow: 0 0 34px rgba(45, 212, 191, .22);
        }

        .hero p {
            color: #c8d7e8;
        }

        .typing-wrap {
            max-width: 100%;
            overflow: hidden;
        }

        .typing-line {
            color: #5eead4;
            text-shadow: 0 0 22px rgba(45, 212, 191, .38);
            border-right-color: #5eead4;
            animation: typing 2.15s steps(68, end) .18s both, blinkCaret .62s step-end infinite;
        }

        .step-card {
            animation: quickPop .34s cubic-bezier(.2, .9, .2, 1) both;
        }

        .step-card:hover {
            transform: translateY(-7px) scale(1.012);
            border-color: rgba(94, 234, 212, .48);
            box-shadow: 0 28px 80px rgba(45, 212, 191, .16), 0 18px 48px rgba(0, 0, 0, .38);
        }

        .step-icon {
            background: linear-gradient(135deg, #22d3ee, #2dd4bf 52%, #facc15);
            color: #04111f;
            box-shadow: 0 0 28px rgba(45, 212, 191, .28);
            animation: softFloat 2.6s ease-in-out infinite;
        }

        .step-card span,
        .step-card p,
        .hero p {
            color: #b9c7d8;
        }

        .step-card strong,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f8fbff;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px) scale(1.01);
            border-color: rgba(96,165,250,.42);
            box-shadow: 0 28px 72px rgba(96,165,250,.15), 0 14px 42px rgba(0,0,0,.34);
        }

        .stTabs [data-baseweb="tab-list"] {
            padding: .35rem;
        }

        .stTabs [data-baseweb="tab"] {
            color: #b9c7d8;
            transition: transform .15s ease, background .15s ease, color .15s ease;
        }

        .stTabs [aria-selected="true"] {
            color: #f8fbff;
            background: linear-gradient(135deg, rgba(45,212,191,.22), rgba(96,165,250,.18));
            box-shadow: inset 0 0 0 1px rgba(94,234,212,.28);
        }

        .stButton button,
        .stDownloadButton button {
            background: linear-gradient(135deg, #22d3ee, #2dd4bf 55%, #facc15);
            color: #03111f;
            box-shadow: 0 18px 42px rgba(45, 212, 191, .25);
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 24px 58px rgba(45, 212, 191, .34);
        }

        section[data-testid="stFileUploaderDropzone"] {
            background:
                linear-gradient(145deg, rgba(15, 23, 42, .78), rgba(8, 19, 36, .70)),
                linear-gradient(115deg, rgba(45,212,191,.12), rgba(96,165,250,.08));
            border-color: rgba(94,234,212,.42);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.025);
        }

        section[data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(250,204,21,.62);
            box-shadow: 0 18px 48px rgba(45,212,191,.16);
        }

        input, textarea, select,
        [data-baseweb="input"],
        [data-baseweb="select"],
        [data-baseweb="textarea"] {
            color: #f8fbff !important;
            background-color: rgba(15, 23, 42, .72) !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            box-shadow: 0 20px 58px rgba(0,0,0,.30);
        }

        :root {
            --canvas: #050816;
            --surface-1: rgba(15, 23, 42, .82);
            --surface-2: rgba(8, 19, 36, .70);
            --stroke: rgba(148, 163, 184, .18);
            --stroke-hot: rgba(94, 234, 212, .48);
            --text-1: #f8fbff;
            --text-2: #b9c7d8;
            --accent-cyan: #22d3ee;
            --accent-mint: #2dd4bf;
            --accent-gold: #facc15;
            --accent-blue: #60a5fa;
        }

        @keyframes chipSlide {
            0% { transform: translateX(-8px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }

        @keyframes railMove {
            from { transform: translateX(-34%); }
            to { transform: translateX(34%); }
        }

        @keyframes nodePing {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 rgba(45,212,191,0); }
            50% { transform: scale(1.18); box-shadow: 0 0 28px rgba(45,212,191,.45); }
        }

        .designer-shell {
            position: relative;
            overflow: hidden;
            display: grid;
            grid-template-columns: 1.1fr 1.35fr .95fr;
            gap: .9rem;
            align-items: center;
            margin: .9rem 0 1rem 0;
            padding: .85rem;
            border-radius: 8px;
            border: 1px solid var(--stroke);
            background:
                linear-gradient(145deg, var(--surface-1), var(--surface-2)),
                linear-gradient(110deg, rgba(34,211,238,.10), rgba(45,212,191,.08), rgba(250,204,21,.05));
            box-shadow: 0 26px 80px rgba(0,0,0,.30);
            backdrop-filter: blur(22px) saturate(1.2);
            animation: quickPop .38s cubic-bezier(.2,.9,.2,1) both;
        }

        .designer-shell::before {
            content: "";
            position: absolute;
            left: -30%;
            right: -30%;
            top: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-mint), var(--accent-gold), transparent);
            animation: railMove 2.4s ease-in-out infinite alternate;
        }

        .workspace-brand {
            display: flex;
            gap: .75rem;
            align-items: center;
            min-width: 0;
        }

        .brand-mark {
            display: grid;
            place-items: center;
            width: 44px;
            height: 44px;
            border-radius: 8px;
            color: #03111f;
            font-weight: 950;
            background: conic-gradient(from 150deg, var(--accent-cyan), var(--accent-mint), var(--accent-gold), var(--accent-blue), var(--accent-cyan));
            box-shadow: 0 0 34px rgba(45,212,191,.28);
            animation: nodePing 2.9s ease-in-out infinite;
        }

        .workspace-brand strong {
            display: block;
            color: var(--text-1);
            font-size: 1rem;
            line-height: 1.15;
        }

        .workspace-brand span {
            display: block;
            margin-top: .18rem;
            color: var(--text-2);
            font-size: .82rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .command-pill {
            display: flex;
            align-items: center;
            gap: .65rem;
            min-height: 44px;
            padding: 0 .9rem;
            border-radius: 8px;
            border: 1px solid rgba(148,163,184,.16);
            background: rgba(3, 7, 18, .48);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.025);
            color: var(--text-2);
            font-size: .92rem;
        }

        .command-dot {
            width: 9px;
            height: 9px;
            border-radius: 99px;
            background: var(--accent-mint);
            box-shadow: 0 0 18px rgba(45,212,191,.75);
            flex: 0 0 auto;
        }

        .status-cluster {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: .45rem;
        }

        .status-chip {
            padding: .45rem .58rem;
            border-radius: 8px;
            border: 1px solid rgba(148,163,184,.16);
            background: rgba(255,255,255,.055);
            color: #ddecff;
            font-size: .78rem;
            font-weight: 800;
            animation: chipSlide .26s ease-out both;
        }

        .status-chip:nth-child(2) { animation-delay: .05s; }
        .status-chip:nth-child(3) { animation-delay: .10s; }

        .ux-focus-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
            margin: 0 0 1rem 0;
        }

        .ux-focus {
            position: relative;
            overflow: hidden;
            padding: .8rem .85rem;
            min-height: 82px;
            border-radius: 8px;
            border: 1px solid rgba(148,163,184,.16);
            background:
                linear-gradient(145deg, rgba(15,23,42,.72), rgba(8,19,36,.58)),
                radial-gradient(circle at 92% 16%, rgba(45,212,191,.18), transparent 34%);
            box-shadow: 0 18px 48px rgba(0,0,0,.22);
            transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
        }

        .ux-focus:hover {
            transform: translateY(-4px);
            border-color: var(--stroke-hot);
            box-shadow: 0 26px 70px rgba(45,212,191,.12), 0 18px 48px rgba(0,0,0,.32);
        }

        .ux-focus span {
            color: var(--accent-mint);
            font-size: .72rem;
            font-weight: 900;
            text-transform: uppercase;
        }

        .ux-focus strong {
            display: block;
            margin-top: .28rem;
            color: var(--text-1);
            font-size: 1rem;
        }

        .ux-focus p {
            margin: .22rem 0 0 0;
            color: var(--text-2);
            font-size: .84rem;
            line-height: 1.3;
        }

        @media (max-width: 980px) {
            .designer-shell { grid-template-columns: 1fr; }
            .status-cluster { justify-content: flex-start; }
            .ux-focus-row { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Sidebar and Inputs
# =============================================================================

def render_sidebar() -> Dict[str, object]:
    """Collect taxpayer profile and tax inputs using simple language."""

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
            help="ITR-4 is commonly used by eligible small taxpayers using presumptive sections 44AD/44ADA/44AE.",
        )

        st.divider()
        st.subheader("Quick Money Inputs")
        other_income = st.number_input("Other income, if any", min_value=0.0, value=0.0, step=5_000.0)
        deductions = st.number_input("Basic deductions you want to track", min_value=0.0, value=0.0, step=5_000.0)
        prepaid_tax = st.number_input("TDS + advance tax already paid", min_value=0.0, value=0.0, step=5_000.0)
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
    """Show expense categories as an interactive donut chart."""

    expenses = transactions[transactions["type"] == "Expense"].copy()
    if expenses.empty:
        expenses = pd.DataFrame({"category": ["No expenses"], "absolute_amount": [1.0]})

    grouped = expenses.groupby("category", as_index=False)["absolute_amount"].sum()
    fig = px.pie(
        grouped,
        values="absolute_amount",
        names="category",
        hole=0.55,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        height=330,
        margin=dict(l=12, r=12, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaf7ff"),
        showlegend=False,
    )
    return fig


def build_monthly_chart(transactions: pd.DataFrame) -> go.Figure:
    """Show month-wise income, expenses, and profit."""

    df = transactions.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly = df.pivot_table(index="month", columns="type", values="amount", aggfunc="sum", fill_value=0)
    monthly["Income"] = monthly.get("Income", 0)
    monthly["Expense"] = monthly.get("Expense", 0).abs()
    monthly["Profit"] = monthly["Income"] - monthly["Expense"]
    monthly = monthly.reset_index()

    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["Income"], name="Income", marker_color="#22c55e")
    fig.add_bar(x=monthly["month"], y=monthly["Expense"], name="Expenses", marker_color="#f59e0b")
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["Profit"],
            mode="lines+markers",
            name="Profit",
            line=dict(color="#2563eb", width=3),
        )
    )
    fig.update_layout(
        height=330,
        barmode="group",
        margin=dict(l=12, r=12, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.42)",
        font=dict(color="#eaf7ff"),
        yaxis_title="Amount",
        xaxis_title="",
        legend=dict(orientation="h"),
        xaxis=dict(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.18)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.18)"),
    )
    return fig


def build_tax_waterfall(estimate: TaxEstimate, prepaid_tax: float) -> go.Figure:
    """Visualize how gross tax moves to final payable."""

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Gross tax", "87A rebate", "Cess", "TDS/advance tax", "Payable"],
            y=[
                estimate.gross_tax,
                -estimate.rebate_87a,
                estimate.cess,
                -min(prepaid_tax, estimate.net_tax),
                estimate.balance_payable,
            ],
            connector={"line": {"color": "rgba(20,32,51,.28)"}},
            increasing={"marker": {"color": "#2563eb"}},
            decreasing={"marker": {"color": "#22c55e"}},
            totals={"marker": {"color": "#f97316"}},
        )
    )
    fig.update_layout(
        height=310,
        margin=dict(l=12, r=12, t=18, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.42)",
        font=dict(color="#eaf7ff"),
        yaxis_title="Amount",
        xaxis=dict(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.18)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.18)"),
    )
    return fig


def make_download_packet(
    profile: Dict[str, object],
    transactions: pd.DataFrame,
    regular_profit: float,
    presumptive_income: float,
    estimate: TaxEstimate,
) -> str:
    """Create a compact JSON packet users can share with their accountant."""

    packet = {
        "generated_on": date.today().isoformat(),
        "assessment_year": ASSESSMENT_YEAR,
        "financial_year": FINANCIAL_YEAR,
        "profile": profile,
        "summary": {
            "uploaded_income": float(transactions.loc[transactions["type"] == "Income", "amount"].sum()),
            "uploaded_expenses": float(transactions.loc[transactions["type"] == "Expense", "absolute_amount"].sum()),
            "regular_profit": float(regular_profit),
            "presumptive_income": float(presumptive_income),
            "estimated_taxable_income": float(estimate.taxable_income),
            "estimated_net_tax": float(estimate.net_tax),
            "estimated_balance_payable": float(estimate.balance_payable),
        },
        "checklist": REQUIRED_CHECKLIST,
    }
    return json.dumps(packet, indent=2)


# =============================================================================
# Main Dashboard Rendering
# =============================================================================

def render_hero() -> None:
    """Render the dashboard identity."""

    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>
                {APP_SUBTITLE}. Upload your bank statement or sales file, review income and expenses,
                estimate your tax, and leave with a filing-ready checklist.
            </p>
            <div class="typing-wrap">
                <div class="typing-line">Your friendly tax co-pilot: attach files, understand numbers, prepare faster.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_design_workspace(profile: Dict[str, object]) -> None:
    """Render a Pixso/Figma-inspired workspace command surface."""

    taxpayer_name = escape(str(profile.get("taxpayer_name", "My Small Business")))
    business_type = escape(str(profile.get("business_type", "Small business")))
    itr_path = escape(str(profile.get("itr_path", "ITR preparation")))

    st.markdown(
        f"""
        <div class="designer-shell">
            <div class="workspace-brand">
                <div class="brand-mark">ITR</div>
                <div>
                    <strong>{taxpayer_name}</strong>
                    <span>{business_type} workspace</span>
                </div>
            </div>
            <div class="command-pill">
                <div class="command-dot"></div>
                <span>Smart flow active: documents, books, tax estimate, handoff</span>
            </div>
            <div class="status-cluster">
                <div class="status-chip">{ASSESSMENT_YEAR}</div>
                <div class="status-chip">Dark DSM</div>
                <div class="status-chip">{itr_path}</div>
            </div>
        </div>
        <div class="ux-focus-row">
            <div class="ux-focus">
                <span>Auto layout</span>
                <strong>Responsive filing flow</strong>
                <p>Each step stays clear on laptop, tablet, and mobile screens.</p>
            </div>
            <div class="ux-focus">
                <span>Design tokens</span>
                <strong>Consistent dark interface</strong>
                <p>Readable colors, repeatable spacing, and crisp contrast.</p>
            </div>
            <div class="ux-focus">
                <span>Handoff</span>
                <strong>Accountant-ready output</strong>
                <p>Cleaned transactions and JSON summary are ready to share.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_step() -> Tuple[pd.DataFrame, List[str]]:
    """Render document upload area and return cleaned transactions plus PDF previews."""

    st.subheader("1. Attach your records")
    st.caption("Upload CSV or Excel statements. PDFs are kept in your document vault with a short preview when possible.")
    uploads = st.file_uploader(
        "Drop bank statements, sales sheets, or expense files",
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

    if not uploads:
        st.markdown(
            """
            <div class="info-box">
                No file uploaded yet, so the app is showing sample transactions. Upload your CSV or Excel file to replace it.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="success-box">
                {len(uploads)} file(s) attached. The table below is ready for review.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return transactions, pdf_previews


def render_review_step(transactions: pd.DataFrame) -> Tuple[float, float, float]:
    """Render transaction review, metrics, and charts."""

    income = float(transactions.loc[transactions["type"] == "Income", "amount"].sum())
    expenses = float(transactions.loc[transactions["type"] == "Expense", "absolute_amount"].sum())
    regular_profit = max(income - expenses, 0.0)

    st.subheader("2. Review the auto-summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Business receipts found", money(income))
    m2.metric("Expenses found", money(expenses))
    m3.metric("Estimated book profit", money(regular_profit))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_monthly_chart(transactions), use_container_width=True)
    with right:
        st.plotly_chart(build_category_chart(transactions), use_container_width=True)

    with st.expander("Open cleaned transaction table"):
        st.dataframe(
            transactions[["date", "description", "type", "category", "amount"]].style.format(
                {"date": lambda value: pd.to_datetime(value).strftime("%d-%b-%Y"), "amount": "₹{:,.0f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    return income, expenses, regular_profit


def render_tax_step(
    profile: Dict[str, object],
    regular_profit: float,
) -> Tuple[float, TaxEstimate, List[str]]:
    """Render presumptive helper and final tax estimate."""

    st.subheader("3. Estimate tax and choose your filing path")

    presumptive_income, notes = estimate_presumptive_income(
        float(profile["digital_receipts"]),
        float(profile["cash_receipts"]),
        float(profile["professional_receipts"]),
        str(profile["presumptive_mode"]),
    )

    if str(profile["itr_path"]).startswith("ITR-4"):
        business_income = presumptive_income if presumptive_income > 0 else regular_profit
        filing_note = "ITR-4 preparation mode selected. Presumptive income is used when you enter receipts."
    else:
        business_income = regular_profit
        filing_note = "ITR-3 preparation mode selected. Book profit from uploaded transactions is used."

    taxable_income = max(business_income + float(profile["other_income"]) - float(profile["deductions"]), 0.0)
    estimate = calculate_tax_estimate(
        taxable_income=taxable_income,
        tds_and_advance_tax=float(profile["prepaid_tax"]),
        enable_87a_rebate=bool(profile["enable_rebate"]),
    )

    st.markdown(f"""<div class="info-box">{filing_note}</div>""", unsafe_allow_html=True)
    for note in notes:
        st.markdown(f"""<div class="warn-box">{note}</div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Business income used", money(business_income))
    c2.metric("Taxable income", money(estimate.taxable_income))
    c3.metric("Estimated net tax", money(estimate.net_tax))
    c4.metric("Balance payable", money(estimate.balance_payable))

    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(build_tax_waterfall(estimate, float(profile["prepaid_tax"])), use_container_width=True)
    with right:
        st.markdown("#### Tax calculation")
        tax_table = pd.DataFrame(
            [
                ("Gross tax", estimate.gross_tax),
                ("Less: 87A rebate", -estimate.rebate_87a),
                ("Health and education cess", estimate.cess),
                ("Net tax", estimate.net_tax),
                ("Less: TDS / advance tax", -float(profile["prepaid_tax"])),
                ("Estimated amount payable", estimate.balance_payable),
            ],
            columns=["Line item", "Amount"],
        )
        st.dataframe(
            tax_table.style.format({"Amount": "₹{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )

    return presumptive_income, estimate, notes


def render_finish_step(
    profile: Dict[str, object],
    transactions: pd.DataFrame,
    regular_profit: float,
    presumptive_income: float,
    estimate: TaxEstimate,
    pdf_previews: List[str],
) -> None:
    """Render checklist, document vault, and downloadable packet."""

    st.subheader("4. Filing-ready checklist")

    done_items = []
    cols = st.columns(3)
    for index, item in enumerate(REQUIRED_CHECKLIST):
        with cols[index % 3]:
            if st.checkbox(item, value=index < 4):
                done_items.append(item)

    progress = len(done_items) / len(REQUIRED_CHECKLIST)
    st.progress(progress, text=f"{len(done_items)} of {len(REQUIRED_CHECKLIST)} items ready")

    if pdf_previews:
        with st.expander("Document vault previews"):
            for preview in pdf_previews:
                st.markdown(preview)
                st.divider()

    packet = make_download_packet(profile, transactions, regular_profit, presumptive_income, estimate)
    csv_buffer = io.StringIO()
    transactions.to_csv(csv_buffer, index=False)

    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download cleaned transaction CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_tax_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "Download accountant handoff JSON",
            data=packet,
            file_name="tax_prep_handoff.json",
            mime="application/json",
            use_container_width=True,
        )


def render_guidance() -> None:
    """Render official-source notes and app disclaimer."""

    with st.expander("Official guidance used in this app"):
        st.markdown(
            """
            This app uses the AY 2026-27 new-regime slab structure and small-business ITR guidance
            from the Income Tax Department. It is a preparation helper, not a filing portal or legal opinion.
            """
        )
        for label, url in OFFICIAL_SOURCES.items():
            st.markdown(f"- [{label}]({url})")

    st.warning(
        "Before filing, reconcile with Form 26AS/AIS/TIS and confirm eligibility on the official Income Tax portal or with a qualified tax professional."
    )


def check_access() -> bool:
    """Simple passcode gate so the app isn't freely usable by anyone with the link.

    Set an ACCESS_CODE value in Streamlit Cloud's Secrets manager
    (App settings -> Secrets) like this:

        ACCESS_CODE = "your-chosen-code"

    Share that code only with users who have paid / been granted access.
    This is a stopgap, not real authentication or payment enforcement.
    """
    configured_code = st.secrets.get("ACCESS_CODE", None)

    if not configured_code:
        # No code configured yet in Secrets -> app stays open.
        # Add ACCESS_CODE to Secrets once you're ready to gate access.
        return True

    if st.session_state.get("access_granted", False):
        return True

    st.markdown(
        """
        <div class="step-card">
            <span>Access required</span>
            <strong>Enter your access code to continue</strong>
            <p>This dashboard is limited to authorized users.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    entered_code = st.text_input("Access code", type="password", key="access_code_input")
    submit = st.button("Unlock")

    if submit:
        if entered_code == configured_code:
            st.session_state["access_granted"] = True
            st.rerun()
        else:
            st.error("Incorrect access code. Please try again or contact the app owner.")

    return False


def main() -> None:
    """Application entry point."""

    configure_page()

    if not check_access():
        st.stop()

    profile = render_sidebar()
    render_hero()
    render_design_workspace(profile)

    st.markdown(
        """
        <div class="step-card">
            <span>Beginner workflow</span>
            <strong>Attach → Review → Estimate → File-ready checklist</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="workflow-grid">
            <div class="step-card">
                <div class="step-icon">1</div>
                <span>Attach</span>
                <strong>Drop your records</strong>
                <p>Use CSV, Excel, or PDF files from your business.</p>
            </div>
            <div class="step-card">
                <div class="step-icon">2</div>
                <span>Review</span>
                <strong>See income and expenses</strong>
                <p>Auto-categorized tables and visual summaries.</p>
            </div>
            <div class="step-card">
                <div class="step-icon">3</div>
                <span>Estimate</span>
                <strong>Check tax payable</strong>
                <p>ITR-3 and ITR-4 paths in beginner language.</p>
            </div>
            <div class="step-card">
                <div class="step-icon">4</div>
                <span>Finish</span>
                <strong>Download your handoff</strong>
                <p>Checklist, cleaned CSV, and accountant-ready JSON.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Attach documents", "Review summary", "Tax estimate", "Finish checklist"])

    with tabs[0]:
        transactions, pdf_previews = render_upload_step()

    with tabs[1]:
        income, expenses, regular_profit = render_review_step(transactions)
        del income, expenses

    with tabs[2]:
        presumptive_income, estimate, _notes = render_tax_step(profile, regular_profit)

    with tabs[3]:
        render_finish_step(profile, transactions, regular_profit, presumptive_income, estimate, pdf_previews)
        render_guidance()


if __name__ == "__main__":
    main()
