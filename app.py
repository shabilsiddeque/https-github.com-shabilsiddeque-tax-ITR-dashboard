"""
Beginner Tax Desk & CA Assist Platform (FY 2025-26 / AY 2026-27)

Features:
- Fixed FPDF Unicode Encoding Issue (ASCII Rupee Symbol in PDF)
- Form 16, Bank Statement & Document Ingestion
- Old vs New Tax Regime Calculation Engine (FY 2025-26 / AY 2026-27)
- Free AI Analysis via Google Gemini 2.5 Flash
- Provenance Audit Trail & Live Execution Logs
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from html import escape
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# =============================================================================
# App Configuration & Legal Slabs (FY 2025-26 / AY 2026-27)
# =============================================================================

APP_TITLE = "Beginner Tax Desk & CA Assist"
APP_SUBTITLE = "Smart ITR preparation, AI auditing, and legal regime optimization"
ASSESSMENT_YEAR = "AY 2026-27"
FINANCIAL_YEAR = "FY 2025-26"
FY_START_YEAR = "2025"

# New Tax Regime Slabs (u/s 115BAC) - FY 2025-26
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
    "PAN and Aadhaar linking verified",
    "Active bank account details with valid IFSC",
    "Form 16 (Part A & Part B) or monthly salary slips",
    "Annual Information Statement (AIS) / TIS cross-verification",
    "Form 26AS tax credit reconciliation",
    "Bank statements for full FY 2025-26",
    "Deduction proofs for Chapter VI-A (80C, 80D, 80CCD)",
    "HRA receipts, rent agreement, and landlord details",
    "GST returns vs sales register reconciliation (for businesses)",
    "Challan receipts for advance tax and self-assessment tax",
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
# Helper Formatting Utilities
# =============================================================================

def money(value: float) -> str:
    """Format currency values into Indian Rupees for Web UI."""
    try:
        return f"₹{value:,.0f}"
    except Exception:
        return "₹0"


def pdf_money(value: float) -> str:
    """Format currency using ASCII 'Rs.' for FPDF to prevent Unicode Encoding Errors."""
    try:
        return f"Rs. {value:,.0f}"
    except Exception:
        return "Rs. 0"


# =============================================================================
# PDF Report Generator Engine
# =============================================================================

class TaxReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, "INCOME TAX COMPUTATION REPORT", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f"Financial Year: {FINANCIAL_YEAR}  |  Assessment Year: {ASSESSMENT_YEAR}", ln=True, align="C")
        self.ln(5)
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Generated via Beginner Tax Desk & CA Assist  |  Page {self.page_no()}", align="C")


def generate_pdf_summary(
    profile: Dict[str, object],
    new_est: TaxEstimate,
    old_est: TaxEstimate,
    checked_items: List[str],
) -> bytes:
    """Creates a downloadable Tax Summary PDF Report using safe ASCII formatting."""
    pdf = TaxReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Taxpayer Profile Section
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "1. Taxpayer Profile & Summary", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(95, 6, f"Taxpayer Name: {profile.get('taxpayer_name', 'N/A')}", ln=False)
    pdf.cell(95, 6, f"Employment Status: {'Salaried' if profile.get('is_salaried') else 'Non-Salaried'}", ln=True)
    pdf.cell(95, 6, f"Date Generated: {date.today().strftime('%d %B %Y')}", ln=False)

    best_regime = "New Tax Regime (u/s 115BAC)" if new_est.net_tax <= old_est.net_tax else "Old Tax Regime"
    pdf.cell(95, 6, f"Recommended Option: {best_regime}", ln=True)
    pdf.ln(4)

    # 2. Side-by-Side Tax Comparison Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "2. Tax Calculation Breakdown (New vs Old Regime)", ln=True)

    # Table Header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(70, 7, " Calculation Head", border=1, fill=True)
    pdf.cell(60, 7, " New Regime (u/s 115BAC)", border=1, fill=True, align="R")
    pdf.cell(60, 7, " Old Tax Regime", border=1, fill=True, align="R")
    pdf.ln()

    # Table Rows with ASCII pdf_money formatting
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)

    rows = [
        ("Gross Annual Income", pdf_money(new_est.gross_income), pdf_money(old_est.gross_income)),
        ("Total Deductions & Exemptions", pdf_money(new_est.deductions_applied), pdf_money(old_est.deductions_applied)),
        ("Net Taxable Income", pdf_money(new_est.taxable_income), pdf_money(old_est.taxable_income)),
        ("Gross Slab Tax", pdf_money(new_est.gross_tax), pdf_money(old_est.gross_tax)),
        ("Section 87A Rebate", pdf_money(-new_est.rebate_87a), pdf_money(-old_est.rebate_87a)),
        ("Health & Education Cess (4%)", pdf_money(new_est.cess), pdf_money(old_est.cess)),
        ("Total Net Tax Liability", pdf_money(new_est.net_tax), pdf_money(old_est.net_tax)),
        ("Prepaid Taxes / TDS Paid", pdf_money(-profile.get("prepaid_tax", 0)), pdf_money(-profile.get("prepaid_tax", 0))),
        ("Final Balance Payable / (Refund)", pdf_money(new_est.balance_payable), pdf_money(old_est.balance_payable)),
    ]

    for head, new_val, old_val in rows:
        pdf.cell(70, 6, f" {head}", border=1)
        pdf.cell(60, 6, f"{new_val} ", border=1, align="R")
        pdf.cell(60, 6, f"{old_val} ", border=1, align="R")
        pdf.ln()

    pdf.ln(6)

    # 3. Compliance Verification Checklist
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "3. Documentation & Verification Checklist Status", ln=True)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)

    for item in REQUIRED_CHECKLIST:
        status_text = "[ OK ] Verified" if item in checked_items else "[    ] Pending Verification"
        pdf.cell(140, 5, f"- {item}", ln=False)
        pdf.cell(50, 5, status_text, ln=True, align="R")

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 4, "Disclaimer: This document is an automated tax estimation summary. Please cross-verify figures with Form 26AS, AIS, and TIS before official e-filing on the Income Tax Portal.")

    return bytes(pdf.output())


# =============================================================================
# Execution Logger & Data Utilities
# =============================================================================

def add_audit_log(stage: str, details: str, status: str = "INFO") -> None:
    if "audit_trail" not in st.session_state:
        st.session_state["audit_trail"] = []
    timestamp = time.strftime("%H:%M:%S")
    st.session_state["audit_trail"].append({
        "time": timestamp,
        "stage": stage,
        "details": details,
        "status": status,
    })


def normalize_column_name(column: object) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(column).strip().lower())
    return cleaned.strip("_")


def infer_category(description: str, amount: float) -> str:
    text = str(description).lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in text for keyword in keywords):
            return category
    if amount > 0:
        return "Sales / Receipts"
    return "Other Expenses"


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return pd.DataFrame()
    except Exception as exc:
        add_audit_log("File Parsing", f"Error reading {uploaded_file.name}: {exc}", "ERROR")
        return pd.DataFrame()


def extract_pdf_full_text(uploaded_file, max_pages: int = 10, max_chars: int = 10_000) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages[:max_pages]]
        text = "\n".join(pages).strip()
        return text[:max_chars]
    except Exception as exc:
        add_audit_log("PDF Extraction", f"Failed reading PDF: {exc}", "WARN")
        return ""


def standardize_transactions(raw_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
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
        add_audit_log("Data Sanitization", f"Processed {len(transactions)} transaction records into unified schema.", "SUCCESS")
        return transactions.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        add_audit_log("Data Sanitization", f"Fallback to sample data due to: {exc}", "WARN")
        return make_sample_transactions()


@st.cache_data(show_spinner=False)
def make_sample_transactions() -> pd.DataFrame:
    rng = np.random.default_rng(101)
    months = pd.date_range("2025-04-01", periods=12, freq="MS")
    rows = []
    for month in months:
        sales = rng.integers(120_000, 210_000)
        rows.extend([
            (month + pd.Timedelta(days=2), "Client UPI Credit Payment", float(sales)),
            (month + pd.Timedelta(days=5), "Inventory & Raw Materials Purchase", -float(sales * 0.36)),
            (month + pd.Timedelta(days=10), "Commercial Office Rent", -25_000.0),
            (month + pd.Timedelta(days=15), "High-speed Broadband & Electricity", -8_200.0),
            (month + pd.Timedelta(days=22), "Google Ads & Social Media Marketing", -6_500.0),
        ])

    sample = pd.DataFrame(rows, columns=["date", "description", "amount"])
    sample["type"] = np.where(sample["amount"] >= 0, "Income", "Expense")
    sample["category"] = [infer_category(d, a) for d, a in zip(sample["description"], sample["amount"])]
    sample["absolute_amount"] = sample["amount"].abs()
    return sample


# =============================================================================
# Income Tax Calculation Engine (FY 2025-26 / AY 2026-27)
# =============================================================================

def calculate_new_regime_tax(gross_income: float, is_salaried: bool, prepaid_tax: float) -> TaxEstimate:
    std_deduction = 75_000.0 if is_salaried else 0.0
    taxable_income = max(gross_income - std_deduction, 0.0)

    gross_tax = 0.0
    for lower, upper, rate in NEW_REGIME_SLABS:
        if taxable_income > lower:
            slice_amt = min(taxable_income, upper) - lower
            gross_tax += slice_amt * rate

    if taxable_income <= 1_200_000:
        rebate = min(gross_tax, 60_000.0)
    elif taxable_income > 1_200_000:
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


# =============================================================================
# Google Gemini 2.5 Flash Free AI Integration
# =============================================================================

def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ModuleNotFoundError:
        return None


def ai_parse_document(doc_text: str) -> Tuple[Optional[Dict[str, float]], str]:
    if not doc_text.strip():
        return None, "No readable text found in PDF."

    client = get_gemini_client()
    if client is None:
        return None, "GEMINI_API_KEY is not configured in Secrets."

    try:
        from google.genai import types

        prompt = f"""
        Extract key Indian Income Tax fields from this document text.
        Return ONLY a JSON object with these exact numeric keys:
        - "gross_salary": gross salary or income (numeric)
        - "standard_deduction": standard deduction if mentioned (numeric)
        - "hra_exemption": HRA exemption u/s 10(13A) (numeric)
        - "sec_80c": Section 80C deductions (PPF, ELSS, LIC, tuition fee) (numeric)
        - "sec_80d": Section 80D health insurance (numeric)
        - "tds_deducted": Total TDS deducted at source (numeric)
        - "employer_name": Employer or company name (string)

        Document Content:
        {doc_text[:8000]}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
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
        add_audit_log("AI Document Ingestion", f"Successfully extracted parameters for employer: {data['employer_name']}", "SUCCESS")
        return data, "Extracted successfully using Gemini AI."
    except Exception as exc:
        add_audit_log("AI Document Ingestion", f"Extraction failed: {exc}", "ERROR")
        return None, f"Document AI extraction error: {type(exc).__name__}"


def ai_recommend_regime(new_est: TaxEstimate, old_est: TaxEstimate, total_gross: float) -> str:
    client = get_gemini_client()
    diff = abs(new_est.net_tax - old_est.net_tax)
    better = "New Tax Regime" if new_est.net_tax <= old_est.net_tax else "Old Tax Regime"

    if client is None:
        return f"**Analysis:** **{better}** is optimal for your profile, saving you {money(diff)} in net tax liability."

    try:
        prompt = f"""
        You are a senior Chartered Accountant reviewing tax liabilities for FY 2025-26 (AY 2026-27).
        Gross Annual Income: {money(total_gross)}
        New Regime Net Tax: {money(new_est.net_tax)} (Deductions: {money(new_est.deductions_applied)})
        Old Regime Net Tax: {money(old_est.net_tax)} (Deductions: {money(old_est.deductions_applied)})

        Provide a concise 3-bullet comparison:
        1. Winning regime and exact savings ({money(diff)}).
        2. Primary driver (e.g., Section 87A ₹12L slab threshold vs Chapter VI-A deductions).
        3. Strategic advice for the upcoming filing.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text or f"Recommend {better} with savings of {money(diff)}."
    except Exception:
        return f"**Analysis:** **{better}** saves {money(diff)}."


def ai_audit_ledger(transactions: pd.DataFrame, query: str) -> str:
    if transactions.empty:
        return "No ledger data available to audit."

    client = get_gemini_client()
    if client is None:
        return "Configure GEMINI_API_KEY in Secrets to activate CA AI Audit Assistant."

    try:
        by_cat_raw = transactions.groupby(["type", "category"])["absolute_amount"].sum().round(0).astype(int).to_dict()
        by_cat = {f"{k[0]} / {k[1]}": v for k, v in by_cat_raw.items()}

        monthly = transactions.copy()
        monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").astype(str)
        m_summary_raw = monthly.groupby(["month", "type"])["absolute_amount"].sum().round(0).astype(int).to_dict()
        m_summary = {f"{k[0]} / {k[1]}": v for k, v in m_summary_raw.items()}

        summary_data = {
            "total_records": len(transactions),
            "category_totals": by_cat,
            "monthly_breakdown": m_summary,
        }

        prompt = f"""
        You are an expert Indian CA auditor. Analyze this summarized client ledger:
        {json.dumps(summary_data, indent=2)}

        CA's Audit Query: {query}

        Provide 3-4 professional findings highlighting potential risks u/s 40A(3), cash ratios, GST mismatches, or unusual expense spikes.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        add_audit_log("CA AI Audit", "Audit inspection completed on current transaction state.", "SUCCESS")
        return response.text or "Audit analysis completed."
    except Exception as exc:
        add_audit_log("CA AI Audit", f"Audit query failed: {exc}", "ERROR")
        return f"Audit Error: {type(exc).__name__}"


# =============================================================================
# Custom UI & Styling Setup
# =============================================================================

def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}

        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(8px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .stApp {
            background: linear-gradient(125deg, #030712 0%, #0b1528 40%, #051a14 100%);
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            animation: fadeIn 0.4s ease-out;
        }

        .glass-card {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }

        .status-box {
            padding: 0.9rem 1.1rem;
            border-radius: 8px;
            margin: 0.6rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.95rem;
        }
        .status-box.info { background: rgba(59, 130, 246, 0.15); border-color: rgba(59, 130, 246, 0.3); color: #dbeafe; }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.6);
            padding: 0.4rem;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(45, 212, 191, 0.2), rgba(59, 130, 246, 0.2));
            color: #f8fafc !important;
            border: 1px solid rgba(45, 212, 191, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, object]:
    with st.sidebar:
        st.title("🧾 Tax Profile")
        st.caption(f"Filing Engine for {FINANCIAL_YEAR} ({ASSESSMENT_YEAR})")

        taxpayer_name = st.text_input("Taxpayer / Business Name", value="Anand Sharma", key="sb_name")
        is_salaried = st.checkbox("Is Salaried Individual?", value=True)

        st.divider()
        st.subheader("1. AI Document Parser")
        doc_file = st.file_uploader("Upload Form 16 / Salary Slip (PDF)", type=["pdf"], key="sb_doc")

        if doc_file is not None and st.button("Parse Document with AI", key="btn_parse_doc", use_container_width=True):
            text = extract_pdf_full_text(doc_file)
            with st.spinner("Extracting parameters with Gemini AI..."):
                parsed, msg = ai_parse_document(text)
            if parsed:
                st.session_state["in_gross_salary"] = parsed["gross_salary"]
                st.session_state["in_80c"] = parsed["sec_80c"]
                st.session_state["in_80d"] = parsed["sec_80d"]
                st.session_state["in_hra"] = parsed["hra_exemption"]
                st.session_state["in_prepaid_tax"] = parsed["tds_deducted"]
                st.success(f"{msg} ({parsed['employer_name']})")
            else:
                st.warning(msg)

        st.divider()
        st.subheader("2. Income & Exemptions")
        gross_salary = st.number_input("Gross Annual Income (₹)", min_value=0.0, value=1350000.0, step=25000.0, key="in_gross_salary")
        sec_80c = st.number_input("Section 80C Deductions (Max ₹1.5L)", min_value=0.0, value=150000.0, step=10000.0, key="in_80c")
        sec_80d = st.number_input("Section 80D Health Insurance (₹)", min_value=0.0, value=25000.0, step=5000.0, key="in_80d")
        sec_80ccd = st.number_input("Section 80CCD(1B) NPS (Max ₹50K)", min_value=0.0, value=50000.0, step=5000.0, key="in_80ccd")
        hra_exemption = st.number_input("HRA Exemption u/s 10(13A) (₹)", min_value=0.0, value=0.0, step=10000.0, key="in_hra")
        other_deductions = st.number_input("Other Deductions (₹)", min_value=0.0, value=0.0, step=5000.0, key="in_other_ded")

        prepaid_tax = st.number_input("TDS + Advance Tax Paid (₹)", min_value=0.0, value=55000.0, step=5000.0, key="in_prepaid_tax")

        st.divider()
        st.subheader("3. Presumptive Income")
        presumptive_mode = st.selectbox("Section Mode", ["Business 44AD", "Profession 44ADA"])
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
                name="New Tax Regime (u/s 115BAC)",
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
        font=dict(color="#f8fafc"),
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
        <div class="glass-card">
            <h1 style="margin:0; font-size:2.2rem; color:#f8fafc;">{APP_TITLE}</h1>
            <p style="margin-top:0.4rem; color:#94a3b8; font-size:1.05rem;">{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "1. Records Ingestion",
        "2. Regime Optimization",
        "3. CA AI Workbench",
        "4. Provenance Audit Log",
        "5. Handoff & PDF Export",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Records Ingestion
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("1. Ingest Bank Statements & Financial Data")
        st.caption("Upload CSV or Excel bank statements or use auto-generated sample ledger data below.")

        uploads = st.file_uploader(
            "Drop bank statements, spreadsheets, or financial CSVs",
            type=["csv", "xlsx", "xls", "pdf"],
            accept_multiple_files=True,
            key="tab1_uploads",
        )

        table_frames = []
        for uploaded in uploads or []:
            if not uploaded.name.lower().endswith(".pdf"):
                table_frames.append(read_uploaded_table(uploaded))

        transactions = standardize_transactions(table_frames)

        income = float(transactions.loc[transactions["type"] == "Income", "amount"].sum())
        expenses = float(transactions.loc[transactions["type"] == "Expense", "absolute_amount"].sum())
        book_profit = max(income - expenses, 0.0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Identified Receipts", money(income))
        m2.metric("Identified Expenses", money(expenses))
        m3.metric("Computed Book Profit", money(book_profit))

        with st.expander("View Cleaned Transactions Table"):
            st.dataframe(transactions, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 2: Legal Regime Optimization
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("2. Legal Income Tax Comparison (FY 2025-26 / AY 2026-27)")

        presumptive_inc = (
            (float(profile["digital_receipts"]) * 0.06 + float(profile["cash_receipts"]) * 0.08)
            if profile["presumptive_mode"] == "Business 44AD"
            else float(profile["professional_receipts"]) * 0.50
        )

        total_gross = float(profile["gross_salary"]) + presumptive_inc
        prepaid = float(profile["prepaid_tax"])

        # Execute Calculations
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

        add_audit_log("Tax Engine", f"Calculated New Regime Tax ({money(new_regime_est.net_tax)}) and Old Regime Tax ({money(old_regime_est.net_tax)})", "INFO")

        # Metrics Overview
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross Income", money(total_gross))
        c2.metric("New Regime Tax", money(new_regime_est.net_tax), delta=f"Effective: {new_regime_est.effective_rate:.1%}")
        c3.metric("Old Regime Tax", money(old_regime_est.net_tax), delta=f"Effective: {old_regime_est.effective_rate:.1%}")

        tax_diff = abs(new_regime_est.net_tax - old_regime_est.net_tax)
        best_regime = "New Regime" if new_regime_est.net_tax <= old_regime_est.net_tax else "Old Regime"
        c4.metric("Optimal Choice", best_regime, delta=f"Saves: {money(tax_diff)}")

        st.plotly_chart(build_regime_comparison_chart(new_regime_est, old_regime_est), use_container_width=True)

        st.markdown('<div class="status-box info">', unsafe_allow_html=True)
        st.markdown("#### Gemini AI Strategy Insights")
        st.markdown(ai_recommend_regime(new_regime_est, old_regime_est, total_gross))
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Line-by-Line Breakdown")
        comp_df = pd.DataFrame(
            [
                ("Gross Annual Income", money(new_regime_est.gross_income), money(old_regime_est.gross_income)),
                ("Standard Deduction", money(new_regime_est.deductions_applied), money(50000.0 if profile["is_salaried"] else 0.0)),
                ("Chapter VI-A Deductions", "N/A", money(old_regime_est.deductions_applied - (50000.0 if profile["is_salaried"] else 0.0))),
                ("Net Taxable Income", money(new_regime_est.taxable_income), money(old_regime_est.taxable_income)),
                ("Gross Slab Tax", money(new_regime_est.gross_tax), money(old_regime_est.gross_tax)),
                ("Section 87A Rebate", money(-new_regime_est.rebate_87a), money(-old_regime_est.rebate_87a)),
                ("Cess (4%)", money(new_regime_est.cess), money(old_regime_est.cess)),
                ("Total Net Tax Liability", money(new_regime_est.net_tax), money(old_regime_est.net_tax)),
                ("Prepaid Tax / TDS", money(-prepaid), money(-prepaid)),
                ("Balance Payable", money(new_regime_est.balance_payable), money(old_regime_est.balance_payable)),
            ],
            columns=["Line Item", "New Tax Regime (u/s 115BAC)", "Old Tax Regime"],
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 3: CA AI Workbench
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("3. CA AI Inspection Workbench")
        st.caption("Ask queries regarding Section 40A(3) disallowance risks, GST reconciliation, or unusual transaction spikes.")

        audit_query = st.text_input(
            "Enter Audit Query",
            placeholder="e.g., Are there any disallowance risks u/s 40A(3) or high cash expenses?",
            key="wb_audit_input",
        )

        if st.button("Execute Audit Inspection", key="btn_wb_audit", use_container_width=False):
            with st.spinner("Analyzing ledger parameters with Gemini 2.5 Flash..."):
                audit_res = ai_audit_ledger(transactions, audit_query)
            st.markdown(f'<div class="status-box info">{escape(audit_res)}</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 4: Provenance Audit Log (Execution Pipeline)
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("4. Provenance Audit Log & Execution Pipeline")
        st.caption("Step-by-step transparency log showing how data was processed and computed.")

        if "audit_trail" in st.session_state and st.session_state["audit_trail"]:
            audit_df = pd.DataFrame(st.session_state["audit_trail"])
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
        else:
            st.info("No execution events logged yet. Perform actions to view live audit entries.")

    # -------------------------------------------------------------------------
    # TAB 5: Handoff & PDF Export
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("5. Document Checklist & Final PDF Report Export")

        checked = [item for item in REQUIRED_CHECKLIST if st.checkbox(item, key=f"t5_chk_{item}")]
        st.progress(len(checked) / len(REQUIRED_CHECKLIST))

        st.divider()

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            # Generate PDF dynamically on button click to prevent premature encoding errors
            pdf_data = generate_pdf_summary(
                profile=profile,
                new_est=new_regime_est,
                old_est=old_regime_est,
                checked_items=checked,
            )
            st.download_button(
                label="📄 Download Tax Summary Report (PDF)",
                data=pdf_data,
                file_name=f"Tax_Summary_{profile['taxpayer_name'].replace(' ', '_')}_{ASSESSMENT_YEAR}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with col_dl2:
            packet_data = {
                "generated_on": date.today().isoformat(),
                "assessment_year": ASSESSMENT_YEAR,
                "financial_year": FINANCIAL_YEAR,
                "profile": profile,
                "new_regime_tax": new_regime_est.net_tax,
                "old_regime_tax": old_regime_est.net_tax,
                "checked_items": checked,
            }
            st.download_button(
                label="💾 Download Handoff Data (JSON)",
                data=json.dumps(packet_data, indent=2),
                file_name=f"tax_summary_{profile['taxpayer_name']}.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
