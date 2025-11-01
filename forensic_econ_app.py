"""Forensic Economics Web App (Enhanced)
----------------------------------------
This Streamlit application implements the forensic economics workflow as
described by the project requirements. It collects personal and
case‑specific information about a decedent, runs a series of
computational agents to estimate economic loss, shows a live status
dashboard for each agent, and produces a formatted Excel report.

Key features of this enhanced version:

* **Live links for assumptions** – Users can specify a Federal Reserve
  (FRED) 1‑year Treasury CSV URL and work‑life expectancy (WLE) CSV or
  PDF URLs. The application will attempt to fetch the latest discount
  rate and WLE values from these sources automatically. If fetching
  fails or the fields are left blank, the user‑provided overrides are
  used instead.
* **Agent pipeline with progress updates** – Each computational step
  (e.g. person validation, discount rate lookup, WLE lookup, wage growth,
  present value calculation, Excel generation) emits progress events
  which are displayed in real‑time in the UI. This mirrors the
  “agent flow” dashboard shown in the example screenshots.
* **Flexible assumptions** – Users may override the annual wage
  growth, discount rate, work‑life expectancy and retirement age. If
  these overrides are left blank, the app will attempt to compute
  sensible defaults from the provided links or fall back to safe
  defaults.

The app requires the following third‑party packages:
    streamlit, pandas, requests, pdfplumber (optional for PDF parsing)
You can install them via pip:
    pip install streamlit pandas requests pdfplumber openpyxl xlsxwriter

Run with:
    streamlit run forensic_econ_app.py
"""

from __future__ import annotations

import datetime as dt
import io
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Literal, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import requests
except ImportError:
    requests = None  # type: ignore

# Default links for external data sources. These are used as placeholder
# values in the UI and may be overridden by the user.
FRED_1Y_CSV_DEFAULT = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1"
WLE_CSV_DEFAULT: str = ""
WLE_PDF_DEFAULT: str = ""


###############################################################################
# Data structures
###############################################################################

@dataclass
class PersonData:
    """Container for the person’s demographic and employment details."""
    first_name: str
    last_name: str
    sex: str  # 'male' or 'female'
    dob: dt.date
    dod: Optional[dt.date]
    occupation: str
    home_state: str
    education_level: str
    employment_status: str  # 'active' or 'inactive'
    valuation_date: dt.date
    base_salary: float


###############################################################################
# Utility functions
###############################################################################

def years_between(d1: dt.date, d2: dt.date) -> float:
    """Return fractional years between two dates."""
    return (d2 - d1).days / 365.25


def age_on(dob: dt.date, on_date: dt.date) -> float:
    """Compute age in years on a specific date."""
    return years_between(dob, on_date)


def portion_first_year_from(d: dt.date) -> float:
    """Return the portion of the calendar year remaining starting from date d.

    Includes the start day approximately. For example, if d is September 15,
    the result is roughly 0.29 (remainder of September through December)."""
    year_end = dt.date(d.year, 12, 31)
    return max(0.0, years_between(d, year_end) + (1 / 365.25))


def to_date_from_mm_yy(month: int, year: int) -> dt.date:
    """Convert a month/year pair into a date using the 15th of the month.

    This helper simplifies converting user inputs like "05/1983" into a
    single date. The 15th is used to avoid issues with months of varying
    lengths and matches the classroom examples."""
    return dt.date(int(year), int(month), min(15, (dt.date(int(year), int(month), 1) + dt.timedelta(days=27)).day))


###############################################################################
# External data agents
###############################################################################

class AutoAssumptionsAgent:
    """Agent to fetch assumptions like discount rates from external links."""

    def fetch_fred_1y_discount(self, csv_url: str) -> Optional[float]:
        """Fetch the latest 1‑year Treasury rate from a FRED CSV.

        Returns the rate as a decimal (e.g. 0.045 for 4.5%), or None if
        fetching fails. The CSV should have at least two columns with the
        second column representing the series values in percent."""
        if not requests:
            return None
        try:
            response = requests.get(csv_url, timeout=20)
            response.raise_for_status()
            df = pd.read_csv(io.BytesIO(response.content))
            series = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna()
            if series.empty:
                return None
            latest_percent = float(series.iloc[-1])
            return latest_percent / 100.0
        except Exception:
            return None


class SkoogWLEAgent:
    """Agent to resolve work‑life expectancy (WLE) from a CSV or PDF."""

    def __init__(self, csv_url: Optional[str] = None, pdf_url: Optional[str] = None) -> None:
        self.csv_url = csv_url or ""
        self.pdf_url = pdf_url or ""
        self._csv_df: Optional[pd.DataFrame] = None
        if self.csv_url:
            try:
                if requests:
                    resp = requests.get(self.csv_url, timeout=30)
                    resp.raise_for_status()
                    df = pd.read_csv(io.BytesIO(resp.content))
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    self._csv_df = df
            except Exception:
                self._csv_df = None

    def wle_from_csv(self, age: int, sex: str, education: Optional[str], initial_state: Optional[str]) -> Optional[float]:
        """Try to get WLE from the loaded CSV based on age and demographics."""
        df = self._csv_df
        if df is None:
            return None
        # Required columns: age, sex, mean_wle; optional: education, initial_state
        for col in ["age", "sex", "mean_wle"]:
            if col not in df.columns:
                return None
        df = df.copy()
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df = df.dropna(subset=["age"])
        df = df.loc[(df["age"].round().astype(int) == int(round(age))) & (df["sex"].astype(str).str.lower() == sex.lower())]
        if education and "education" in df.columns:
            maybe = df.loc[df["education"].astype(str).str.lower() == education.lower()]
            if not maybe.empty:
                df = maybe
        if initial_state and "initial_state" in df.columns:
            maybe = df.loc[df["initial_state"].astype(str).str.lower() == initial_state.lower()]
            if not maybe.empty:
                df = maybe
        if df.empty:
            return None
        try:
            val = pd.to_numeric(df.iloc[0]["mean_wle"], errors="coerce")
            if pd.notnull(val):
                return float(val)
        except Exception:
            return None
        return None

    def wle_from_pdf(self, age: int, sex: str) -> Optional[float]:
        """Best‑effort extraction of WLE from a PDF table using pdfplumber.

        This function attempts to load the PDF at self.pdf_url and search
        for tables containing age and WLE columns. It returns the WLE value
        matching the provided age and sex if found. If pdfplumber is not
        installed or parsing fails, the function returns None."""
        if not self.pdf_url:
            return None
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return None
        if not requests:
            return None
        try:
            resp = requests.get(self.pdf_url, timeout=45)
            resp.raise_for_status()
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables() or []
                    for t in tables:
                        df = pd.DataFrame(t)
                        # Find header row containing 'age'
                        header_row = None
                        for i, row in enumerate(df.values.tolist()):
                            if any(isinstance(x, str) and "age" in x.lower() for x in row):
                                header_row = i
                                break
                        if header_row is None:
                            continue
                        header = [str(x).strip().lower() for x in df.iloc[header_row].tolist()]
                        data = df.iloc[header_row + 1:].reset_index(drop=True)
                        data.columns = header
                        if not any("age" in c for c in data.columns):
                            continue
                        data["__age__"] = pd.to_numeric(data[[c for c in data.columns if "age" in c][0]], errors="coerce")
                        wle_cols = [c for c in data.columns if ("mean" in c and "wle" in c) or ("work" in c and "mean" in c)]
                        if not wle_cols:
                            wle_cols = [c for c in data.columns if "wle" in c or "work-life" in c or "worklife" in c]
                        if not wle_cols:
                            continue
                        idx = (data["__age__"] - age).abs().argsort()[:1]
                        match = data.loc[idx]
                        for c in data.columns:
                            if "sex" in c:
                                tmp = match[match[c].astype(str).str.lower().str.contains(sex.lower())]
                                if not tmp.empty:
                                    match = tmp
                        if match.empty:
                            continue
                        val = pd.to_numeric(match.iloc[0][wle_cols[0]], errors="coerce")
                        if pd.notnull(val):
                            return float(val)
        except Exception:
            return None
        return None

    def resolve(self, dob: dt.date, start: dt.date, sex: str, education: str,
                initial_state: str, override: Optional[float], retire_cap: Optional[float]) -> float:
        """Determine WLE considering override, CSV/PDF sources, and retirement cap."""
        # Use override if provided
        if override is not None:
            wle = float(override)
        else:
            age = int(round(age_on(dob, start)))
            wle = self.wle_from_csv(age, sex, education, initial_state)
            if wle is None:
                wle = self.wle_from_pdf(age, sex)
            if wle is None:
                wle = 30.0  # fallback safe default
        # Apply retirement cap if provided
        if retire_cap is not None:
            current_age = age_on(dob, start)
            wle = min(wle, max(0.0, retire_cap - current_age))
        return float(max(wle, 0.0))


###############################################################################
# Economic calculation functions
###############################################################################

def portions_from_years(total_years: float, first_year_portion: float) -> List[float]:
    """Generate a list of yearly portions from a total number of years.

    The first element is an explicit portion (for the first, possibly
    fractional year), followed by full years (1.0) and a final fraction if
    needed to match the total. Negative or zero totals return an empty list."""
    if total_years <= 0:
        return []
    portions = []
    # First year portion cannot exceed total years
    first = min(first_year_portion, total_years)
    portions.append(round(first, 2))
    remaining = max(total_years - first, 0.0)
    full = int(math.floor(remaining))
    portions.extend([1.0] * full)
    final = round(remaining - full, 2)
    if final > 1e-8:
        portions.append(final)
    return portions


def build_schedule(person: PersonData, growth: float, disc_rate: float, wle_years: float) -> pd.DataFrame:
    """Construct the loss schedule DataFrame.

    Pre‑valuation years are assigned a discount factor of 1.0. The valuation
    year uses t=1. Subsequent years increment t. The schedule includes
    age, start date, year number, portion of year, full year value, actual
    value, cumulative value, discount factor, present value, and cumulative
    present value."""
    # Determine the starting date: if deceased, use DOD; otherwise use valuation date
    start = person.dod if person.dod else person.valuation_date
    first_portion = portion_first_year_from(start) if person.dod else 1.0
    # Generate portion list
    portions = portions_from_years(wle_years, first_portion)
    rows = []
    if not portions:
        return pd.DataFrame(columns=[
            "Age", "Start Date", "Year Number", "Portion of Year", "Full Year Value",
            "Actual Value", "Cumulative Value", "Discount Factor", "Present Value",
            "Cumulative Present Value"
        ])
    start_age = age_on(person.dob, start)
    valuation_year = person.valuation_date.year
    cumulative_actual = 0.0
    cumulative_pv = 0.0
    for idx, portion in enumerate(portions, start=1):
        year = start.year + (idx - 1)
        # Discount factor
        if year < valuation_year:
            dfactor = 1.0
        else:
            t = (year - valuation_year) + 1
            dfactor = 1.0 / ((1.0 + disc_rate) ** t)
        full_val = person.base_salary * ((1.0 + growth) ** (idx - 1))
        actual_val = full_val * portion
        pv = actual_val * dfactor
        cumulative_actual += actual_val
        cumulative_pv += pv
        rows.append({
            "Age": round(start_age + (idx - 1), 2),
            "Start Date": year,
            "Year Number": float(idx),
            "Portion of Year": round(portion, 2),
            "Full Year Value": round(full_val, 2),
            "Actual Value": round(actual_val, 2),
            "Cumulative Value": round(cumulative_actual, 2),
            "Discount Factor": round(dfactor, 6),
            "Present Value": round(pv, 2),
            "Cumulative Present Value": round(cumulative_pv, 2)
        })
    return pd.DataFrame(rows)


def make_excel_bytes(df: pd.DataFrame, person: PersonData, growth: float, disc: float) -> bytes:
    """Create an Excel file in memory with Summary and LossSchedule sheets.

    This function writes a summary sheet containing the person’s name,
    base salary, discount rate, growth rate, total PV and valuation date.
    The loss schedule is written on a separate sheet. Returns the raw
    bytes of the file so that Streamlit’s download button can serve it.
    """
    bio = io.BytesIO()
    try:
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            # Summary sheet
            summary = pd.DataFrame({
                "Item": [
                    "Name", "Item", "Base Value", "Discount rate", "Annual growth rate",
                    "Cumulative Present Value", "Present Value Month", "Present Value Year"
                ],
                "Value": [
                    f"{person.first_name} {person.last_name}",
                    "Earnings Loss (More Conservative)",
                    person.base_salary,
                    disc,
                    growth,
                    float(df["Present Value"].sum()) if not df.empty else 0.0,
                    person.valuation_date.month,
                    person.valuation_date.year,
                ]
            })
            summary.to_excel(writer, index=False, sheet_name="Summary")
            wb = writer.book
            ws = writer.sheets["Summary"]
            money_fmt = wb.add_format({"num_format": "$#,##0"})
            pct_fmt = wb.add_format({"num_format": "0.00%"})
            bold_fmt = wb.add_format({"bold": True})
            ws.set_column(0, 0, 30, bold_fmt)
            ws.set_column(1, 1, 22)
            # Apply number formatting on money and percent fields
            ws.write(2, 1, person.base_salary, money_fmt)
            ws.write(3, 1, disc, pct_fmt)
            ws.write(4, 1, growth, pct_fmt)
            ws.write(5, 1, float(df["Present Value"].sum()) if not df.empty else 0.0, money_fmt)
            # Loss schedule sheet
            df.to_excel(writer, index=False, sheet_name="LossSchedule")
            ws2 = writer.sheets["LossSchedule"]
            header_fmt = wb.add_format({"bold": True, "bg_color": "#EFEFEF"})
            money_fmt2 = wb.add_format({"num_format": "$#,##0"})
            disc_fmt = wb.add_format({"num_format": "0.000000"})
            yellow_fmt = wb.add_format({"bg_color": "#FFF2CC"})
            for j, col in enumerate(df.columns):
                ws2.write(0, j, col, header_fmt)
                width = max(12, min(28, max(len(col) + 2, int(df[col].astype(str).str.len().quantile(0.95)) + 2)))
                ws2.set_column(j, j, width)
            # Highlight Age column
            ws2.set_column(0, 0, 10, yellow_fmt)
            # Apply money formatting to currency columns
            for nm in ["Full Year Value", "Actual Value", "Present Value", "Cumulative Value", "Cumulative Present Value"]:
                if nm in df.columns:
                    idx = list(df.columns).index(nm)
                    ws2.set_column(idx, idx, None, money_fmt2)
            # Apply discount factor formatting
            if "Discount Factor" in df.columns:
                idx = list(df.columns).index("Discount Factor")
                ws2.set_column(idx, idx, None, disc_fmt)
    except Exception:
        # Fallback to openpyxl if xlsxwriter is not available
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="LossSchedule")
    return bio.getvalue()


###############################################################################
# Agent event pipeline
###############################################################################

AgentState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]


def agent_event(agent: str, state: AgentState, message: str = "", payload: Optional[dict] = None) -> Dict[str, object]:
    """Create a structured event dictionary for pipeline progress."""
    return {"agent": agent, "state": state, "message": message, "payload": payload or {}}


def run_pipeline_with_events(
    person: PersonData,
    fred_csv_url: str,
    wle_csv_url: str,
    wle_pdf_url: str,
    growth_override: Optional[float],
    discount_override: Optional[float],
    wle_override_years: Optional[float],
    retire_cap: Optional[float]
) -> Generator[Dict[str, object], None, Dict[str, object]]:
    """Execute the analysis pipeline, emitting events for each stage.

    This generator yields dictionaries representing progress of each agent. At
    the end, it returns a dictionary containing the DataFrame, Excel bytes
    and the resolved assumptions. If any step fails, it returns early with
    an error payload.
    """
    # 1) Validate person data
    yield agent_event("Person Investigation", "RUNNING", "Validating inputs…")
    if person.base_salary <= 0:
        yield agent_event("Person Investigation", "FAILED", "Base salary must be greater than zero.")
        return {"error": "Invalid salary"}
    time.sleep(0.2)
    yield agent_event(
        "Person Investigation",
        "COMPLETED",
        f"Validated {person.first_name} {person.last_name}"
    )

    # 2) Fetch discount rate from FRED if override not provided
    yield agent_event("Federal Reserve", "RUNNING", "Fetching discount rate…")
    auto = AutoAssumptionsAgent()
    if discount_override is not None:
        discount_rate = float(discount_override)
        disc_msg = f"Using override discount rate = {discount_rate:.4%}"
    else:
        discount_rate = auto.fetch_fred_1y_discount(fred_csv_url)
        if discount_rate is None:
            discount_rate = 0.045  # fallback to 4.5%
            disc_msg = "Failed to fetch; defaulting to 4.5%"
        else:
            disc_msg = f"Fetched discount rate = {discount_rate:.4%}"
    time.sleep(0.2)
    yield agent_event("Federal Reserve", "COMPLETED", disc_msg, {"discount_rate": discount_rate})

    # 3) Resolve WLE from CSV/PDF or override
    yield agent_event("Skoog Table", "RUNNING", "Resolving work‑life expectancy…")
    skoog = SkoogWLEAgent(csv_url=wle_csv_url, pdf_url=wle_pdf_url)
    start = person.dod if person.dod else person.valuation_date
    wle_years = skoog.resolve(
        dob=person.dob,
        start=start,
        sex=person.sex,
        education=person.education_level,
        initial_state=person.employment_status,
        override=wle_override_years,
        retire_cap=retire_cap
    )
    wle_msg = f"Work‑life expectancy = {wle_years:.2f} years"
    time.sleep(0.2)
    yield agent_event("Skoog Table", "COMPLETED", wle_msg, {"wle_years": wle_years})

    # 4) Determine annual growth rate
    yield agent_event("Annual Growth", "RUNNING", "Determining wage growth…")
    if growth_override is not None:
        growth_rate = float(growth_override)
        gr_msg = f"Using override growth = {growth_rate:.2%}"
    else:
        growth_rate = 0.0234  # default growth = 2.34%
        gr_msg = "Default growth rate = 2.34%"
    time.sleep(0.2)
    yield agent_event("Annual Growth", "COMPLETED", gr_msg, {"growth": growth_rate})

    # 5) Build schedule and compute present value
    yield agent_event("Present Value", "RUNNING", "Building loss schedule…")
    df = build_schedule(person, growth=growth_rate, disc_rate=discount_rate, wle_years=wle_years)
    total_pv = float(df["Present Value"].sum()) if not df.empty else 0.0
    time.sleep(0.2)
    yield agent_event(
        "Present Value",
        "COMPLETED",
        f"Schedule created with {len(df)} rows; PV = ${total_pv:,.0f}",
        {"df": df, "total_pv": total_pv}
    )

    # 6) Generate Excel report
    yield agent_event("Excel Report", "RUNNING", "Generating Excel report…")
    excel_bytes = make_excel_bytes(df, person, growth_rate, discount_rate)
    time.sleep(0.2)
    yield agent_event(
        "Excel Report",
        "COMPLETED",
        "Report created",
        {"excel_bytes": excel_bytes}
    )

    return {
        "df": df,
        "excel_bytes": excel_bytes,
        "discount_rate": discount_rate,
        "growth": growth_rate,
        "wle_years": wle_years
    }


###############################################################################
# Streamlit user interface
###############################################################################

def parse_mm_yyyy(s: str) -> Tuple[int, int]:
    """Parse a month/year string (e.g. '05/1983') into (month, year)."""
    s = s.strip()
    if not s:
        raise ValueError("Empty date string")
    parts = s.split("/")
    if len(parts) != 2:
        raise ValueError("Date must be in mm/yyyy format")
    m, y = parts
    return int(m), int(y)


def main() -> None:
    st.set_page_config(page_title="Forensic Economics Project", page_icon="⚖️", layout="wide")
    st.markdown(
        "<h2 style='margin-bottom:0'>⚖️ Forensic Economics Project</h2>"
        "<div style='color:#6c757d'>Economic Loss Analysis of Person \u2022 Agent workflow with live updates</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Layout: form on left, assumptions on right
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader("Person Data Collection")
        c1, c2, c3 = st.columns(3)
        first_name = c1.text_input("First Name", "Jane")
        last_name = c2.text_input("Last Name", "Doe")
        sex = c3.selectbox("Gender", ["female", "male"], index=0)

        c4, c5, c6 = st.columns(3)
        dob_str = c4.text_input("Date of Birth (mm/yyyy)", "05/1983")
        dod_str = c5.text_input("Date of Death (mm/yyyy) (leave blank if alive)", "09/2023")
        val_str = c6.text_input("Valuation Date (mm/yyyy)", "01/2025")

        c7, c8, c9 = st.columns(3)
        salary_str = c7.text_input("Annual Salary (numbers only)", "86895")
        occupation = c8.text_input("Occupation", "Sales Manager")
        home_state = c9.text_input("State (e.g., CA)", "CA")

        c10, c11, c12 = st.columns(3)
        employment_status = c10.selectbox("Employment Status", ["active", "inactive"], index=0)
        education_level = c11.text_input("Education Level", "bachelor")
        # Placeholder
        c12.write("")

    with right:
        st.subheader("Assumptions & Links")
        fred_url = st.text_input("FRED 1Y Treasury CSV URL", FRED_1Y_CSV_DEFAULT)
        wle_csv_url = st.text_input("Work‑life Expectancy CSV URL", WLE_CSV_DEFAULT)
        wle_pdf_url = st.text_input("Work‑life Expectancy PDF URL", WLE_PDF_DEFAULT)
        growth_override_str = st.text_input("Override Annual Growth Rate (decimal)", "")
        discount_override_str = st.text_input("Override Discount Rate (decimal)", "")
        wle_override_str = st.text_input("Override WLE (years)", "")
        retire_cap_str = st.text_input("Retirement Age Cap (e.g., 67)", "")
        st.caption("Provide URLs for FRED (discount) and WLE. Leave blank to use overrides or defaults.")
        run = st.button("Start Analysis", type="primary")

    if run:
        try:
            mm_dob, yy_dob = parse_mm_yyyy(dob_str)
            dob_date = to_date_from_mm_yy(mm_dob, yy_dob)
            dod_date: Optional[dt.date] = None
            if dod_str.strip():
                mm_dod, yy_dod = parse_mm_yyyy(dod_str)
                dod_date = to_date_from_mm_yy(mm_dod, yy_dod)
            mm_val, yy_val = parse_mm_yyyy(val_str)
            val_date = to_date_from_mm_yy(mm_val, yy_val)
            base_salary = float(salary_str)
        except Exception as e:
            st.error(f"Invalid date or salary input: {e}")
            return

        # Parse overrides
        gr_override = float(growth_override_str) if growth_override_str.strip() else None
        dr_override = float(discount_override_str) if discount_override_str.strip() else None
        wle_override = float(wle_override_str) if wle_override_str.strip() else None
        retire_cap = float(retire_cap_str) if retire_cap_str.strip() else None

        # Build person object
        person = PersonData(
            first_name=first_name.strip() or "",
            last_name=last_name.strip() or "",
            sex=sex,
            dob=dob_date,
            dod=dod_date,
            occupation=occupation.strip() or "",
            home_state=home_state.strip() or "",
            education_level=education_level.strip() or "",
            employment_status=employment_status,
            valuation_date=val_date,
            base_salary=base_salary
        )

        # Prepare UI placeholders
        progress = st.progress(0.0)
        status_box = st.status("Starting analysis…", expanded=True)
        flow_placeholder = st.empty()
        flow_df = pd.DataFrame(columns=["Agent", "State", "Message"])
        completed_count = 0
        total_steps = 6  # total number of agent steps
        final_payload: Dict[str, object] = {}

        # Execute pipeline with events
        for ev in run_pipeline_with_events(
            person=person,
            fred_csv_url=fred_url,
            wle_csv_url=wle_csv_url,
            wle_pdf_url=wle_pdf_url,
            growth_override=gr_override,
            discount_override=dr_override,
            wle_override_years=wle_override,
            retire_cap=retire_cap,
        ):
            # Update status table
            flow_df = pd.concat([flow_df, pd.DataFrame([{"Agent": ev["agent"], "State": ev["state"], "Message": ev["message"]}])], ignore_index=True)
            flow_placeholder.dataframe(flow_df, use_container_width=True, height=360)
            # Update status box and progress bar
            if ev["state"] == "RUNNING":
                status_box.update(label=f"{ev['agent']}: Running…", state="running")
            elif ev["state"] == "COMPLETED":
                completed_count += 1
                progress.progress(completed_count / total_steps)
                status_box.update(label=f"{ev['agent']}: Completed – {ev['message']}", state="complete")
            elif ev["state"] == "FAILED":
                status_box.update(label=f"{ev['agent']}: Failed – {ev['message']}", state="error")
                break
            # Capture payload
            final_payload.update(ev.get("payload", {}))

        # Display results
        df = final_payload.get("df")
        excel_bytes = final_payload.get("excel_bytes")
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.success("Analysis completed successfully!")
            st.subheader("Loss Schedule")
            st.dataframe(df, use_container_width=True)
            if excel_bytes:
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_bytes,
                    file_name=f"{person.first_name}_{person.last_name}_loss_schedule.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        elif "error" in final_payload:
            st.error(final_payload.get("error", "An error occurred during analysis."))
        else:
            st.warning("No results to display. Please check your inputs.")


if __name__ == "__main__":
    main()
