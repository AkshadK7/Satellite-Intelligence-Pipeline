# Data Engineer Assignment – Satellite Intelligence
**Akshad Kolhatkar | Carnot Technologies**

---

## Repository Layout

```
├── notebook.ipynb                      ← main notebook (Tasks 1–4 + Correlation + Forecasting)
├── script
    ├──app.py                           ← standalone Python script (full pipeline end-to-end)
    ├── requirements.txt                ← Python dependencies
├── cleaned_parcel_timeseries.csv       ← Task 2 output
├── parcel_metadata.csv                 ← input (unchanged)
├── parcel_readings.csv                 ← input (unchanged)
└── README.md                           ← this file
```

---

## Approach & AI Tool Usage

I used **Claude (claude.ai)** to assist with:
- Drafting the initial data quality audit after manually exploring the CSVs in Python
- Suggesting the flexible multi-format date parser pattern
- Structuring the README, notebook layout, and forecasting section

All analytical decisions — which issues to repair vs. drop, join strategy, analysis windows, model selection rationale — were made after hands-on exploration of the raw data. All code was written and verified by running end-to-end.

---

## Setup

Open `notebook.ipynb` in Jupyter

Script :

```bash
cd script
pip install -r requirements.txt
python app.py        # standalone script
# or open notebook.ipynb in Jupyter
```

---

## Task 1 — Data Quality Audit

### Inputs

| File | Rows | Columns |
|------|------|---------|
| `parcel_metadata.csv` | 28 | 5 |
| `parcel_readings.csv` | 3,447 | 6 |

### Issues Found

| # | Issue | Prevalence | Decision | Justification |
|---|-------|-----------|----------|---------------|
| 1 | **Mixed date formats** in `parcel_readings.date` — three coexisting formats: `YYYY-MM-DD` (70%), `DD/MM/YYYY` (20%), `DD-Mon-YYYY` (10%) | ~30% of rows | **Repair** — per-row format detection via `parse_date_flexible()` | Dates are the primary join key; discarding 30% of rows is unacceptable. A single `infer_datetime_format` fails on mixed-format columns |
| 2 | **`sensor_status` case & whitespace inconsistency** — `OK`, `ok`, ` OK`, `Error`, `ERROR`, `error` | ~10% of rows | **Repair** — `.str.strip().str.upper()` | Trivial normalisation; semantic meaning is unambiguous |
| 3 | **`ndvi_value` outside valid range [-1, 1]** — min: -1.976, max: 1.997 | 104 rows / 3% | **Flag & nullify** — set to `NaN`, add `ndvi_out_of_range` boolean | Physically impossible; imputing fabricates signal. Flag preserves the row for temperature/rainfall analysis |
| 4 | **`sensor_status` NaN** | 137 rows / 4% | **Flag** — fill with `UNKNOWN`, treat as non-OK | Cannot assume sensor health; safer to exclude from NDVI analysis |
| 5 | **Duplicate `parcel_id × date` rows** | 8 duplicate keys (16 rows) | **De-duplicate** — keep first | Re-ingestion artefacts; keeping first is deterministic |
| 6 | **`PARCEL_098`, `PARCEL_099` in readings, absent from metadata** | 2 parcels / ~70 rows | **Drop** | No crop/mill info; would produce all-null metadata columns |
| 7 | **`PARCEL_050`, `PARCEL_051`, `PARCEL_052` in metadata, absent from readings** | 3 parcels | **Retain in metadata only** | Metadata is valid; gap in readings is a collection issue, not a data error |

---

## Task 2 — Clean Pipeline

Steps:

1. **Ingest** both CSVs with `pd.read_csv`
2. **Clean metadata** — parse `sowing_date` to `datetime`, lowercase `crop_type`
3. **Clean readings:**
   - Multi-format date parsing: tries `%Y-%m-%d` → `%d/%m/%Y` → `%d-%b-%Y` per row
   - `sensor_status` normalised to `OK` / `ERROR` / `UNKNOWN`
   - `ndvi_out_of_range` flag added; out-of-range values nullified
   - 8 duplicate parcel×date rows removed (keep first)
   - 2 unmatched parcels dropped
4. **Left-join** readings onto metadata on `parcel_id`
5. **Sort** by `parcel_id`, `date` and write `cleaned_parcel_timeseries.csv`

**Output:** 3,399 rows × 11 columns

---

## Task 3 — Quick Analysis

### NDVI Before / After Sowing by Crop Type

*Filtered to `sensor_status == 'OK'` and non-null `ndvi_value` only.*

| crop_type | mean_ndvi_before | mean_ndvi_after | n_parcels |
|-----------|-----------------|----------------|-----------|
| soybean   | 0.1706 | 0.3126 | 4 |
| sugarcane | 0.1775 | 0.3361 | 19 |
| wheat     | 0.1761 | 0.3101 | 2 |

### Interpretation

All three crop types show a clear rise in mean NDVI after sowing relative to the 30 days prior — the expected agronomic signal as bare or recently tilled soil transitions to early canopy cover during germination and emergence.

Sugarcane shows the largest absolute NDVI gain (~+0.16), consistent with its rapid canopy closure and dense biomass accumulation under tropical conditions; it also has the largest parcel sample (n=19), making the estimate most reliable.

Wheat and soybean show smaller but positive post-sowing NDVI increases. Wheat's result is directionally consistent with the other crops, but the sample of only 2 parcels warrants caution before drawing crop-level conclusions.

---

## Task 4 — Production-Readiness Reflection

### Three Things I Would Change at 100× Scale

**1. Replace pandas full-load with partitioned Parquet + Polars/PySpark**

At 100×, a monolithic CSV read into pandas will hit memory limits. Parquet partitioned by `parcel_id` or `year-month`, with lazy evaluation in Polars or PySpark, means you never load more data than a given operation requires. The date-parse step — currently O(n) Python `.apply()` — would move to vectorised format-detection.

**2. Data contracts, not magic numbers**

The valid NDVI range `[-1, 1]` is currently a literal in a notebook cell. In production that belongs in a Great Expectations suite — versioned, testable, emitting structured failure reports to a monitoring dashboard rather than silently producing NaN values.

**3. Incremental / idempotent processing**

A daily job shouldn't re-clean all historical data. Track `max_processed_date` per parcel in a state table, load only new partitions, and upsert. Reruns after failures are safe; no double-counting.

### What I Would Monitor in Production

| Metric | Alert Condition |
|--------|----------------|
| Row count per parcel per day | Drop below rolling 7-day p10 → sensor outage |
| % `sensor_status == ERROR` by parcel, 7-day rolling | > 50% for any parcel → field team escalation |
| % `ndvi_out_of_range` by date | Spike > 10% → bad satellite pass or processing bug |
| Pipeline freshness: `max(date)` | < today − 2 days → ingestion failure |
| Schema drift on `sensor_status`, `crop_type` | New unexpected values → upstream format change |
| Join yield | % readings matched to metadata < 98% → metadata lag |
| Satellite anomaly detection | Correlated error-rate spikes across multiple parcels simultaneously → automated alert |

### Most Likely Thing to Silently Break

**The date format parser.** A new upstream provider, locale change, or timezone shift can produce date strings that parse without error but are semantically wrong — `04/05/2026` read as 4 May instead of 5 April. The pipeline doesn't crash; NDVI values look plausible; but every time-relative window (before/after sowing, rolling means) is now shifted by weeks or months with no visible signal in monitoring.

Fix: a post-parse assertion `max(parsed_date) <= today + 1 day` and a distribution check that the date histogram matches expected field-season patterns.

---

## Correlation & Environmental Analysis

Six plots examining how NDVI, temperature, and rainfall relate to each other across the dataset:

| Plot | What it shows |
|------|---------------|
| Individual effects | How each variable independently drives vegetation health (derived from actual binned sensor data) |
| Monthly time-series | Seasonal co-movement of NDVI, temperature, and total rainfall |
| Correlation heatmap | Pairwise linear correlation strength between all three variables |
| Scatter plot matrix | Full variable-pair relationships coloured by crop type |
| Rolling average | 7-day smoothed NDVI trend removing day-to-day noise |
| Anomaly plot | Monthly NDVI deviation from dataset mean — flags unusual periods |

---

## NDVI Forecasting

Six models across two tiers applied to the 5-month monthly NDVI series (Jan–May 2026).

### Tier 1 — Baseline Models

| Model | Type | Notes |
|-------|------|-------|
| SARIMA | Statistical | Trend-following; seasonal component omitted (5 months insufficient for period=12) |
| ETS | Statistical | Additive-trend extrapolation; robust baseline |
| Random Forest | ML | Lag-feature baseline; insufficient training data to generalise |
| XGBoost | ML | Same caveat; included for structural comparison |

### Tier 2 — Better-Fit Models

| Model | Type | Why better for this use-case |
|-------|------|------------------------------|
| **Prophet** | Statistical/ML hybrid | Natively handles irregular seasonality; **sowing dates passed as changepoints** — architecturally the strongest fit |
| **Ridge Regression** | Regularised linear ML | Uses agronomic features (temp, rainfall, cyclical month, lag NDVI); interpretable coefficients reveal which field variable drives NDVI |

> **Dataset caveat:** 5 months is far below the 24+ months needed for ML models. All results are methodological demonstrations. SARIMA, ETS, and Prophet are the only credible forecasters at this scale. With 2–3 years of history, Prophet + Ridge would be the recommended production starting point.

