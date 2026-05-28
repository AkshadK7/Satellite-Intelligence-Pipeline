"""
Satellite Intelligence Pipeline
Carnot Data Engineer Take-Home Assignment
-----------------------------------------
Run:  python satellite_pipeline.py
Outputs: cleaned_parcel_timeseries.csv
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — DATA QUALITY AUDIT
# ─────────────────────────────────────────────────────────────────────────────

meta_raw     = pd.read_csv('parcel_metadata.csv')
readings_raw = pd.read_csv('parcel_readings.csv')

print(f'metadata   : {meta_raw.shape[0]:>5} rows × {meta_raw.shape[1]} cols')
print(f'readings   : {readings_raw.shape[0]:>5} rows × {readings_raw.shape[1]} cols')

# Issue 1: Mixed date formats
def detect_date_fmt(d):
    if re.match(r'\d{4}-\d{2}-\d{2}$', str(d)):       return 'YYYY-MM-DD'
    if re.match(r'\d{2}/\d{2}/\d{4}$', str(d)):        return 'DD/MM/YYYY'
    if re.match(r'\d{2}-[A-Za-z]{3}-\d{4}$', str(d)):  return 'DD-Mon-YYYY'
    return f'OTHER: {d}'

fmt_counts = Counter(detect_date_fmt(d) for d in readings_raw['date'])
print('\nDate format distribution:')
for fmt, cnt in fmt_counts.most_common():
    print(f'  {fmt:<20} {cnt:>5} ({cnt/len(readings_raw)*100:.1f}%)')

# Issue 2: sensor_status
print('\nRaw sensor_status counts:')
print(readings_raw['sensor_status'].value_counts(dropna=False))

# Issue 3: NDVI range
ndvi_bad = readings_raw[(readings_raw['ndvi_value'] < -1) | (readings_raw['ndvi_value'] > 1)]
print(f'\nNDVI out-of-range: {len(ndvi_bad)} rows ({len(ndvi_bad)/len(readings_raw)*100:.1f}%)')

# Issue 4: null sensor_status
null_status = readings_raw['sensor_status'].isnull().sum()
print(f'Null sensor_status: {null_status} ({null_status/len(readings_raw)*100:.1f}%)')

# Issue 5: duplicates
dups = readings_raw[readings_raw.duplicated(subset=['parcel_id', 'date'], keep=False)]
print(f'Duplicate parcel×date rows: {len(dups)}')

# Issue 6 & 7: parcel ID mismatches
ids_readings = set(readings_raw['parcel_id'].unique())
ids_meta     = set(meta_raw['parcel_id'].unique())
print(f'Parcels in readings NOT in metadata: {ids_readings - ids_meta}')
print(f'Parcels in metadata NOT in readings: {ids_meta - ids_readings}')


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — CLEAN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

# 2.1 Clean metadata
meta = meta_raw.copy()
meta['sowing_date'] = pd.to_datetime(meta['sowing_date'], format='%Y-%m-%d')
meta['crop_type']   = meta['crop_type'].str.strip().str.lower()
print(f'\nClean metadata: {meta.shape}')

# 2.2 Clean readings
readings = readings_raw.copy()

def parse_date_flexible(d):
    d = str(d).strip()
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%b-%Y'):
        try:
            return pd.to_datetime(d, format=fmt)
        except ValueError:
            continue
    return pd.NaT

readings['date'] = readings['date'].apply(parse_date_flexible)
print(f'Date parse failures (NaT): {readings["date"].isnull().sum()}')

readings['sensor_status'] = (
    readings['sensor_status']
    .fillna('UNKNOWN')
    .str.strip()
    .str.upper()
)

readings['ndvi_out_of_range'] = (readings['ndvi_value'] < -1) | (readings['ndvi_value'] > 1)
readings.loc[readings['ndvi_out_of_range'], 'ndvi_value'] = np.nan
print(f'NDVI nullified (out-of-range): {readings["ndvi_out_of_range"].sum()}')

before = len(readings)
readings = readings.drop_duplicates(subset=['parcel_id', 'date'], keep='first')
print(f'Duplicates removed: {before - len(readings)} rows')

valid_parcels = set(meta['parcel_id'])
before = len(readings)
readings = readings[readings['parcel_id'].isin(valid_parcels)]
print(f'Rows dropped (no metadata): {before - len(readings)}')
print(f'Clean readings: {readings.shape}')

# 2.3 Join
timeseries = readings.merge(meta, on='parcel_id', how='left')
timeseries = timeseries.sort_values(['parcel_id', 'date']).reset_index(drop=True)
print(f'\nFinal timeseries: {timeseries.shape}')
print(f'Date range: {timeseries["date"].min().date()} → {timeseries["date"].max().date()}')

# 2.4 Write output
timeseries.to_csv('cleaned_parcel_timeseries.csv', index=False)
print('Written → cleaned_parcel_timeseries.csv')


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — QUICK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

ts = pd.read_csv('cleaned_parcel_timeseries.csv', parse_dates=['date', 'sowing_date'])

good = ts[
    (ts['sensor_status'] == 'OK') &
    ts['ndvi_value'].notna() &
    ts['sowing_date'].notna()
].copy()
good['days'] = (good['date'] - good['sowing_date']).dt.days

before_df = good[(good['days'] >= -30) & (good['days'] <  0)]
after_df  = good[(good['days'] >   0) & (good['days'] <= 30)]

summary = pd.DataFrame({
    'mean_ndvi_before': before_df.groupby('crop_type')['ndvi_value'].mean(),
    'mean_ndvi_after':  after_df.groupby('crop_type')['ndvi_value'].mean(),
    'n_parcels': (
        pd.concat([before_df[['crop_type','parcel_id']], after_df[['crop_type','parcel_id']]])
        .drop_duplicates()
        .groupby('crop_type')['parcel_id'].nunique()
    )
}).reset_index()

print('\n=== NDVI Before/After Sowing by Crop Type ===')
print(summary.to_string(index=False))

print("""
INTERPRETATION
All three crop types show a clear rise in mean NDVI after sowing relative to
the 30 days before. Sugarcane shows the largest absolute gain (~+0.16),
consistent with its fast canopy closure. Wheat and soybean show comparable
post-sowing increases; wheat's 2-parcel sample warrants caution.
""")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS (Task 3)
# ─────────────────────────────────────────────────────────────────────────────

CROP_COLORS = {'sugarcane': '#2ecc71', 'soybean': '#e67e22', 'wheat': '#3498db'}
plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11})

window = good[(good['days'] >= -30) & (good['days'] <= 30)].copy()
window['period'] = np.where(window['days'] < 0, 'Before Sowing', 'After Sowing')

# Plot 1: NDVI trajectory
traj = window.groupby(['crop_type','days'])['ndvi_value'].agg(['mean','sem']).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
for crop, grp in traj.groupby('crop_type'):
    grp = grp.sort_values('days')
    color = CROP_COLORS[crop]
    ax.plot(grp['days'], grp['mean'], color=color, linewidth=2.2, label=crop.capitalize())
    ax.fill_between(grp['days'], grp['mean']-grp['sem'], grp['mean']+grp['sem'], color=color, alpha=0.15)
ax.axvline(0, color='#e74c3c', linestyle='--', linewidth=1.4, label='Sowing day')
ax.axvspan(-30, 0, color='#f9f0f0', alpha=0.4, zorder=0)
ax.axvspan(0, 30, color='#f0f9f0', alpha=0.4, zorder=0)
ax.set_xlabel('Days relative to sowing date'); ax.set_ylabel('Mean NDVI')
ax.set_title('Mean NDVI trajectory ±30 days around sowing\n(shaded band = ±1 SEM)')
ax.legend(frameon=False); ax.set_xlim(-30, 30)
plt.tight_layout(); plt.savefig('plot1_ndvi_trajectory.png', bbox_inches='tight'); plt.show()

# Plot 2: Before/after bar chart
crops = summary['crop_type'].tolist()
x, w = np.arange(len(crops)), 0.35
fig, ax = plt.subplots(figsize=(8, 5))
bars_b = ax.bar(x - w/2, summary['mean_ndvi_before'], w, color=[CROP_COLORS[c] for c in crops], alpha=0.45, edgecolor='white', label='Before sowing')
bars_a = ax.bar(x + w/2, summary['mean_ndvi_after'],  w, color=[CROP_COLORS[c] for c in crops], alpha=1.0,  edgecolor='white', label='After sowing')
for bar in list(bars_b)+list(bars_a):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{c.capitalize()}\n(n={row.n_parcels})" for c, row in zip(crops, summary.itertuples())])
ax.set_ylabel('Mean NDVI'); ax.set_ylim(0, 0.5)
ax.set_title('Mean NDVI: 30 days before vs. after sowing'); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig('plot2_before_after_bar.png', bbox_inches='tight'); plt.show()

# Plot 3: Box + strip distribution
fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
for ax, crop in zip(axes, ['sugarcane', 'soybean', 'wheat']):
    subset = window[window['crop_type'] == crop]
    sns.boxplot(data=subset, x='period', y='ndvi_value', order=['Before Sowing','After Sowing'],
                palette={'Before Sowing':'#d5e8d4','After Sowing':CROP_COLORS[crop]}, width=0.45, ax=ax)
    sns.stripplot(data=subset, x='period', y='ndvi_value', order=['Before Sowing','After Sowing'],
                  color='#333', alpha=0.25, size=2.5, jitter=True, ax=ax)
    ax.set_title(f'{crop.capitalize()}  (n={subset["parcel_id"].nunique()})')
    ax.set_xlabel(''); ax.set_ylabel('NDVI' if crop=='sugarcane' else '')
fig.suptitle('NDVI distribution before vs. after sowing', fontsize=13, y=1.01)
plt.tight_layout(); plt.savefig('plot3_distribution.png', bbox_inches='tight'); plt.show()

# Plot 4: Heatmap (sugarcane)
pivot = window[window['crop_type']=='sugarcane'].pivot_table(index='parcel_id', columns='days', values='ndvi_value', aggfunc='mean')
pivot = pivot.loc[pivot.loc[:,pivot.columns>0].mean(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.8, interpolation='nearest')
day_cols = pivot.columns.tolist()
tick_pos = [i for i, d in enumerate(day_cols) if d % 5 == 0]
ax.set_xticks(tick_pos); ax.set_xticklabels([day_cols[i] for i in tick_pos], fontsize=9)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
sowing_x = day_cols.index(0)
ax.axvline(sowing_x-0.5, color='white', linestyle='--', linewidth=1.8)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label('NDVI')
ax.set_xlabel('Days relative to sowing'); ax.set_ylabel('Parcel ID')
ax.set_title('Per-parcel NDVI heatmap ±30 days — Sugarcane')
plt.tight_layout(); plt.savefig('plot4_heatmap.png', bbox_inches='tight'); plt.show()

# Plot 5: Combined time-series
ts['date'] = pd.to_datetime(ts['date'])
daily = ts[ts['sensor_status']=='OK'].groupby(['crop_type','date'])['ndvi_value'].mean().reset_index().sort_values('date')
fig, ax = plt.subplots(figsize=(14, 5))
for crop, grp in daily.groupby('crop_type'):
    color = CROP_COLORS[crop]
    ax.plot(grp['date'], grp['ndvi_value'], color=color, linewidth=1.2, alpha=0.4, label='_nolegend_')
    rolled = grp.set_index('date')['ndvi_value'].rolling('7D').mean()
    ax.plot(rolled.index, rolled.values, color=color, linewidth=2.2, label=crop.capitalize())
for crop, grp in ts.dropna(subset=['sowing_date']).groupby('crop_type'):
    for sow_date in grp['sowing_date'].unique():
        ax.axvline(pd.to_datetime(sow_date), color=CROP_COLORS[crop], linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Date'); ax.set_ylabel('Mean NDVI')
ax.set_title('Daily NDVI time-series by crop type\n(faint = daily mean, bold = 7-day rolling, dotted = sowing dates)')
ax.legend(frameon=False); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig('plot5_timeseries.png', bbox_inches='tight'); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CORRELATION & ENVIRONMENTAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

good_ok = ts[(ts['sensor_status']=='OK') & ts['ndvi_value'].notna() & ts['temperature_c'].notna() & ts['rainfall_mm'].notna()].copy()

good_ok['ndvi_bin'] = pd.cut(good_ok['ndvi_value'], bins=np.arange(0, 1.1, 0.1))
ndvi_bins   = good_ok.groupby('ndvi_bin', observed=True)['ndvi_value'].mean().values
good_ok['temp_bin'] = pd.cut(good_ok['temperature_c'], bins=range(10, 45, 5))
good_ok['rain_bin'] = pd.cut(good_ok['rainfall_mm'], bins=[0,2,4,6,8,10,12,14,16])
temp_centers  = good_ok.groupby('temp_bin', observed=True)['temperature_c'].mean().values
temp_ndvi_avg = good_ok.groupby('temp_bin', observed=True)['ndvi_value'].mean().values
rain_centers  = good_ok.groupby('rain_bin', observed=True)['rainfall_mm'].mean().values
rain_ndvi_avg = good_ok.groupby('rain_bin', observed=True)['ndvi_value'].mean().values

def to_health(arr):
    return (arr - arr.min()) / (arr.max() - arr.min()) * 90 + 5

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Individual Effects on Vegetation Health", fontsize=14, fontweight='bold')
axes[0].scatter(ndvi_bins, to_health(ndvi_bins), color='#1D9E75', s=80, zorder=3)
axes[0].plot(ndvi_bins, to_health(ndvi_bins), color='#1D9E75', linewidth=1.5, alpha=0.5)
axes[0].set_title("NDVI vs Vegetation Health"); axes[0].set_xlabel("NDVI"); axes[0].set_ylabel("Vegetation Health (%)")
axes[0].set_xlim(0,1); axes[0].set_ylim(0,100); axes[0].grid(True, alpha=0.2)
axes[1].plot(temp_centers, to_health(temp_ndvi_avg), color='#D85A30', linewidth=2, marker='o', markersize=5)
axes[1].fill_between(temp_centers, to_health(temp_ndvi_avg), alpha=0.08, color='#D85A30')
axes[1].set_title("Temperature vs Vegetation Health"); axes[1].set_xlabel("Temperature (°C)"); axes[1].set_ylabel("Vegetation Health (%)")
axes[1].set_ylim(0,100); axes[1].grid(True, alpha=0.2)
axes[2].plot(rain_centers, to_health(rain_ndvi_avg), color='#378ADD', linewidth=2, marker='o', markersize=5)
axes[2].fill_between(rain_centers, to_health(rain_ndvi_avg), alpha=0.08, color='#378ADD')
axes[2].set_title("Rainfall vs Vegetation Health"); axes[2].set_xlabel("Rainfall (mm/day)"); axes[2].set_ylabel("Vegetation Health (%)")
axes[2].set_ylim(0,100); axes[2].grid(True, alpha=0.2)
plt.tight_layout(); plt.savefig('plot6_individual_effects.png', bbox_inches='tight'); plt.show()

# Correlation heatmap
corr_df = ts[ts['sensor_status']=='OK'][['ndvi_value','temperature_c','rainfall_mm']].rename(
    columns={'ndvi_value':'NDVI','temperature_c':'Temperature','rainfall_mm':'Rainfall'}).dropna()
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='YlGn', vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax, cbar_kws={"shrink":0.8})
ax.set_title("Correlation Matrix — NDVI, Temperature & Rainfall", fontsize=12, fontweight='bold')
plt.tight_layout(); plt.savefig('plot7_correlation.png', bbox_inches='tight'); plt.show()

# Rolling average
daily_ndvi = ts[ts['sensor_status']=='OK'].groupby('date')['ndvi_value'].mean().sort_index().reset_index()
daily_ndvi['rolling_7d'] = daily_ndvi['ndvi_value'].rolling(window=7, center=True).mean()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily_ndvi['date'], daily_ndvi['ndvi_value'], color='#1D9E75', linewidth=1, alpha=0.35, label='Daily mean NDVI')
ax.plot(daily_ndvi['date'], daily_ndvi['rolling_7d'], color='#1D9E75', linewidth=2.5, label='7-day rolling avg')
ax.fill_between(daily_ndvi['date'], daily_ndvi['ndvi_value'], daily_ndvi['rolling_7d'], alpha=0.08, color='#1D9E75')
ax.set_ylabel("NDVI"); ax.set_ylim(0,1)
ax.set_title("NDVI Trend with 7-Day Rolling Average", fontweight='bold')
ax.legend(frameon=False); ax.grid(True, alpha=0.2)
plt.tight_layout(); plt.savefig('plot8_rolling.png', bbox_inches='tight'); plt.show()

# Anomaly plot
monthly_ndvi_anom = ts[ts['sensor_status']=='OK'].groupby(ts['date'].dt.to_period('M'))['ndvi_value'].mean().sort_index().reset_index()
monthly_ndvi_anom.columns = ['month','ndvi']
ndvi_mean_val = monthly_ndvi_anom['ndvi'].mean()
monthly_ndvi_anom['anomaly'] = monthly_ndvi_anom['ndvi'] - ndvi_mean_val
monthly_ndvi_anom['month_label'] = monthly_ndvi_anom['month'].dt.strftime('%b %Y')
anom_colors = ['#D85A30' if v < 0 else '#1D9E75' for v in monthly_ndvi_anom['anomaly']]
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(monthly_ndvi_anom['month_label'], monthly_ndvi_anom['anomaly'], color=anom_colors, edgecolor='white', linewidth=0.5)
ax.axhline(0, color='gray', linewidth=1, linestyle='--')
ax.set_ylabel("NDVI deviation from mean")
ax.set_title(f"Monthly NDVI Anomaly  (mean = {ndvi_mean_val:.3f})", fontweight='bold')
ax.legend(handles=[mpatches.Patch(color='#1D9E75', label='Above average'), mpatches.Patch(color='#D85A30', label='Below average')])
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout(); plt.savefig('plot9_anomaly.png', bbox_inches='tight'); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — NDVI FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet

# Build monthly series
monthly_ndvi = (
    ts[(ts['sensor_status']=='OK') & ts['ndvi_value'].notna()]
    .groupby(ts['date'].dt.to_period('M'))['ndvi_value']
    .mean().sort_index()
)
monthly_ndvi.index = monthly_ndvi.index.to_timestamp()
print(f'\nMonthly NDVI series ({len(monthly_ndvi)} months):')
print(monthly_ndvi.round(4).to_string())

# Train/test split
n_forecast = 2
train = monthly_ndvi.iloc[:-n_forecast]
test  = monthly_ndvi.iloc[-n_forecast:]

# SARIMA
sarima_model = SARIMAX(train, order=(1,0,1), enforce_stationarity=False, enforce_invertibility=False)
sarima_fit   = sarima_model.fit(disp=False)
sarima_pred  = sarima_fit.forecast(steps=n_forecast)
mae_sarima   = mean_absolute_error(test, sarima_pred)
rmse_sarima  = np.sqrt(mean_squared_error(test, sarima_pred))
print(f'SARIMA        — MAE: {mae_sarima:.4f}  RMSE: {rmse_sarima:.4f}')

# ETS
ets_model = ExponentialSmoothing(train, trend='add', seasonal=None)
ets_fit   = ets_model.fit()
ets_pred  = ets_fit.forecast(steps=n_forecast)
mae_ets   = mean_absolute_error(test, ets_pred)
rmse_ets  = np.sqrt(mean_squared_error(test, ets_pred))
print(f'ETS           — MAE: {mae_ets:.4f}  RMSE: {rmse_ets:.4f}')

# ML models (lag features, reduced n_forecast=1 due to data size)
def make_features(series, lags=1):
    df_feat = pd.DataFrame({'NDVI': series})
    for i in range(1, lags+1):
        df_feat[f'lag_{i}'] = df_feat['NDVI'].shift(i)
    df_feat['month'] = df_feat.index.month
    return df_feat.dropna()

full_features = make_features(monthly_ndvi, lags=1)
n_forecast_ml = 1
split_idx = len(full_features) - n_forecast_ml
X_train = full_features.iloc[:split_idx].drop(columns='NDVI')
y_train = full_features.iloc[:split_idx]['NDVI']
X_test  = full_features.iloc[split_idx:].drop(columns='NDVI')
y_test  = full_features.iloc[split_idx:]['NDVI']

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train); rf_pred = rf_model.predict(X_test)
mae_rf  = mean_absolute_error(y_test, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f'Random Forest — MAE: {mae_rf:.4f}  RMSE: {rmse_rf:.4f}')

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train); xgb_pred = xgb_model.predict(X_test)
mae_xgb  = mean_absolute_error(y_test, xgb_pred)
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f'XGBoost       — MAE: {mae_xgb:.4f}  RMSE: {rmse_xgb:.4f}')

# Prophet with sowing changepoints
prophet_df = pd.DataFrame({'ds': monthly_ndvi.index, 'y': monthly_ndvi.values})
sowing_changepoints = ts.dropna(subset=['sowing_date'])['sowing_date'].dt.to_period('M').dt.to_timestamp().unique().tolist()
train_cutoff = prophet_df['ds'].iloc[-(n_forecast+1)]
train_changepoints = [d for d in sowing_changepoints if d <= train_cutoff]
prophet_train = prophet_df[prophet_df['ds'] <= train_cutoff]
prophet_test  = prophet_df[prophet_df['ds'] >  train_cutoff]
prophet_model = Prophet(changepoints=train_changepoints, yearly_seasonality=False, weekly_seasonality=False,
                         daily_seasonality=False, seasonality_mode='additive', interval_width=0.80)
prophet_model.fit(prophet_train)
future = prophet_model.make_future_dataframe(periods=len(prophet_test), freq='MS')
forecast = prophet_model.predict(future)
prophet_pred = forecast.set_index('ds')['yhat'].reindex(prophet_test['ds'].values).values
mae_prophet  = mean_absolute_error(prophet_test['y'].values, prophet_pred)
rmse_prophet = np.sqrt(mean_squared_error(prophet_test['y'].values, prophet_pred))
print(f'Prophet       — MAE: {mae_prophet:.4f}  RMSE: {rmse_prophet:.4f}')

# Ridge with agronomic features
monthly_features = (
    ts[ts['sensor_status']=='OK']
    .assign(month=ts['date'].dt.to_period('M'))
    .groupby('month')
    .agg(ndvi=('ndvi_value','mean'), temp_mean=('temperature_c','mean'), temp_max=('temperature_c','max'),
         rainfall_total=('rainfall_mm','sum'), rainfall_days=('rainfall_mm', lambda x: (x>0).sum()))
    .reset_index().sort_values('month')
)
monthly_features['month_num'] = monthly_features['month'].dt.month
monthly_features['month_sin'] = np.sin(2*np.pi*monthly_features['month_num']/12)
monthly_features['month_cos'] = np.cos(2*np.pi*monthly_features['month_num']/12)
monthly_features['ndvi_lag1'] = monthly_features['ndvi'].shift(1)
monthly_features = monthly_features.dropna()
monthly_features.index = monthly_features['month'].dt.to_timestamp()
feature_cols = ['temp_mean','temp_max','rainfall_total','rainfall_days','month_sin','month_cos','ndvi_lag1']
split = len(monthly_features) - n_forecast
X_r_train = monthly_features.iloc[:split][feature_cols]; y_r_train = monthly_features.iloc[:split]['ndvi']
X_r_test  = monthly_features.iloc[split:][feature_cols];  y_r_test  = monthly_features.iloc[split:]['ndvi']
scaler = StandardScaler()
ridge_model = RidgeCV(alphas=[0.01,0.1,1.0,10.0,100.0], cv=3)
ridge_model.fit(scaler.fit_transform(X_r_train), y_r_train)
ridge_pred = ridge_model.predict(scaler.transform(X_r_test))
mae_ridge  = mean_absolute_error(y_r_test, ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_r_test, ridge_pred))
print(f'Ridge         — MAE: {mae_ridge:.4f}  RMSE: {rmse_ridge:.4f}  alpha: {ridge_model.alpha_:.3f}')

# Final comparison plot
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly_ndvi.index, monthly_ndvi.values, color='#1D9E75', linewidth=2, marker='o', markersize=5, label='Actual NDVI')
ax.axvline(test.index[0], color='gray', linestyle='--', linewidth=1, label='Train / Test split')
ax.plot(test.index, sarima_pred.values, color='#D85A30', linestyle='--', marker='s', markersize=7, label='SARIMA')
ax.plot(test.index, ets_pred.values,    color='#378ADD', linestyle='--', marker='^', markersize=7, label='ETS')
ax.plot(y_test.index, rf_pred,          color='#7F77DD', linestyle='--', marker='D', markersize=7, label='Random Forest')
ax.plot(y_test.index, xgb_pred,         color='#BA7517', linestyle='--', marker='P', markersize=7, label='XGBoost')
ax.plot(prophet_test['ds'], prophet_pred, color='#E91E8C', linestyle='--', marker='*', markersize=9, label='Prophet')
ax.plot(y_r_test.index, ridge_pred,     color='#00BCD4', linestyle='--', marker='h', markersize=7, label='Ridge')
ax.set_title("NDVI Forecasting — All 6 Models Compared", fontsize=13, fontweight='bold')
ax.set_ylabel("Mean Monthly NDVI"); ax.set_ylim(0,1); ax.set_xlabel("Month")
ax.legend(loc='lower left', fontsize=9, frameon=False, ncol=2); ax.grid(True, alpha=0.2)
plt.tight_layout(); plt.savefig('plot10_all_forecasts.png', bbox_inches='tight'); plt.show()

print('\nPipeline complete.')
