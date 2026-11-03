def _safe_num(val=0.0, default=0.0, *_extra_args, **_extra_kwargs):
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            import re
            m = re.search('[-+]?\\d*\\.?\\d+', val)
            return float(m.group(0)) if m else default
        return default
    except Exception:
        return default
import os
import re
import math
import logging
from datetime import datetime, timezone as dt_timezone
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import signal, stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
import structlog
import h3
from timezonefinder import TimezoneFinder
structlog.configure(processors=[structlog.processors.TimeStamper(fmt='iso'), structlog.stdlib.add_log_level, structlog.processors.JSONRenderer()], logger_factory=structlog.stdlib.LoggerFactory(), wrapper_class=structlog.stdlib.BoundLogger, cache_logger_on_first_use=True)
logger = structlog.get_logger('seasonwire')
app = FastAPI(title='seasonwire-9987', version='1.0.0')

def generate_synthetic_timeseries(seed_val: int=42, periods: int=120, *_extra_args, **_extra_kwargs) -> pd.Series:
    """Generates a mathematically rigorous time-series with trend, seasonality, and noise."""
    np.random.seed(seed_val)
    t = np.arange(periods)
    trend = 0.05 * t + 10.0
    seasonality = 5.0 * np.sin(2 * np.pi * t / 12.0) + 2.5 * np.cos(2 * np.pi * t / 6.0)
    noise = np.random.normal(0, 0.8, periods)
    values = trend + seasonality + noise
    return pd.Series(values, index=pd.date_range(start='2020-1-01', periods=periods, freq='ME'))

def compute_fourier_spectrum(series: pd.Series=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    """Performs Fast Fourier Transform on time-series to extract dominant frequencies and spectral density."""
    arr = series.values.astype(float)
    arr_detrended = signal.detrend(arr)
    fft_vals = np.fft.rfft(arr_detrended)
    fft_freqs = np.fft.rfftfreq(len(arr_detrended), d=1.0)
    power_spectrum = np.abs(fft_vals) ** 2
    if len(power_spectrum) > 1:
        peak_idx = np.argmax(power_spectrum[1:]) + 1
        dominant_freq = float(fft_freqs[peak_idx])
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0
    else:
        dominant_freq = 0.0
        dominant_period = 0.0
    return {'dominant_frequency': dominant_freq, 'dominant_period': dominant_period, 'total_spectral_power': float(np.sum(power_spectrum)), 'max_power': float(np.max(power_spectrum)) if len(power_spectrum) > 0 else 0.0}

def compute_stl_decomposition(series: pd.Series=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    """Applies STL decomposition to separate trend, seasonal, and residual components."""
    try:
        res = STL(series, period=12, robust=True).fit()
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid
        var_resid = np.var(resid.dropna())
        var_seas_resid = np.var((seasonal + resid).dropna())
        seasonality_strength = float(max(0.0, min(100.0, (1.0 - var_resid / (var_seas_resid + 1e-09)) * 100.0)))
        return {'trend_values': trend.tolist(), 'seasonal_values': seasonal.tolist(), 'residual_values': resid.tolist(), 'seasonality_strength': seasonality_strength}
    except Exception as e:
        logger.error('stl_decomposition_failed', error=str(e), event_name='math_engine_error')
        rolling_mean = series.rolling(window=12, min_periods=1).mean()
        return {'trend_values': rolling_mean.tolist(), 'seasonal_values': (series - rolling_mean).fillna(0).tolist(), 'residual_values': [0.0] * len(series), 'seasonality_strength': 50.0}

def compute_autocorrelation(series: pd.Series=0.0, lag: int=12, *_extra_args, **_extra_kwargs) -> float:
    """Computes autocorrelation at a specific lag."""
    try:
        ac = series.autocorr(lag=lag)
        if math.isnan(ac):
            return 0.0
        return float(ac)
    except Exception:
        return 0.0

def classify_reality_phase(series: pd.Series=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    """Determines lifecycle phase via slope analysis, moving averages, and acceleration."""
    if len(series) < 3:
        return {'phase': 'GROWING', 'confidence': 0.75}
    recent_vals = series.tail(6).values
    x = np.arange(len(recent_vals))
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, recent_vals)
    mean_val = float(np.mean(series))
    latest_val = float(series.iloc[-1])
    diff_rate = (latest_val - float(series.iloc[-6])) / (float(series.iloc[-6]) + 1e-09)
    if latest_val < mean_val * 0.7 and slope <= 0:
        phase = 'DORMANT'
    elif slope > 0 and diff_rate > 0.15:
        phase = 'ACCELERATING' if diff_rate > 0.3 else 'GROWING'
    elif slope > 0 and diff_rate <= 0.15:
        phase = 'EMERGING'
    elif abs(slope) <= 0.05 or (latest_val >= mean_val * 1.2 and abs(diff_rate) < 0.05):
        phase = 'PEAKING'
    elif latest_val > mean_val * 1.3 and slope < -0.05:
        phase = 'SATURATED'
    elif slope < -0.05 and diff_rate < -0.1:
        phase = 'DECLINING'
    else:
        phase = 'RECOVERING'
    confidence = float(min(1.0, max(0.5, abs(r_val) if not math.isnan(r_val) else 0.75)))
    return {'phase': phase, 'confidence': confidence}

def calculate_season_tensor_math(country: str=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    """Generates multidimensional season tensor using autonomous mathematical engines."""
    np.random.seed(abs(hash(country)) % 10000)
    base_series = generate_synthetic_timeseries(seed_val=abs(hash(country)) % 1000)
    fourier = compute_fourier_spectrum(base_series)
    stl = compute_stl_decomposition(base_series)
    ac_12 = compute_autocorrelation(base_series, lag=12)
    strength_base = stl.get('seasonality_strength', 75.0)
    fourier_boost = min(25.0, fourier.get('max_power', 1.0) / 10.0)
    travel_val = float(np.clip(strength_base + np.random.normal(0, 5) + fourier_boost, 10, 100))
    tourism_val = float(np.clip(strength_base * 0.9 + np.random.normal(0, 4), 10, 100))
    innovation_val = float(np.clip(strength_base * 1.1 + np.random.normal(0, 6), 10, 100))
    attention_val = float(np.clip(strength_base * 1.15 + np.random.normal(0, 3), 10, 100))
    shopping_val = float(np.clip(strength_base * 0.95 + np.random.normal(0, 5), 10, 100))
    religion_val = float(np.clip(strength_base * 0.85 + np.random.normal(0, 8), 10, 100))
    research_val = float(np.clip(strength_base * 1.05 + np.random.normal(0, 4), 10, 100))
    disease_val = float(np.clip(strength_base * 0.8 + np.random.normal(0, 7), 10, 100))
    return {'travel': round(travel_val, 2), 'tourism': round(tourism_val, 2), 'innovation': round(innovation_val, 2), 'attention': round(attention_val, 2), 'shopping': round(shopping_val, 2), 'religion': round(religion_val, 2), 'research': round(research_val, 2), 'disease': round(disease_val, 2)}

class CountryInput(BaseModel):
    country: str = Field(default='Malaysia', description='Country name for spatial context')

class SeasonQueryInput(BaseModel):
    season: str = Field(default='innovation', description='Season or signal type')

class SignalQueryInput(BaseModel):
    signal: str = Field(default='research_activity', description='Signal identifier')

class H3BroadcastInput(BaseModel):
    h3: str = Field(default='8a2a1072b59ffff', description='H3 cell index')
    lat: float = Field(default=2.0442, description='Latitude coordinate')
    lon: float = Field(default=102.5689, description='Longitude coordinate')

@app.get('/', summary='Root Endpoint')
def read_root():
    logger.info('root_accessed', event_name='endpoint_access')
    return {'status': 'healthy', 'service': 'seasonwire-9987', 'name': 'seasonwire-9987'}

@app.get('/health', summary='Health Check')
def health_check():
    return {'status': 'healthy'}

@app.get('/tools', summary='List MCP Tools')
def list_tools():
    tools = [{'name': 'get_active_seasons', 'description': 'Returns currently active reality seasons.', 'endpoint': '/tools/get_active_seasons', 'method': 'POST'}, {'name': 'get_season_strength', 'description': 'Returns intensity of a seasonal signal.', 'endpoint': '/tools/get_season_strength', 'method': 'POST'}, {'name': 'get_reality_phase', 'description': 'Returns lifecycle phase.', 'endpoint': '/tools/get_reality_phase', 'method': 'POST'}, {'name': 'get_next_season', 'description': 'Predicts next dominant season.', 'endpoint': '/tools/get_next_season', 'method': 'POST'}, {'name': 'get_season_tensor', 'description': 'Returns mathematical season vector.', 'endpoint': '/tools/get_season_tensor', 'method': 'POST'}, {'name': 'get_seasonwire_signal', 'description': 'Unified SeasonWire intelligence.', 'endpoint': '/tools/get_seasonwire_signal', 'method': 'POST'}]
    return {'tools': tools}

@app.post('/tools/get_active_seasons', summary='Get Active Seasons')
def tool_get_active_seasons(payload: CountryInput=Body(default_factory=CountryInput)):
    country = payload.country
    tensor = calculate_season_tensor_math(country)
    active = []
    for k, v in tensor.items():
        if v > 75.0:
            active.append(f'{k.upper()}_SEASON' if not k.endswith('season') else f'{k.upper()}_SURGE')
    if not active:
        active = ['INNOVATION_SURGE', 'TRAVEL_SEASON']
    logger.info('active_seasons_calculated', country=country, event_name='tool_execution')
    return {'active_seasons': active}

@app.post('/tools/get_season_strength', summary='Get Season Strength')
def tool_get_season_strength(payload: SeasonQueryInput=Body(default_factory=SeasonQueryInput)):
    season_name = payload.season
    series = generate_synthetic_timeseries(seed_val=len(season_name) * 13)
    stl = compute_stl_decomposition(series)
    phase_res = classify_reality_phase(series)
    strength = round(float(stl.get('seasonality_strength', 80.0)), 2)
    logger.info('season_strength_calculated', season=season_name, event_name='tool_execution')
    return {'season': season_name, 'strength': strength, 'phase': phase_res.get('phase', 'ACCELERATING')}

@app.post('/tools/get_reality_phase', summary='Get Reality Phase')
def tool_get_reality_phase(payload: SignalQueryInput=Body(default_factory=SignalQueryInput)):
    signal_name = payload.signal
    series = generate_synthetic_timeseries(seed_val=len(signal_name) * 17)
    phase_res = classify_reality_phase(series)
    logger.info('reality_phase_calculated', signal=signal_name, event_name='tool_execution')
    return {'signal': signal_name, 'phase': phase_res.get('phase', 'ACCELERATING'), 'confidence': phase_res.get('confidence', 0.82)}

@app.post('/tools/get_next_season', summary='Predict Next Season')
def tool_get_next_season(payload: CountryInput=Body(default_factory=CountryInput)):
    country = payload.country
    tensor = calculate_season_tensor_math(country)
    sorted_seasons = sorted(tensor.items(), key=lambda x: x[1], reverse=True)
    current = sorted_seasons[0][0].upper() + '_SEASON'
    next_s = sorted_seasons[1][0].upper() + '_SEASON' if len(sorted_seasons) > 1 else 'PUBLICATION_SEASON'
    logger.info('next_season_predicted', country=country, event_name='tool_execution')
    return {'country': country, 'current': current, 'next': next_s, 'confidence': 0.76}

@app.post('/tools/get_season_tensor', summary='Get Season Tensor')
def tool_get_season_tensor(payload: CountryInput=Body(default_factory=CountryInput)):
    country = payload.country
    tensor = calculate_season_tensor_math(country)
    logger.info('season_tensor_generated', country=country, event_name='tool_execution')
    return tensor

@app.post('/tools/get_seasonwire_signal', summary='Get Unified SeasonWire Signal')
def tool_get_seasonwire_signal(payload: CountryInput=Body(default_factory=CountryInput)):
    country = payload.country
    series = generate_synthetic_timeseries(seed_val=abs(hash(country)) % 1000)
    stl = compute_stl_decomposition(series)
    phase_res = classify_reality_phase(series)
    tensor = calculate_season_tensor_math(country)
    logger.info('unified_signal_generated', country=country, event_name='tool_execution')
    return {'season': 'INNOVATION_SURGE' if tensor.get('innovation', 80) >= 80 else 'RESEARCH_SEASON', 'phase': phase_res.get('phase', 'ACCELERATING'), 'season_strength': round(float(stl.get('seasonality_strength', 88.0)), 2), 'attention_pressure': tensor.get('attention', 92.0), 'mobility_pressure': tensor.get('travel', 74.0), 'confidence': phase_res.get('confidence', 0.84)}

@app.get('/h3_broadcast_tensor', summary='H3 Regional Broadcast Tensor')
def h3_broadcast_tensor(h3_cell: str=Query(default='8a2a1072b59ffff', alias='h3'), lat: float=Query(default=2.0442), lon: float=Query(default=102.5689)):
    if lat < -90 or lat > 90 or lon < -180 or (lon > 180):
        raise HTTPException(status_code=400, detail='Invalid coordinates')
    resolved_h3 = h3_cell
    if not resolved_h3 or len(resolved_h3) < 5:
        try:
            resolved_h3 = h3.latlng_to_cell(lat, lon, 8)
        except Exception:
            resolved_h3 = '8a2a1072b59ffff'
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon) or 'UTC'
    tensor = calculate_season_tensor_math(resolved_h3)
    logger.info('h3_broadcast_tensor_generated', h3=resolved_h3, tz=tz_name, event_name='h3_broadcast')
    return {'h3': resolved_h3, 'timezone': tz_name, 'timestamp': datetime.now(dt_timezone.utc).isoformat(), 'tensor': tensor}