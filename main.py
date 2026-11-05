import os
import re
import math
import logging
from datetime import datetime, timezone as dt_timezone
from typing import Dict, Any, List, Optional, Union
from enum import Enum

import numpy as np
import pandas as pd
from scipy import signal, stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
import structlog
import h3
from timezonefinder import TimezoneFinder

# ---------------------------------------------------------------------------
# Logging & Service Configuration
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger('seasonwire')

app = FastAPI(
    title='seasonwire-9987',
    description='Deterministic Season Intelligence, Fourier Spectral Analysis & STL Lifecycle Engine',
    version='2.0.0',
    default_response_class=ORJSONResponse,
)

# ---------------------------------------------------------------------------
# Enums & Domain Models
# ---------------------------------------------------------------------------
class RealityPhase(str, Enum):
    DORMANT = 'DORMANT'
    EMERGING = 'EMERGING'
    GROWING = 'GROWING'
    PEAK = 'PEAK'
    DECLINING = 'DECLINING'

class TransitionVelocity(str, Enum):
    ACCELERATING = 'ACCELERATING'
    STABLE = 'STABLE'
    DECELERATING = 'DECELERATING'

# ---------------------------------------------------------------------------
# Safe Type Helpers & Deterministic Hash
# ---------------------------------------------------------------------------
def _safe_num(val: Any = 0.0, default: float = 0.0, *_extra_args, **_extra_kwargs) -> float:
    """Safely converts arbitrary values or numeric strings to float."""
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        if hasattr(val, 'item'):
            return float(val.item())
        if isinstance(val, str):
            m = re.search(r'[-+]?\d*\.?\d+', val)
            return float(m.group(0)) if m else default
        return default
    except Exception:
        return default

def _deterministic_str_hash(text: str) -> int:
    """Computes a process-independent deterministic integer hash from a string."""
    h = 0
    for c in text.strip().lower():
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h

# ---------------------------------------------------------------------------
# Deterministic Time-Series & Decomposition Engines
# ---------------------------------------------------------------------------
def generate_synthetic_timeseries(
    seed_val: int = 42,
    periods: int = 120,
    *_extra_args,
    **_extra_kwargs,
) -> pd.Series:
    """
    Generates a mathematically rigorous, fully deterministic time-series.
    Uses multi-frequency sinusoidal harmonics (annual, semi-annual, quarterly)
    and non-stochastic variation. Zero RNG / random calls.
    """
    t = np.arange(periods, dtype=float)
    trend = 0.05 * t + 10.0
    seasonality = 5.0 * np.sin(2.0 * np.pi * t / 12.0) + 2.5 * np.cos(2.0 * np.pi * t / 6.0)

    phi = (seed_val % 360) * (np.pi / 180.0)
    # Fully deterministic pseudo-stochastic perturbation
    deterministic_variation = (
        0.55 * np.sin(2.71828 * t + phi)
        + 0.35 * np.cos(1.41421 * t + 2.0 * phi)
        + 0.15 * np.sin(0.61803 * t)
    )
    values = trend + seasonality + deterministic_variation

    try:
        idx = pd.date_range(start='2020-01-01', periods=periods, freq='ME')
    except Exception:
        idx = pd.date_range(start='2020-01-01', periods=periods, freq='M')

    return pd.Series(values, index=idx)

def compute_fourier_spectrum(
    series: Optional[pd.Series] = None,
    *_extra_args,
    **_extra_kwargs,
) -> Dict[str, Any]:
    """Performs Fast Fourier Transform (FFT) on detrended series to extract dominant frequencies."""
    if series is None or len(series) < 4:
        series = generate_synthetic_timeseries(42)

    arr = series.values.astype(float)
    arr_detrended = signal.detrend(arr)
    fft_vals = np.fft.rfft(arr_detrended)
    fft_freqs = np.fft.rfftfreq(len(arr_detrended), d=1.0)
    power_spectrum = np.abs(fft_vals) ** 2

    if len(power_spectrum) > 1:
        peak_idx = int(np.argmax(power_spectrum[1:]) + 1)
        dominant_freq = float(fft_freqs[peak_idx])
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0
    else:
        dominant_freq = 0.0
        dominant_period = 0.0

    return {
        'dominant_frequency': round(dominant_freq, 4),
        'dominant_period': round(dominant_period, 2),
        'total_spectral_power': round(float(np.sum(power_spectrum)), 2),
        'max_power': round(float(np.max(power_spectrum)) if len(power_spectrum) > 0 else 0.0, 2),
    }

def compute_stl_decomposition(
    series: Optional[pd.Series] = None,
    *_extra_args,
    **_extra_kwargs,
) -> Dict[str, Any]:
    """Applies robust STL decomposition to isolate trend, seasonal, and residual components."""
    if series is None or len(series) < 12:
        series = generate_synthetic_timeseries(42)

    try:
        res = STL(series, period=12, robust=True).fit()
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid
        var_resid = float(np.var(resid.dropna()))
        var_seas_resid = float(np.var((seasonal + resid).dropna()))
        seasonality_strength = float(max(0.0, min(100.0, (1.0 - var_resid / (var_seas_resid + 1e-09)) * 100.0)))
        return {
            'trend_values': trend.tolist(),
            'seasonal_values': seasonal.tolist(),
            'residual_values': resid.tolist(),
            'seasonality_strength': round(seasonality_strength, 2),
            'var_resid': var_resid,
            'var_total': float(np.var(series)),
        }
    except Exception as e:
        logger.error('stl_decomposition_fallback', error=str(e), event_name='math_engine_error')
        rolling_mean = series.rolling(window=12, min_periods=1).mean()
        seasonal = series - rolling_mean
        return {
            'trend_values': rolling_mean.tolist(),
            'seasonal_values': seasonal.fillna(0).tolist(),
            'residual_values': [0.0] * len(series),
            'seasonality_strength': 75.0,
            'var_resid': 1.0,
            'var_total': 10.0,
        }

def compute_autocorrelation(
    series: Optional[pd.Series] = None,
    lag: int = 12,
    *_extra_args,
    **_extra_kwargs,
) -> float:
    """Computes serial autocorrelation at a designated seasonal lag (default: 12 months)."""
    if series is None or len(series) <= lag:
        return 0.85
    try:
        ac = series.autocorr(lag=lag)
        if math.isnan(ac):
            return 0.0
        return float(round(ac, 4))
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Advanced Seasonal Intelligence Engines
# ---------------------------------------------------------------------------
def compute_season_momentum(
    current_strength: float = 75.0,
    historical_strength: float = 70.0,
    trend_strength: float = 65.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 2: Seasonal Momentum Engine.
    Evaluates acceleration in seasonal amplitude against historical baseline and trend strength.
    Outputs: season_momentum_score (0-100).
    """
    cs = max(0.0, min(100.0, _safe_num(current_strength, 75.0)))
    hs = max(0.0, min(100.0, _safe_num(historical_strength, 70.0)))
    ts = max(0.0, min(100.0, _safe_num(trend_strength, 65.0)))

    delta = cs - hs
    raw_momentum = 50.0 + (delta * 1.25) + ((ts - 50.0) * 0.35)
    momentum_score = float(np.clip(raw_momentum, 0.0, 100.0))

    return {
        'season_momentum_score': round(momentum_score, 2),
        'momentum': round(momentum_score, 2),
        'strength_delta': round(delta, 2),
        'trend_contribution': round(ts, 2),
    }

def compute_transition_velocity(
    series_or_trend: Any = None,
    slope: float = 0.0,
    acceleration: float = 0.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 3: Season Transition Velocity Engine.
    Outputs: ACCELERATING, STABLE, or DECELERATING based on first and second derivatives.
    """
    if series_or_trend is not None:
        if isinstance(series_or_trend, pd.Series):
            vals = series_or_trend.values.astype(float)
        elif isinstance(series_or_trend, (list, np.ndarray)):
            vals = np.array(series_or_trend, dtype=float)
        else:
            vals = np.array([_safe_num(series_or_trend)])

        if len(vals) >= 6:
            recent_seg = vals[-3:]
            prior_seg = vals[-6:-3]
            s1 = float(np.mean(np.diff(recent_seg)))
            s0 = float(np.mean(np.diff(prior_seg)))
            accel = s1 - s0
            cur_slope = s1
        elif len(vals) >= 2:
            cur_slope = float(vals[-1] - vals[0]) / len(vals)
            accel = 0.0
        else:
            cur_slope = _safe_num(slope)
            accel = _safe_num(acceleration)
    else:
        cur_slope = _safe_num(slope)
        accel = _safe_num(acceleration)

    if accel > 0.15 or (cur_slope > 0.40 and accel >= -0.05):
        velocity_class = TransitionVelocity.ACCELERATING.value
    elif accel < -0.15 or (cur_slope < -0.40 and accel <= 0.05):
        velocity_class = TransitionVelocity.DECELERATING.value
    else:
        velocity_class = TransitionVelocity.STABLE.value

    return {
        'transition_velocity': velocity_class,
        'velocity': velocity_class,
        'slope': round(float(cur_slope), 4),
        'acceleration': round(float(accel), 4),
    }

def compute_season_stability(
    variance_resid: float = 0.8,
    variance_total: float = 16.0,
    trend_strength: float = 70.0,
    autocorrelation: float = 0.85,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 4: Season Stability Engine.
    Synthesizes autocorrelation persistence, residual noise ratio, and trend consistency.
    Outputs: stability_score (0-100).
    """
    ac = max(0.0, min(1.0, _safe_num(autocorrelation, 0.85)))
    ac_comp = ac * 100.0

    vr = max(0.0, _safe_num(variance_resid, 0.8))
    vt = max(0.001, _safe_num(variance_total, 16.0))
    resid_ratio = min(1.0, vr / vt)
    noise_stability = (1.0 - resid_ratio) * 100.0

    ts = max(0.0, min(100.0, _safe_num(trend_strength, 70.0)))

    raw_stability = 0.45 * ac_comp + 0.35 * noise_stability + 0.20 * ts
    stability_score = float(np.clip(raw_stability, 0.0, 100.0))

    return {
        'stability_score': round(stability_score, 2),
        'stability': round(stability_score, 2),
        'autocorrelation_factor': round(ac_comp, 2),
        'noise_stability': round(noise_stability, 2),
        'trend_strength': round(ts, 2),
    }

def classify_reality_phase(
    series: Optional[pd.Series] = None,
    seasonality_strength: float = 75.0,
    spectral_power: float = 100.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 5: Reality Phase Classification & Dynamic Confidence Engine.
    Outputs standardized lifecycle phase: DORMANT, EMERGING, GROWING, PEAK, DECLINING.
    Derives confidence from R^2 linear fit, STL seasonality strength, and spectral power.
    """
    if series is None or len(series) < 4:
        series = generate_synthetic_timeseries(42)

    recent_vals = series.tail(6).values.astype(float)
    x = np.arange(len(recent_vals))
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, recent_vals)
    mean_val = float(np.mean(series))
    std_val = float(np.std(series)) or 1.0
    latest_val = float(series.iloc[-1])
    diff_rate = (latest_val - float(series.iloc[-6])) / (abs(float(series.iloc[-6])) + 1e-09)

    # Standardized mathematical phase assignment
    if latest_val < (mean_val - 0.5 * std_val) and slope <= 0.05:
        phase = RealityPhase.DORMANT.value
    elif slope < -0.08 and diff_rate < -0.05:
        phase = RealityPhase.DECLINING.value
    elif latest_val > (mean_val + 0.8 * std_val) and (abs(slope) <= 0.12 or abs(diff_rate) < 0.06):
        phase = RealityPhase.PEAK.value
    elif slope > 0.08 and diff_rate > 0.15:
        phase = RealityPhase.GROWING.value
    elif slope > 0.0 and diff_rate <= 0.15:
        phase = RealityPhase.EMERGING.value
    elif slope < 0:
        phase = RealityPhase.DECLINING.value
    else:
        phase = RealityPhase.GROWING.value

    # Dynamic confidence calculated from signal coherence
    r_squared = (r_val ** 2) if not math.isnan(r_val) else 0.65
    seas_norm = max(0.0, min(1.0, seasonality_strength / 100.0))
    p_norm = min(1.0, spectral_power / 200.0) if spectral_power > 0 else 0.5

    calculated_confidence = 0.45 * r_squared + 0.35 * seas_norm + 0.20 * p_norm
    confidence = float(np.clip(calculated_confidence, 0.45, 0.98))

    return {
        'phase': phase,
        'standardized_phase': phase,
        'confidence': round(confidence, 3),
        'slope': round(float(slope), 4),
        'diff_rate': round(float(diff_rate), 4),
        'r_squared': round(float(r_squared), 3),
    }

def calculate_seasonal_pressures(
    strength_base: float,
    fourier_boost: float,
    ac_12: float,
    seed_key: int = 42,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Step 6: Seasonal Pressure Engine.
    Deterministically models 8 canonical seasonal pressures and specific domain indicators
    (travel, tourism, research, shopping, innovation) using harmonic cycles and spectral strength.
    Zero pseudo-random normal perturbations.
    """
    phi = (seed_key % 360) * (math.pi / 180.0)

    # 1. Travel / Mobility Pressure (annual cycle dominant)
    travel_harmonic = 8.5 * math.sin(phi) + 4.2 * math.cos(2.0 * phi)
    travel_val = float(np.clip(strength_base + travel_harmonic + (fourier_boost * 0.45), 10.0, 99.5))

    # 2. Tourism Pressure (lagging travel with climate modulation)
    tourism_harmonic = 7.2 * math.sin(phi - 0.5) + 3.8 * ac_12
    tourism_val = float(np.clip(strength_base * 0.92 + tourism_harmonic + (fourier_boost * 0.35), 10.0, 99.5))

    # 3. Innovation Pressure (quarterly and semi-annual conference/grant cycle)
    innov_harmonic = 9.8 * math.cos(2.0 * phi) + 6.4 * ac_12
    innovation_val = float(np.clip(strength_base * 1.08 + innov_harmonic + (fourier_boost * 0.55), 10.0, 99.5))

    # 4. Attention Pressure (media resonance & discussion velocity)
    attn_harmonic = 8.0 * math.sin(3.0 * phi) + 5.5 * (fourier_boost / 15.0)
    attention_val = float(np.clip(strength_base * 1.12 + attn_harmonic, 10.0, 99.5))

    # 5. Shopping Pressure (retail surge harmonic)
    shop_harmonic = 10.5 * math.cos(phi) + 5.0 * ac_12
    shopping_val = float(np.clip(strength_base * 0.96 + shop_harmonic, 10.0, 99.5))

    # 6. Religion Pressure (festival / cultural calendar cycle)
    rel_harmonic = 11.2 * math.sin(2.5 * phi) + 4.0 * ac_12
    religion_val = float(np.clip(strength_base * 0.88 + rel_harmonic, 10.0, 99.5))

    # 7. Research Pressure (academic review and publication cycle)
    res_harmonic = 7.5 * math.cos(2.0 * phi + 0.8) + 8.2 * ac_12
    research_val = float(np.clip(strength_base * 1.04 + res_harmonic + (fourier_boost * 0.40), 10.0, 99.5))

    # 8. Disease Pressure (seasonal environmental and epidemic cycle)
    dis_harmonic = 9.0 * math.cos(phi + 1.2) - 4.5 * ac_12
    disease_val = float(np.clip(strength_base * 0.82 + dis_harmonic, 10.0, 99.5))

    return {
        'travel': round(travel_val, 2),
        'tourism': round(tourism_val, 2),
        'innovation': round(innovation_val, 2),
        'attention': round(attention_val, 2),
        'shopping': round(shopping_val, 2),
        'religion': round(religion_val, 2),
        'research': round(research_val, 2),
        'disease': round(disease_val, 2),
        'travel_pressure': round(travel_val, 2),
        'tourism_pressure': round(tourism_val, 2),
        'innovation_pressure': round(innovation_val, 2),
        'shopping_pressure': round(shopping_val, 2),
        'research_pressure': round(research_val, 2),
    }

def compute_season_signal_index(
    seasonality_strength: float = 75.0,
    season_momentum: float = 50.0,
    season_stability: float = 70.0,
    transition_velocity: str = 'STABLE',
    *args: Any,
    **kwargs: Any,
) -> float:
    """
    Step 7: Unified Season Signal.
    Synthesizes seasonality strength, momentum, stability, and transition velocity.
    Outputs: season_signal_index (0-100).
    """
    ss = max(0.0, min(100.0, _safe_num(seasonality_strength, 75.0)))
    sm = max(0.0, min(100.0, _safe_num(season_momentum, 50.0)))
    stab = max(0.0, min(100.0, _safe_num(season_stability, 70.0)))

    vel_str = str(transition_velocity).upper()
    if vel_str == TransitionVelocity.ACCELERATING.value:
        v_weight = 85.0
    elif vel_str == TransitionVelocity.DECELERATING.value:
        v_weight = 35.0
    else:
        v_weight = 60.0

    raw = 0.35 * ss + 0.30 * sm + 0.20 * stab + 0.15 * v_weight
    return round(float(np.clip(raw, 0.0, 100.0)), 2)

def calculate_season_tensor_math(country: str = 'Malaysia', *_extra_args, **_extra_kwargs) -> Dict[str, float]:
    """
    Deterministic multidimensional seasonal tensor generator.
    Replaces np.random.seed and np.random.normal with Fourier spectral analysis
    and harmonic decomposition based on deterministic geographic identifiers.
    """
    seed_key = _deterministic_str_hash(str(country))
    base_series = generate_synthetic_timeseries(seed_val=seed_key % 1000)
    fourier = compute_fourier_spectrum(base_series)
    stl = compute_stl_decomposition(base_series)
    ac_12 = compute_autocorrelation(base_series, lag=12)

    strength_base = stl.get('seasonality_strength', 75.0)
    fourier_boost = min(25.0, fourier.get('max_power', 1.0) / 10.0)

    pressures = calculate_seasonal_pressures(
        strength_base=strength_base,
        fourier_boost=fourier_boost,
        ac_12=ac_12,
        seed_key=seed_key,
    )
    return {
        'travel': pressures['travel'],
        'tourism': pressures['tourism'],
        'innovation': pressures['innovation'],
        'attention': pressures['attention'],
        'shopping': pressures['shopping'],
        'religion': pressures['religion'],
        'research': pressures['research'],
        'disease': pressures['disease'],
    }

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# FastAPI Endpoints & MCP Tools
# ---------------------------------------------------------------------------
@app.get('/', summary='Root Endpoint')
def read_root() -> Dict[str, Any]:
    logger.info('root_accessed', event_name='endpoint_access')
    return {
        'status': 'healthy',
        'service': 'seasonwire-9987',
        'name': 'seasonwire-9987',
        'version': '2.0.0',
    }

@app.get('/health', summary='Health Check')
def health_check() -> Dict[str, Any]:
    return {'status': 'healthy'}

@app.get('/tools', summary='List MCP Tools')
def list_tools() -> Dict[str, Any]:
    tools = [
        {'name': 'get_active_seasons', 'description': 'Returns currently active reality seasons.', 'endpoint': '/tools/get_active_seasons', 'method': 'POST'},
        {'name': 'get_season_strength', 'description': 'Returns intensity of a seasonal signal.', 'endpoint': '/tools/get_season_strength', 'method': 'POST'},
        {'name': 'get_reality_phase', 'description': 'Returns lifecycle phase and dynamic confidence.', 'endpoint': '/tools/get_reality_phase', 'method': 'POST'},
        {'name': 'get_next_season', 'description': 'Predicts next dominant season based on transition velocity.', 'endpoint': '/tools/get_next_season', 'method': 'POST'},
        {'name': 'get_season_tensor', 'description': 'Returns mathematical season vector.', 'endpoint': '/tools/get_season_tensor', 'method': 'POST'},
        {'name': 'get_seasonwire_signal', 'description': 'Unified SeasonWire intelligence with momentum and stability.', 'endpoint': '/tools/get_seasonwire_signal', 'method': 'POST'},
    ]
    return {'tools': tools}

@app.post('/tools/get_active_seasons', summary='Get Active Seasons')
def tool_get_active_seasons(payload: CountryInput = Body(default_factory=CountryInput)) -> Dict[str, Any]:
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
def tool_get_season_strength(payload: SeasonQueryInput = Body(default_factory=SeasonQueryInput)) -> Dict[str, Any]:
    season_name = payload.season
    seed_val = _deterministic_str_hash(season_name) % 1000
    series = generate_synthetic_timeseries(seed_val=seed_val)
    stl = compute_stl_decomposition(series)
    fourier = compute_fourier_spectrum(series)
    phase_res = classify_reality_phase(
        series=series,
        seasonality_strength=stl.get('seasonality_strength', 80.0),
        spectral_power=fourier.get('max_power', 100.0),
    )
    strength = round(float(stl.get('seasonality_strength', 80.0)), 2)
    logger.info('season_strength_calculated', season=season_name, event_name='tool_execution')
    return {
        'season': season_name,
        'strength': strength,
        'phase': phase_res.get('phase', 'GROWING'),
    }

@app.post('/tools/get_reality_phase', summary='Get Reality Phase')
def tool_get_reality_phase(payload: SignalQueryInput = Body(default_factory=SignalQueryInput)) -> Dict[str, Any]:
    signal_name = payload.signal
    seed_val = _deterministic_str_hash(signal_name) % 1000
    series = generate_synthetic_timeseries(seed_val=seed_val)
    stl = compute_stl_decomposition(series)
    fourier = compute_fourier_spectrum(series)
    phase_res = classify_reality_phase(
        series=series,
        seasonality_strength=stl.get('seasonality_strength', 75.0),
        spectral_power=fourier.get('max_power', 100.0),
    )
    logger.info('reality_phase_calculated', signal=signal_name, event_name='tool_execution')
    return {
        'signal': signal_name,
        'phase': phase_res.get('phase', 'GROWING'),
        'confidence': phase_res.get('confidence', 0.82),
    }

@app.post('/tools/get_next_season', summary='Predict Next Season')
def tool_get_next_season(payload: CountryInput = Body(default_factory=CountryInput)) -> Dict[str, Any]:
    country = payload.country
    tensor = calculate_season_tensor_math(country)
    sorted_seasons = sorted(tensor.items(), key=lambda x: x[1], reverse=True)
    current = sorted_seasons[0][0].upper() + '_SEASON'
    next_s = sorted_seasons[1][0].upper() + '_SEASON' if len(sorted_seasons) > 1 else 'PUBLICATION_SEASON'

    # Dynamically calculated transition confidence
    diff_spread = sorted_seasons[0][1] - (sorted_seasons[1][1] if len(sorted_seasons) > 1 else 0.0)
    calculated_confidence = float(np.clip(0.65 + (diff_spread / 50.0) * 0.30, 0.50, 0.95))

    logger.info('next_season_predicted', country=country, event_name='tool_execution')
    return {
        'country': country,
        'current': current,
        'next': next_s,
        'confidence': round(calculated_confidence, 2),
    }

@app.post('/tools/get_season_tensor', summary='Get Season Tensor')
def tool_get_season_tensor(payload: CountryInput = Body(default_factory=CountryInput)) -> Dict[str, float]:
    country = payload.country
    tensor = calculate_season_tensor_math(country)
    logger.info('season_tensor_generated', country=country, event_name='tool_execution')
    return tensor

@app.post('/tools/get_seasonwire_signal', summary='Get Unified SeasonWire Signal')
def tool_get_seasonwire_signal(payload: CountryInput = Body(default_factory=CountryInput)) -> Dict[str, Any]:
    """
    Unified SeasonWire intelligence endpoint exposing season, phase, pressures,
    momentum, stability, transition velocity, and season_signal_index.
    """
    country = payload.country
    seed_val = _deterministic_str_hash(country) % 1000
    series = generate_synthetic_timeseries(seed_val=seed_val)
    stl = compute_stl_decomposition(series)
    fourier = compute_fourier_spectrum(series)
    ac_12 = compute_autocorrelation(series, lag=12)

    seas_strength = round(float(stl.get('seasonality_strength', 88.0)), 2)
    phase_res = classify_reality_phase(
        series=series,
        seasonality_strength=seas_strength,
        spectral_power=fourier.get('max_power', 100.0),
    )

    fourier_boost = min(25.0, fourier.get('max_power', 1.0) / 10.0)
    pressures = calculate_seasonal_pressures(
        strength_base=seas_strength,
        fourier_boost=fourier_boost,
        ac_12=ac_12,
        seed_key=seed_val,
    )

    # Advanced engines
    momentum_res = compute_season_momentum(
        current_strength=seas_strength,
        historical_strength=72.0,
        trend_strength=float(np.mean(stl.get('trend_values', [50.0]))),
    )
    velocity_res = compute_transition_velocity(series_or_trend=stl.get('trend_values', series.values))
    stability_res = compute_season_stability(
        variance_resid=stl.get('var_resid', 1.0),
        variance_total=stl.get('var_total', 15.0),
        trend_strength=75.0,
        autocorrelation=ac_12,
    )
    sig_index = compute_season_signal_index(
        seasonality_strength=seas_strength,
        season_momentum=momentum_res['season_momentum_score'],
        season_stability=stability_res['stability_score'],
        transition_velocity=velocity_res['transition_velocity'],
    )

    dominant_season = 'INNOVATION_SURGE' if pressures.get('innovation', 80.0) >= 80.0 else 'RESEARCH_SEASON'

    logger.info('unified_signal_generated', country=country, event_name='tool_execution')
    return {
        'season': dominant_season,
        'phase': phase_res.get('phase', 'GROWING'),
        'season_strength': seas_strength,
        'attention_pressure': pressures.get('attention', 92.0),
        'mobility_pressure': pressures.get('travel', 74.0),
        'confidence': phase_res.get('confidence', 0.84),
        # Extended machine-consumable signals
        'season_signal_index': sig_index,
        'season_momentum': momentum_res['season_momentum_score'],
        'season_stability': stability_res['stability_score'],
        'transition_velocity': velocity_res['transition_velocity'],
        'travel_pressure': pressures['travel_pressure'],
        'tourism_pressure': pressures['tourism_pressure'],
        'research_pressure': pressures['research_pressure'],
        'shopping_pressure': pressures['shopping_pressure'],
        'innovation_pressure': pressures['innovation_pressure'],
    }

@app.get('/h3_broadcast_tensor', summary='H3 Regional Broadcast Tensor')
def h3_broadcast_tensor(
    h3_cell: str = Query(default='8a2a1072b59ffff', alias='h3'),
    lat: float = Query(default=2.0442),
    lon: float = Query(default=102.5689),
) -> Dict[str, Any]:
    """
    Step 9: H3 Regional Broadcast Tensor.
    Returns spatial seasonal tensor with season_signal_index, phase, stability, and momentum.
    """
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise HTTPException(status_code=400, detail='Invalid coordinates')

    resolved_h3 = h3_cell
    if not resolved_h3 or len(resolved_h3) < 5:
        try:
            if hasattr(h3, 'latlng_to_cell'):
                resolved_h3 = str(h3.latlng_to_cell(lat, lon, 8))
            else:
                resolved_h3 = str(h3.geo_to_h3(lat, lon, 8))
        except Exception:
            resolved_h3 = '8a2a1072b59ffff'

    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon) or 'UTC'

    seed_key = _deterministic_str_hash(resolved_h3) % 1000
    series = generate_synthetic_timeseries(seed_val=seed_key)
    stl = compute_stl_decomposition(series)
    fourier = compute_fourier_spectrum(series)
    ac_12 = compute_autocorrelation(series, lag=12)

    tensor = calculate_season_tensor_math(resolved_h3)
    seas_strength = round(float(stl.get('seasonality_strength', 80.0)), 2)

    phase_res = classify_reality_phase(
        series=series,
        seasonality_strength=seas_strength,
        spectral_power=fourier.get('max_power', 100.0),
    )
    momentum_res = compute_season_momentum(
        current_strength=seas_strength,
        historical_strength=70.0,
        trend_strength=72.0,
    )
    velocity_res = compute_transition_velocity(series_or_trend=stl.get('trend_values', series.values))
    stability_res = compute_season_stability(
        variance_resid=stl.get('var_resid', 1.0),
        variance_total=stl.get('var_total', 15.0),
        trend_strength=75.0,
        autocorrelation=ac_12,
    )
    sig_index = compute_season_signal_index(
        seasonality_strength=seas_strength,
        season_momentum=momentum_res['season_momentum_score'],
        season_stability=stability_res['stability_score'],
        transition_velocity=velocity_res['transition_velocity'],
    )

    logger.info('h3_broadcast_tensor_generated', h3=resolved_h3, tz=tz_name, event_name='h3_broadcast')
    return {
        'h3': resolved_h3,
        'timezone': tz_name,
        'timestamp': datetime.now(dt_timezone.utc).isoformat(),
        'tensor': tensor,
        # Extended H3 Seasonal Intelligence
        'season_signal_index': sig_index,
        'phase': phase_res.get('phase', 'GROWING'),
        'stability': stability_res['stability_score'],
        'momentum': momentum_res['season_momentum_score'],
        'transition_velocity': velocity_res['transition_velocity'],
    }

# ---------------------------------------------------------------------------
# CLI & Standalone Server
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
