# Semiconductor Equipment Health Monitor
### Multivariate Sensor Anomaly Detection + LLM-Powered Diagnostics

---

## Problem

Semiconductor manufacturing equipment generates continuous sensor streams — temperature, pressure, RF power, gas flow. Equipment drift or failure manifests in sensor data **before** visible defects occur on the wafer. Early detection prevents costly unplanned downtime, which can exceed $1M/hour in advanced fabs.

## Approach

A two-layer detection and diagnosis pipeline:

**Layer 1 — Anomaly Detection (ML)**
- Simulate realistic CVD chamber sensor time-series with injected fault signatures
- Statistical baseline: rolling z-score threshold per sensor
- Machine learning: Isolation Forest for multivariate anomaly detection
- Combined detection: flag events caught by either method

**Layer 2 — Diagnostic Explanation (GenAI)**
- Extract anomaly event summaries (sensor deviations, timing, severity)
- Send structured event data to LLM via API
- Generate plain-language engineering diagnosis: likely fault, physical mechanism, recommended action

## Physical Systems Context

Each sensor maps to a real physical process in a CVD chamber:

| Sensor | Baseline | Physical Meaning |
|--------|----------|-----------------|
| Temperature | 300°C | Chamber thermal stability, heater performance |
| Pressure | 10 mTorr | Gas flow dynamics, vacuum integrity |
| RF Power | 500W | Plasma generation efficiency |
| Gas Flow | 100 sccm | Precursor delivery, reaction stoichiometry |

## Results

| Method | Anomaly Points Detected | Precision | Recall |
|--------|------------------------|-----------|--------|
| Statistical (z-score) | — | — | — |
| Isolation Forest | — | — | — |
| Combined | — | — | — |

*(Run the notebook to populate results)*

## Example LLM Diagnostic Output

```
🔴 ANOMALY EVENT 1 | Time steps 200–214
   Temp: +18.0°C  Pressure: +0.0 mTorr  RF: +0.0W  Flow: +0.0 sccm

   🤖 LLM Diagnosis:
   The isolated temperature spike without corresponding changes in pressure 
   or RF power suggests a localized heater malfunction or thermal runaway 
   in a specific zone. This pattern is consistent with a failing resistive 
   heating element or degraded thermal contact. Recommended action: halt 
   the process run, inspect heater continuity and thermocouple calibration 
   before resuming.
```

## Repository Structure

```
semiconductor_health_monitor/
├── semiconductor_health_monitor.ipynb   # Main notebook
├── requirements.txt                     # Dependencies
└── README.md                           # This file
```

## Installation & Usage

```bash
pip install -r requirements.txt
jupyter notebook semiconductor_health_monitor.ipynb
```

For LLM diagnostics, the notebook calls the Anthropic API. The anomaly detection (Parts 1 & 2) runs fully without an API key.

## Next Steps

- [ ] Apply to real [SECOM dataset](https://archive.ics.uci.edu/ml/datasets/SECOM) (semiconductor manufacturing, UCI ML Repository)
- [ ] Add LSTM autoencoder for sequence-aware detection
- [ ] Fine-tune LLM on semiconductor equipment documentation
- [ ] Build interactive real-time dashboard


