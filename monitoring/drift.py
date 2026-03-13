"""
DiabéticaAI — Data Drift Monitoring Engine
============================================
Production MLOps strategy for detecting when real-world patient
data diverges from training distribution, triggering model retraining alerts.

Strategy:
  1. Population Stability Index (PSI) — industry-standard drift metric
  2. Kolmogorov-Smirnov test for continuous features
  3. Chi-Squared test for categorical/binary features
  4. Jensen-Shannon divergence for probability distributions
  5. Prediction drift — monitor output probability distribution shift

Deployment:
  - Run as a daily/weekly batch job against incoming EHR data
  - Alerts via PagerDuty / Slack webhook / email on threshold breach
  - Logs stored for regulatory audit trail
"""

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
logger = logging.getLogger("DiabéticaAI.DriftMonitor")


# ─────────────────────────────────────────────
#  POPULATION STABILITY INDEX (PSI)
# ─────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6
) -> float:
    """
    Population Stability Index (PSI):
      PSI < 0.10  → No significant shift (model stable)
      PSI 0.10–0.25 → Moderate shift (investigate)
      PSI > 0.25  → Significant shift (retrain model)

    Standard metric used by financial and clinical ML deployments.
    """
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1
    )
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current,   bins=breakpoints)

    ref_pct = ref_counts / (ref_counts.sum() + eps) + eps
    cur_pct = cur_counts / (cur_counts.sum() + eps) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def interpret_psi(psi: float) -> dict:
    if psi < 0.10:
        return {"severity": "NONE",     "action": "Monitor normally",          "color": "#22c55e"}
    elif psi < 0.25:
        return {"severity": "MODERATE", "action": "Investigate root cause",    "color": "#f59e0b"}
    else:
        return {"severity": "CRITICAL", "action": "Trigger model retraining",  "color": "#ef4444"}


# ─────────────────────────────────────────────
#  FEATURE-LEVEL DRIFT DETECTION
# ─────────────────────────────────────────────

class FeatureDriftDetector:
    """
    Per-feature drift analysis combining multiple statistical tests
    for robust detection across different feature distributions.
    """

    CONTINUOUS_FEATURES = [
        "Glucose", "BMI", "Age", "BloodPressure",
        "Insulin", "SkinThickness", "DiabetesPedigreeFunction"
    ]
    CATEGORICAL_FEATURES = [
        "Pregnancies", "Smoking", "PhysicalActivity", "GeneralHealth"
    ]

    PSI_THRESHOLDS = {"warn": 0.10, "critical": 0.25}
    KS_ALPHA       = 0.05   # Significance level for KS test

    def __init__(self, reference_data: pd.DataFrame):
        """
        Args:
            reference_data: Training/baseline dataset used for comparison.
        """
        self.reference = reference_data
        logger.info(f"Drift detector initialized with {len(reference_data)} reference samples.")

    def detect(self, current_data: pd.DataFrame) -> dict:
        """
        Run full drift detection suite on incoming data.
        Returns structured report with per-feature and overall drift assessment.
        """
        report = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "reference_size":  len(self.reference),
            "current_size":    len(current_data),
            "features":        {},
            "overall":         {},
            "alert":           False,
            "recommended_action": ""
        }

        critical_drifts = 0
        total_psi = 0.0

        features_to_check = [
            f for f in self.CONTINUOUS_FEATURES + self.CATEGORICAL_FEATURES
            if f in self.reference.columns and f in current_data.columns
        ]

        for feature in features_to_check:
            ref_vals = self.reference[feature].dropna().values
            cur_vals = current_data[feature].dropna().values

            if len(ref_vals) < 30 or len(cur_vals) < 30:
                logger.warning(f"Insufficient samples for {feature}, skipping.")
                continue

            psi = compute_psi(ref_vals, cur_vals)
            interpretation = interpret_psi(psi)

            # KS test for continuous features
            if feature in self.CONTINUOUS_FEATURES:
                ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, cur_vals)
                drift_detected_ks = ks_pvalue < self.KS_ALPHA
            else:
                ks_stat, ks_pvalue, drift_detected_ks = None, None, False

            # Jensen-Shannon divergence
            ref_counts, bins = np.histogram(ref_vals, bins=20, density=True)
            cur_counts, _    = np.histogram(cur_vals, bins=bins, density=True)
            ref_prob = ref_counts / (ref_counts.sum() + 1e-9)
            cur_prob = cur_counts / (cur_counts.sum() + 1e-9)
            js_div   = float(jensenshannon(ref_prob, cur_prob))

            # Distribution statistics
            ref_mean, ref_std = float(np.mean(ref_vals)), float(np.std(ref_vals))
            cur_mean, cur_std = float(np.mean(cur_vals)), float(np.std(cur_vals))
            mean_shift_pct = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100

            report["features"][feature] = {
                "psi":            round(psi, 4),
                "psi_severity":   interpretation["severity"],
                "psi_action":     interpretation["action"],
                "ks_statistic":   round(ks_stat, 4) if ks_stat is not None else None,
                "ks_pvalue":      round(ks_pvalue, 4) if ks_pvalue is not None else None,
                "ks_drift":       drift_detected_ks,
                "js_divergence":  round(js_div, 4),
                "ref_mean":       round(ref_mean, 3),
                "cur_mean":       round(cur_mean, 3),
                "mean_shift_pct": round(mean_shift_pct, 2),
                "ref_std":        round(ref_std, 3),
                "cur_std":        round(cur_std, 3),
            }

            total_psi += psi
            if interpretation["severity"] == "CRITICAL":
                critical_drifts += 1

        # Overall assessment
        avg_psi = total_psi / max(len(features_to_check), 1)
        overall_interpretation = interpret_psi(avg_psi)

        report["overall"] = {
            "avg_psi":          round(avg_psi, 4),
            "critical_features": critical_drifts,
            "severity":         overall_interpretation["severity"],
            "action":           overall_interpretation["action"],
        }

        report["alert"] = critical_drifts > 0 or avg_psi > self.PSI_THRESHOLDS["warn"]
        if critical_drifts >= 3:
            report["recommended_action"] = "IMMEDIATE_RETRAIN"
        elif critical_drifts >= 1 or avg_psi > self.PSI_THRESHOLDS["warn"]:
            report["recommended_action"] = "SCHEDULE_RETRAIN"
        else:
            report["recommended_action"] = "CONTINUE_MONITORING"

        logger.info(f"Drift report: avg_PSI={avg_psi:.4f}, critical_features={critical_drifts}, action={report['recommended_action']}")
        return report


# ─────────────────────────────────────────────
#  PREDICTION DRIFT MONITOR
# ─────────────────────────────────────────────

class PredictionDriftMonitor:
    """
    Monitors shift in model output probability distribution.
    A model making consistently different predictions (even without
    feature drift) signals concept drift or label shift.
    """

    def __init__(self, reference_probabilities: np.ndarray):
        self.ref_probs = reference_probabilities
        self.ref_mean  = np.mean(reference_probabilities)
        self.ref_pos_rate = np.mean(reference_probabilities > 0.5)

    def check(self, current_probabilities: np.ndarray) -> dict:
        psi = compute_psi(self.ref_probs, current_probabilities)
        ks_stat, ks_pvalue = stats.ks_2samp(self.ref_probs, current_probabilities)

        cur_mean     = np.mean(current_probabilities)
        cur_pos_rate = np.mean(current_probabilities > 0.5)

        interpretation = interpret_psi(psi)

        return {
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "psi":                round(psi, 4),
            "severity":           interpretation["severity"],
            "ks_pvalue":          round(ks_pvalue, 4),
            "ref_mean_prob":      round(self.ref_mean, 4),
            "cur_mean_prob":      round(cur_mean, 4),
            "ref_positive_rate":  round(self.ref_pos_rate, 4),
            "cur_positive_rate":  round(cur_pos_rate, 4),
            "alert":              interpretation["severity"] != "NONE",
            "concept_drift_hint": (
                abs(cur_pos_rate - self.ref_pos_rate) > 0.10
            ),
        }


# ─────────────────────────────────────────────
#  DRIFT REPORTER (scheduled job entrypoint)
# ─────────────────────────────────────────────

class DriftReporter:
    """
    Orchestrates drift checks and persists reports.
    Designed to run as a daily cron/Airflow DAG task.
    """

    def __init__(self, reports_dir: str = "monitoring/reports/"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_and_save(
        self,
        feature_detector: FeatureDriftDetector,
        pred_monitor: PredictionDriftMonitor,
        current_features: pd.DataFrame,
        current_probs: np.ndarray,
    ) -> dict:
        feature_report = feature_detector.detect(current_features)
        pred_report    = pred_monitor.check(current_probs)

        full_report = {
            "run_id":          f"drift-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "feature_drift":   feature_report,
            "prediction_drift": pred_report,
            "requires_action": (
                feature_report["recommended_action"] != "CONTINUE_MONITORING" or
                pred_report["alert"]
            )
        }

        outpath = self.reports_dir / f"{full_report['run_id']}.json"
        with open(outpath, "w") as f:
            json.dump(full_report, f, indent=2)

        logger.info(f"Drift report saved → {outpath}")
        self._trigger_alerts(full_report)
        return full_report

    def _trigger_alerts(self, report: dict):
        """
        Alert dispatch. In production, replace with:
          - PagerDuty: pd.trigger_event(...)
          - Slack webhook: requests.post(SLACK_WEBHOOK, json={...})
          - Email: smtplib / SendGrid
        """
        if report["requires_action"]:
            action = report["feature_drift"]["recommended_action"]
            logger.warning(f"🚨 DRIFT ALERT — Action required: {action}")
            logger.warning(f"   Feature PSI avg: {report['feature_drift']['overall']['avg_psi']}")
            logger.warning(f"   Prediction drift: {report['prediction_drift']['severity']}")
            # TODO: dispatch to alerting system
        else:
            logger.info("✅ No significant drift detected. Model performance stable.")


# ─────────────────────────────────────────────
#  RETRAINING STRATEGY DOCUMENTATION
# ─────────────────────────────────────────────

RETRAINING_STRATEGY = """
═══════════════════════════════════════════════════════════════════
DiabéticaAI — Data Drift & Retraining Strategy
═══════════════════════════════════════════════════════════════════

MONITORING SCHEDULE:
  • Daily:   PSI check on top 5 high-impact features (Glucose, BMI, Age, ...)
  • Weekly:  Full feature + prediction drift report
  • Monthly: Manual clinical validation with subject matter experts

TRIGGER THRESHOLDS:
  • PSI > 0.10 on any feature  → Warning + Slack alert
  • PSI > 0.25 on any feature  → Critical alert + schedule retrain
  • PSI > 0.25 avg across all  → Immediate retrain flag
  • KS p-value < 0.01          → Supplementary evidence for drift
  • Positive rate shift > 10%  → Concept drift investigation

RETRAINING PIPELINE:
  1. Automated trigger via Airflow DAG
  2. Ingest last 90 days of labeled EHR data
  3. Re-run full training pipeline (smote → tune → validate)
  4. A/B shadow deployment: new model alongside old for 2 weeks
  5. Clinical validation: review 50 random predictions with endocrinologists
  6. Promote to production if F1 ≥ 0.85 AND Recall ≥ 0.90
  7. Archive old model with full audit record

REGULATORY DOCUMENTATION:
  • All drift reports retained for 7 years (FDA SaMD guidance)
  • Model change log with version, trigger, clinical reviewer sign-off
  • SHAP explanation snapshots for each model version

TOOLS RECOMMENDED:
  • MLflow: Model versioning and experiment tracking
  • Apache Airflow: Pipeline orchestration
  • Evidently AI: Advanced drift dashboard (open source)
  • Great Expectations: Data quality contracts
═══════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(RETRAINING_STRATEGY)
