"""
DiabéticaAI — Production ML Pipeline
=====================================
Senior ML Engineer Implementation
Handles: Data ingestion, preprocessing, ensemble training,
         cross-validation, SHAP explainability, risk stratification.
"""

import os
import warnings
import logging
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import shap

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DiabéticaAI.Pipeline")


# ─────────────────────────────────────────────
#  1. DATA LOADING & MULTI-DATASET INTEGRATION
# ─────────────────────────────────────────────

class DataLoader:
    """
    Supports multiple data sources:
      - Pima Indians Diabetes Dataset (baseline)
      - CDC BRFSS / Diabetes Health Indicators Dataset (100k+ records)
      - Custom EHR CSV exports
    """

    PIMA_COLUMNS = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]

    # CDC BRFSS column mapping → internal schema
    BRFSS_COLUMN_MAP = {
        "BMI5CAT": "BMI",
        "DIABETE4": "Outcome",
        "_AGE80":   "Age",
        "BLOODCHO": "Cholesterol",
        "BPHIGH6":  "BloodPressure",
        "SMOKE100": "Smoking",
        "EXERANY2": "PhysicalActivity",
        "GENHLTH":  "GeneralHealth",
    }

    def load_pima(self, path: str) -> pd.DataFrame:
        logger.info(f"Loading Pima dataset from {path}")
        df = pd.read_csv(path, names=self.PIMA_COLUMNS, header=0)
        # Flag biologically impossible zeros as NaN for imputation
        zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_invalid] = df[zero_invalid].replace(0, np.nan)
        df["dataset_source"] = "pima"
        return df

    def load_cdc_brfss(self, path: str) -> pd.DataFrame:
        """
        Load the CDC BRFSS Diabetes Health Indicators Dataset.
        Available at: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
        Approx 253,680 records.
        """
        logger.info(f"Loading CDC BRFSS dataset from {path}")
        df = pd.read_csv(path)
        # Standardize binary target: 0=No, 1=Yes (pre-diabetes mapped to 1 for sensitivity)
        if "Diabetes_012" in df.columns:
            df["Outcome"] = (df["Diabetes_012"] >= 1).astype(int)
            df.drop(columns=["Diabetes_012"], inplace=True)
        elif "Diabetes_binary" in df.columns:
            df.rename(columns={"Diabetes_binary": "Outcome"}, inplace=True)
        df["dataset_source"] = "cdc_brfss"
        return df

    def load_custom_ehr(self, path: str, target_col: str = "diabetes_label") -> pd.DataFrame:
        logger.info(f"Loading custom EHR CSV from {path}")
        df = pd.read_csv(path)
        if target_col in df.columns:
            df.rename(columns={target_col: "Outcome"}, inplace=True)
        df["dataset_source"] = "ehr"
        return df

    def merge_datasets(self, datasets: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge heterogeneous datasets on shared feature schema."""
        logger.info(f"Merging {len(datasets)} datasets")
        merged = pd.concat(datasets, axis=0, ignore_index=True, sort=False)
        logger.info(f"Merged shape: {merged.shape} | Class balance:\n{merged['Outcome'].value_counts(normalize=True)}")
        return merged


# ─────────────────────────────────────────────
#  2. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

class ClinicalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade preprocessing:
      - IterativeImputer (MICE) for missing values
      - StandardScaler for numerical normalization
      - Feature engineering: BMI class, age group, glucose category
    """

    def __init__(self, numeric_features: list[str]):
        self.numeric_features = numeric_features
        self.imputer = IterativeImputer(max_iter=10, random_state=42)
        self.scaler = StandardScaler()

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "BMI" in df.columns:
            df["BMI_Obese"]      = (df["BMI"] >= 30).astype(int)
            df["BMI_Overweight"] = ((df["BMI"] >= 25) & (df["BMI"] < 30)).astype(int)
        if "Glucose" in df.columns:
            df["Glucose_High"]   = (df["Glucose"] >= 140).astype(int)
            df["Glucose_PreDM"]  = ((df["Glucose"] >= 100) & (df["Glucose"] < 140)).astype(int)
        if "Age" in df.columns:
            df["Age_Senior"]     = (df["Age"] >= 60).astype(int)
            df["Age_MiddleAge"]  = ((df["Age"] >= 40) & (df["Age"] < 60)).astype(int)
        return df

    def fit(self, X, y=None):
        X_eng = self._engineer_features(X)
        num_cols = [c for c in self.numeric_features if c in X_eng.columns]
        self.imputer.fit(X_eng[num_cols])
        imputed = self.imputer.transform(X_eng[num_cols])
        self.scaler.fit(imputed)
        self.feature_names_out_ = list(X_eng.columns)
        self.numeric_features_used_ = num_cols
        return self

    def transform(self, X, y=None):
        X_eng = self._engineer_features(X)
        num_cols = self.numeric_features_used_
        imputed  = self.imputer.transform(X_eng[num_cols])
        scaled   = self.scaler.transform(imputed)
        X_out    = X_eng.copy()
        X_out[num_cols] = scaled
        return X_out

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


# ─────────────────────────────────────────────
#  3. ENSEMBLE MODEL TRAINING
# ─────────────────────────────────────────────

class EnsembleTrainer:
    """
    Trains XGBoost, LightGBM, RandomForest.
    Uses GridSearchCV for hyperparameter tuning.
    Final model: soft-voting ensemble for maximum clinical reliability.
    Targets: F1 > 0.85, Recall (Sensitivity) > 0.90
    """

    XGB_PARAM_GRID = {
        "n_estimators":   [200, 400],
        "max_depth":      [4, 6],
        "learning_rate":  [0.05, 0.1],
        "subsample":      [0.8, 1.0],
        "scale_pos_weight": [1, 2],    # handles class imbalance
    }

    LGBM_PARAM_GRID = {
        "n_estimators":   [200, 400],
        "max_depth":      [4, 6],
        "learning_rate":  [0.05, 0.1],
        "num_leaves":     [31, 63],
        "class_weight":   ["balanced"],
    }

    RF_PARAM_GRID = {
        "n_estimators":  [200, 500],
        "max_depth":     [None, 10, 20],
        "class_weight":  ["balanced"],
        "min_samples_split": [2, 5],
    }

    def __init__(self, cv_folds: int = 10, n_jobs: int = -1):
        self.cv_folds = cv_folds
        self.n_jobs   = n_jobs
        self.models_  = {}
        self.best_estimators_ = {}

    def _tune(self, name: str, estimator, param_grid: dict,
              X_train: np.ndarray, y_train: np.ndarray) -> object:
        logger.info(f"Tuning {name} with GridSearchCV ({self.cv_folds}-fold)…")
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        gs = GridSearchCV(
            estimator, param_grid, cv=cv,
            scoring="f1", n_jobs=self.n_jobs, verbose=0
        )
        gs.fit(X_train, y_train)
        logger.info(f"  {name} best F1={gs.best_score_:.4f} | params={gs.best_params_}")
        return gs.best_estimator_

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        # Apply SMOTE for class balancing
        logger.info("Applying SMOTE for class balancing…")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {np.bincount(y_res.astype(int))}")

        xgb_model  = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
        lgbm_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        rf_model   = RandomForestClassifier(random_state=42)

        self.best_estimators_["xgb"]  = self._tune("XGBoost",     xgb_model,  self.XGB_PARAM_GRID,  X_res, y_res)
        self.best_estimators_["lgbm"] = self._tune("LightGBM",    lgbm_model, self.LGBM_PARAM_GRID, X_res, y_res)
        self.best_estimators_["rf"]   = self._tune("RandomForest", rf_model,  self.RF_PARAM_GRID,   X_res, y_res)

        self.ensemble_ = VotingClassifier(
            estimators=[
                ("xgb",  self.best_estimators_["xgb"]),
                ("lgbm", self.best_estimators_["lgbm"]),
                ("rf",   self.best_estimators_["rf"]),
            ],
            voting="soft",
            weights=[2, 2, 1],   # slight boost for gradient boosters
            n_jobs=self.n_jobs
        )
        logger.info("Training final soft-voting ensemble…")
        self.ensemble_.fit(X_res, y_res)
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred  = self.ensemble_.predict(X_test)
        y_proba = self.ensemble_.predict_proba(X_test)[:, 1]

        metrics = {
            "f1_score":   round(f1_score(y_test, y_pred), 4),
            "recall":     round(recall_score(y_test, y_pred), 4),
            "precision":  round(precision_score(y_test, y_pred), 4),
            "roc_auc":    round(roc_auc_score(y_test, y_proba), 4),
        }
        logger.info(f"\n{'='*50}\nModel Evaluation Metrics:\n{json.dumps(metrics, indent=2)}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

        if metrics["f1_score"] < 0.85:
            logger.warning("⚠ F1-score below 0.85 target. Review data quality and feature selection.")
        if metrics["recall"] < 0.90:
            logger.warning("⚠ Recall below 0.90 target. Consider adjusting classification threshold.")

        return metrics

    def cross_validate_ensemble(self, X: np.ndarray, y: np.ndarray) -> dict:
        """10-fold cross-validation on ensemble for robust generalization estimate."""
        logger.info("Running 10-fold cross-validation on ensemble…")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_validate(
            self.ensemble_, X, y, cv=cv,
            scoring=["f1", "recall", "precision", "roc_auc"],
            n_jobs=self.n_jobs
        )
        summary = {
            "cv_f1_mean":        round(cv_results["test_f1"].mean(), 4),
            "cv_f1_std":         round(cv_results["test_f1"].std(), 4),
            "cv_recall_mean":    round(cv_results["test_recall"].mean(), 4),
            "cv_recall_std":     round(cv_results["test_recall"].std(), 4),
            "cv_roc_auc_mean":   round(cv_results["test_roc_auc"].mean(), 4),
            "cv_roc_auc_std":    round(cv_results["test_roc_auc"].std(), 4),
        }
        logger.info(f"Cross-Validation Summary:\n{json.dumps(summary, indent=2)}")
        return summary


# ─────────────────────────────────────────────
#  4. EXPLAINABILITY ENGINE (SHAP)
# ─────────────────────────────────────────────

class ExplainabilityEngine:
    """
    Generates SHAP-based explanations for individual patient predictions.
    Critical for clinical trust and regulatory compliance.
    Output: human-readable clinical narrative for each prediction.
    """

    FEATURE_LABELS = {
        "Glucose":                  "Blood Glucose Level",
        "BMI":                      "Body Mass Index",
        "Age":                      "Patient Age",
        "DiabetesPedigreeFunction": "Family History Score",
        "BloodPressure":            "Diastolic Blood Pressure",
        "Insulin":                  "Serum Insulin",
        "SkinThickness":            "Skin Fold Thickness",
        "Pregnancies":              "Number of Pregnancies",
        "BMI_Obese":                "Obese BMI Category",
        "Glucose_High":             "High Glucose (≥140 mg/dL)",
        "Age_Senior":               "Senior Age Group (≥60)",
    }

    def __init__(self, model, feature_names: list[str]):
        self.feature_names = feature_names
        # Use TreeExplainer for ensemble (fastest for tree-based models)
        # For VotingClassifier, use best XGB sub-estimator for SHAP
        try:
            self.explainer = shap.TreeExplainer(model.estimators_[0])  # XGB
            self.explainer_model = "xgboost"
        except Exception:
            self.explainer = shap.KernelExplainer(model.predict_proba, shap.sample)
            self.explainer_model = "kernel"
        logger.info(f"SHAP explainer initialized: {self.explainer_model}")

    def explain_patient(self, X_patient: np.ndarray, top_n: int = 5) -> dict:
        """Generate explanation for a single patient prediction."""
        shap_values = self.explainer.shap_values(X_patient)
        # For binary classification, use positive class
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        contributions = sorted(
            zip(self.feature_names, sv, X_patient[0]),
            key=lambda x: abs(x[1]), reverse=True
        )[:top_n]

        factors = []
        for feat, shap_val, raw_val in contributions:
            label = self.FEATURE_LABELS.get(feat, feat)
            direction = "increases" if shap_val > 0 else "decreases"
            factors.append({
                "feature":    feat,
                "label":      label,
                "value":      round(float(raw_val), 3),
                "shap_value": round(float(shap_val), 4),
                "direction":  direction,
                "impact":     "High" if abs(shap_val) > 0.2 else ("Medium" if abs(shap_val) > 0.1 else "Low")
            })

        narrative = self._build_narrative(factors)
        return {"factors": factors, "clinical_narrative": narrative}

    def _build_narrative(self, factors: list[dict]) -> str:
        high = [f for f in factors if f["impact"] == "High" and f["direction"] == "increases"]
        med  = [f for f in factors if f["impact"] == "Medium" and f["direction"] == "increases"]
        protective = [f for f in factors if f["direction"] == "decreases"]

        parts = ["Clinical risk factors identified: "]
        if high:
            parts.append("Primary drivers — " + "; ".join(
                f"{f['label']} ({f['value']:.1f})" for f in high
            ) + ". ")
        if med:
            parts.append("Contributing factors — " + "; ".join(
                f"{f['label']}" for f in med
            ) + ". ")
        if protective:
            parts.append("Protective factors — " + "; ".join(
                f"{f['label']}" for f in protective
            ) + ".")
        return "".join(parts)


# ─────────────────────────────────────────────
#  5. RISK STRATIFICATION ENGINE
# ─────────────────────────────────────────────

class RiskStratifier:
    """
    Converts raw probability into clinical risk tiers with
    actionable, evidence-based lifestyle recommendations.
    """

    TIERS = [
        {"label": "Low Risk",      "range": (0.0,  0.30), "color": "#22c55e", "code": "LOW"},
        {"label": "Moderate Risk", "range": (0.30, 0.60), "color": "#f59e0b", "code": "MODERATE"},
        {"label": "High Risk",     "range": (0.60, 0.80), "color": "#f97316", "code": "HIGH"},
        {"label": "Critical Risk", "range": (0.80, 1.01), "color": "#ef4444", "code": "CRITICAL"},
    ]

    RECOMMENDATIONS = {
        "LOW": [
            "Maintain current BMI through balanced nutrition.",
            "Perform 150 min/week moderate aerobic activity.",
            "Annual HbA1c screening recommended.",
            "Reduce processed sugar and refined carbohydrate intake.",
        ],
        "MODERATE": [
            "Consult a registered dietitian for a personalized nutrition plan.",
            "Target 5–7% body weight reduction if BMI ≥ 25.",
            "Increase physical activity to 300 min/week.",
            "Monitor fasting glucose every 3 months.",
            "Consider referral to a Diabetes Prevention Program (DPP).",
        ],
        "HIGH": [
            "Immediate referral to endocrinology for comprehensive evaluation.",
            "HbA1c and fasting plasma glucose testing within 2 weeks.",
            "Structured diabetes prevention program enrollment.",
            "Discuss metformin prophylaxis with treating physician.",
            "Daily blood glucose self-monitoring.",
            "Cardiometabolic risk assessment (lipids, blood pressure).",
        ],
        "CRITICAL": [
            "URGENT: Schedule endocrinology consultation within 48–72 hours.",
            "Immediate HbA1c, C-peptide, and autoantibody panel.",
            "Continuous glucose monitoring (CGM) device evaluation.",
            "Emergency lifestyle intervention program.",
            "Pharmacological intervention likely warranted — consult physician.",
            "Renal function and ophthalmology baseline screening.",
        ],
    }

    def stratify(self, probability: float) -> dict:
        for tier in self.TIERS:
            lo, hi = tier["range"]
            if lo <= probability < hi:
                code = tier["code"]
                return {
                    "risk_label":       tier["label"],
                    "risk_code":        code,
                    "risk_probability": round(probability, 4),
                    "risk_percentage":  f"{probability * 100:.1f}%",
                    "color":            tier["color"],
                    "recommendations":  self.RECOMMENDATIONS[code],
                    "follow_up_days":   {"LOW": 365, "MODERATE": 90, "HIGH": 14, "CRITICAL": 2}[code],
                }
        return self.stratify(min(probability, 0.9999))


# ─────────────────────────────────────────────
#  6. MODEL PERSISTENCE
# ─────────────────────────────────────────────

class ModelRegistry:
    """Handles model versioning and serialization."""

    def __init__(self, registry_path: str = "models/"):
        self.path = Path(registry_path)
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: dict, version: str = "v1.0.0"):
        out = self.path / f"diabetica_{version}.pkl"
        with open(out, "wb") as f:
            pickle.dump(artifact, f)
        logger.info(f"Model saved → {out}")
        return str(out)

    def load(self, version: str = "v1.0.0") -> dict:
        path = self.path / f"diabetica_{version}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
#  7. ORCHESTRATOR
# ─────────────────────────────────────────────

def build_and_train_pipeline(
    pima_path: Optional[str] = None,
    brfss_path: Optional[str] = None,
    output_dir: str = "models/"
) -> dict:
    """
    Full end-to-end training orchestration.
    Returns trained artifact dict for API loading.
    """
    from sklearn.model_selection import train_test_split

    loader = DataLoader()
    datasets = []

    if pima_path and Path(pima_path).exists():
        datasets.append(loader.load_pima(pima_path))
    if brfss_path and Path(brfss_path).exists():
        datasets.append(loader.load_cdc_brfss(brfss_path))

    if not datasets:
        logger.warning("No data paths provided. Generating synthetic demo data…")
        datasets.append(_generate_synthetic_demo())

    df = loader.merge_datasets(datasets) if len(datasets) > 1 else datasets[0]

    # Drop metadata
    df.drop(columns=["dataset_source"], inplace=True, errors="ignore")

    target = "Outcome"
    features = [c for c in df.columns if c != target]
    numeric_features = df[features].select_dtypes(include=np.number).columns.tolist()

    X = df[features]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ClinicalPreprocessor(numeric_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    feat_names = list(X_train_proc.columns)
    X_train_arr = X_train_proc.values
    X_test_arr  = X_test_proc.values

    trainer = EnsembleTrainer(cv_folds=10)
    trainer.train(X_train_arr, y_train.values)

    metrics    = trainer.evaluate(X_test_arr, y_test.values)
    cv_metrics = trainer.cross_validate_ensemble(X_train_arr, y_train.values)

    artifact = {
        "model":         trainer.ensemble_,
        "preprocessor":  preprocessor,
        "feature_names": feat_names,
        "metrics":       metrics,
        "cv_metrics":    cv_metrics,
        "version":       "1.0.0",
    }

    registry = ModelRegistry(output_dir)
    registry.save(artifact)
    logger.info("✅ Pipeline complete. Model ready for API deployment.")
    return artifact


def _generate_synthetic_demo(n: int = 2000) -> pd.DataFrame:
    """Synthetic data for demo/testing purposes only."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "Pregnancies":              rng.randint(0, 17, n),
        "Glucose":                  rng.normal(120, 32, n).clip(50, 250),
        "BloodPressure":            rng.normal(72, 12, n).clip(40, 130),
        "SkinThickness":            rng.normal(29, 11, n).clip(5, 80),
        "Insulin":                  rng.exponential(80, n).clip(0, 850),
        "BMI":                      rng.normal(32, 7, n).clip(15, 65),
        "DiabetesPedigreeFunction": rng.exponential(0.47, n).clip(0.07, 2.5),
        "Age":                      rng.randint(21, 81, n),
    })
    # Simple rule-based labeling for synthetic data
    risk_score = (
        (df["Glucose"] > 140).astype(int) * 2 +
        (df["BMI"] > 30).astype(int) +
        (df["Age"] > 45).astype(int) +
        rng.binomial(1, 0.1, n)
    )
    df["Outcome"] = (risk_score >= 2).astype(int)
    df["dataset_source"] = "synthetic"
    return df


if __name__ == "__main__":
    artifact = build_and_train_pipeline()
    print(f"\nFinal Metrics: {json.dumps(artifact['metrics'], indent=2)}")
    print(f"CV Metrics:    {json.dumps(artifact['cv_metrics'], indent=2)}")
