"""Experiment 1: Dataset analysis and preprocessing.

Supports CSV and Excel input. Produces a cleaned CSV, a report, and basic plots.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json


def _read_table(file_path: str | Path):
    import pandas as pd
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # utf-8-sig handles CSVs exported from Excel with a BOM.
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Upload CSV, XLSX, or XLS.")


def analyze_dataset(file_path: str | Path, output_dir: str | Path = "outputs/exp01") -> dict[str, Any]:
    """Analyze, clean and export a tabular dataset.

    Numeric missing values are filled with their column median; categorical missing
    values are filled with the most frequent value (or 'Unknown' when empty).
    Duplicate rows are removed. The input file is never overwritten.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _read_table(file_path)
    if df.empty:
        raise ValueError("The uploaded dataset has no rows.")

    missing_before = df.isna().sum().sort_values(ascending=False)
    duplicate_count = int(df.duplicated().sum())
    cleaned = df.drop_duplicates().copy()

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in cleaned.columns if c not in numeric_cols]

    for col in numeric_cols:
        median = cleaned[col].median()
        cleaned[col] = cleaned[col].fillna(0 if pd.isna(median) else median)
    for col in categorical_cols:
        mode = cleaned[col].mode(dropna=True)
        replacement = mode.iloc[0] if not mode.empty else "Unknown"
        cleaned[col] = cleaned[col].fillna(replacement)

    clean_path = out / "cleaned_dataset.csv"
    cleaned.to_csv(clean_path, index=False)

    # Prepare an ML-ready feature matrix: standardize numeric columns and one-hot
    # encode categorical columns. This is a demonstration preprocessing transform,
    # not a supervised model fit; no target column is assumed.
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
    feature_path = out / "preprocessed_features.npy"
    feature_names_path = out / "preprocessed_feature_names.json"
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
        feature_matrix = preprocessor.fit_transform(cleaned)
        feature_names = [str(x) for x in preprocessor.get_feature_names_out()]
        np.save(feature_path, np.asarray(feature_matrix, dtype=np.float32))
        feature_names_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
        feature_shape = list(np.asarray(feature_matrix).shape)
    else:
        np.save(feature_path, np.empty((len(cleaned), 0), dtype=np.float32))
        feature_names = []
        feature_shape = [len(cleaned), 0]
        feature_names_path.write_text("[]", encoding="utf-8")

    # Missingness plot before imputation.
    missing_plot = out / "missing_values.png"
    top_missing = missing_before[missing_before > 0].head(25)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(top_missing):
        top_missing.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Missing values by column (before cleaning)")
        ax.set_xlabel("Number of missing values")
    else:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(missing_plot, dpi=140)
    plt.close(fig)

    numeric_plot = out / "numeric_distributions.png"
    plot_cols = numeric_cols[:6]
    if plot_cols:
        rows = (len(plot_cols) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(12, 3.4 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, plot_cols):
            cleaned[col].plot(kind="hist", bins=25, ax=ax, title=str(col))
            ax.set_xlabel(str(col))
        for ax in axes[len(plot_cols):]:
            ax.set_visible(False)
        fig.suptitle("Numeric feature distributions after cleaning")
        fig.tight_layout()
        fig.savefig(numeric_plot, dpi=140)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No numeric columns available for histograms", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(numeric_plot, dpi=140, bbox_inches="tight")
        plt.close(fig)

    # Correlation heatmap is useful only with two or more numeric columns.
    corr_plot = out / "correlation_heatmap.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(numeric_cols) >= 2:
        corr = cleaned[numeric_cols].corr(numeric_only=True)
        image = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
        ax.set_title("Numeric feature correlation")
        fig.colorbar(image, ax=ax, shrink=0.75)
    else:
        ax.text(0.5, 0.5, "Need at least two numeric columns for a correlation heatmap", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(corr_plot, dpi=140)
    plt.close(fig)

    report = {
        "source_file": Path(file_path).name,
        "original_rows": int(df.shape[0]),
        "original_columns": int(df.shape[1]),
        "cleaned_rows": int(cleaned.shape[0]),
        "removed_duplicate_rows": duplicate_count,
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_values_before_cleaning": {str(k): int(v) for k, v in missing_before.items()},
        "numeric_columns": [str(c) for c in numeric_cols],
        "categorical_columns": [str(c) for c in categorical_cols],
        "summary_statistics": json.loads(df.describe(include="all").fillna("").to_json(date_format="iso")),
        "preprocessing": "Removed duplicate rows; numeric missing values filled with median; categorical missing values filled with mode or 'Unknown'; numeric features standardized and categorical features one-hot encoded for ML use.",
        "preprocessed_feature_shape": feature_shape,
        "preprocessed_feature_names": feature_names,
    }
    report_path = out / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return {
        "dataframe": df,
        "cleaned_dataframe": cleaned,
        "report": report,
        "clean_csv_path": str(clean_path),
        "report_path": str(report_path),
        "features_path": str(feature_path),
        "feature_names_path": str(feature_names_path),
        "plots": [str(missing_plot), str(numeric_plot), str(corr_plot)],
    }


def summarize_dataset_with_langchain(df, huggingface_pipeline) -> str:
    """Create a concise dataset profile through LangChain + a local HF text model.

    Pass a Hugging Face text2text-generation pipeline, for example the FLAN-T5
    pipeline loaded by experiments.exp03_qa.load_qa_pipeline(). No hosted API key
    is required, though the model needs to be downloaded on first use.
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_huggingface import HuggingFacePipeline

    llm = HuggingFacePipeline(pipeline=huggingface_pipeline)
    numeric = df.select_dtypes(include="number")
    numeric_summary = numeric.describe().round(3).to_string() if len(numeric.columns) else "No numeric columns."
    profile = {
        "shape": str(df.shape),
        "columns": ", ".join(map(str, df.columns)),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "sample_rows": df.head(5).to_string(index=False),
        "numeric_summary": numeric_summary,
    }
    prompt = PromptTemplate.from_template(
        "You are a data science assistant. Write a concise dataset profile for a student. "
        "Mention dataset size, meaningful columns, missing values, numeric patterns, and "
        "reasonable next EDA steps. Do not invent facts absent from the profile.\n"
        "Shape: {shape}\nColumns: {columns}\nDtypes: {dtypes}\nMissing values: {missing_values}\n"
        "Sample rows:\n{sample_rows}\nNumeric summary:\n{numeric_summary}\n"
        "Dataset profile:"
    )
    response = (prompt | llm).invoke({k: str(v) for k, v in profile.items()})
    return str(response).strip()
