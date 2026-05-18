import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.core.logging import get_logger
from src.io.paths import ProjectPaths


logger = get_logger(__name__)


def load_forward_metrics(group):
    path = ProjectPaths.forward_results_dir(group) / "metrics.csv"
    df = pd.read_csv(path)
    df["Type"] = "Forward"
    return df


def load_retro_metrics(group):
    path = ProjectPaths.retrospective_evaluation_dir(group) / "model_results.csv"

    if not path.exists():
        raise ValueError(f"Missing retrospective file: {path}")

    df = pd.read_csv(path)

    df = df.rename(columns={
        "model": "Model",
        "r2": "R2",
        "mae": "MAE",
        "rmse": "RMSE"
    })

    df["Type"] = "Retrospective"

    logger.info("✅ Loaded retrospective metrics from %s", path)

    return df


def plot_residuals(group):

    path = ProjectPaths.forward_results_dir(group) / "ml_predictions.csv"
    df = pd.read_csv(path)

    plot_dir = ProjectPaths.comparison_plots_dir(group)

    models = {
        "Linear Regression": "LR_Pred",
        "Random Forest": "RF_Pred",
        "Gradient Boosting": "GB_Pred"
    }

    for model_name, col in models.items():

        if col not in df.columns:
            logger.warning("⚠️ %s predictions not found, skipping", model_name)
            continue

        residuals = df["Actual"] - df[col]

        plt.figure()
        plt.hist(residuals, bins=50)

        plt.title(f"{group.upper()} Residuals ({model_name})")
        plt.xlabel("Error (Actual - Predicted)")
        plt.ylabel("Frequency")

        save_path = plot_dir / f"residuals_{col.lower().replace('_pred','')}.png"

        plt.tight_layout()
        plt.savefig(save_path)
        plt.clf()

        logger.info("✅ Saved residual plot → %s", save_path)


def compare_forward_vs_retro(group):
    forward_dir = ProjectPaths.forward_results_dir(group)
    forward = pd.read_csv(forward_dir / "metrics.csv")
    retro = pd.read_csv(forward_dir / "metrics_retro.csv")

    forward.columns = forward.columns.str.strip().str.upper()
    retro.columns = retro.columns.str.strip().str.upper()

    forward = forward.rename(columns={"MODEL": "Model"})
    retro = retro.rename(columns={"MODEL": "Model"})

    def normalise_model_name(name):
        name = name.strip().lower()

        if "random" in name:
            return "Random Forest"
        elif "gradient" in name or "boost" in name:
            return "Gradient Boosting"
        elif "linear" in name:
            return "Linear Regression"
        elif "epley" in name:
            return "Epley"
        elif "brzycki" in name:
            return "Brzycki"
        else:
            return name

    forward["Model"] = forward["Model"].apply(normalise_model_name)
    retro["Model"] = retro["Model"].apply(normalise_model_name)

    results = []

    logger.info("--- PERFORMANCE DROP (%s) ---", group.upper())

    common_models = set(forward["Model"]).intersection(set(retro["Model"]))

    logger.info("Forward models: %s", forward["Model"].unique())
    logger.info("Retro models: %s", retro["Model"].unique())
    logger.info("Common models: %s", common_models)

    for model in common_models:
        f_row = forward[forward["Model"] == model]
        r_row = retro[retro["Model"] == model]

        if f_row.empty or r_row.empty:
            logger.warning("%s | %s: ⚠️ Skipped", group.upper(), model)
            continue

        f_r2 = f_row["R2"].values[0]
        r_r2 = r_row["R2"].values[0]
        r2_diff = f_r2 - r_r2

        f_mae = f_row["MAE"].values[0]
        r_mae = r_row["MAE"].values[0]
        mae_pct = ((f_mae - r_mae) / (r_mae + 1e-6)) * 100

        f_rmse = f_row["RMSE"].values[0]
        r_rmse = r_row["RMSE"].values[0]
        rmse_pct = ((f_rmse - r_rmse) / (r_rmse + 1e-6)) * 100

        logger.info(
            "%s: ΔR²=%.3f, MAE %+.1f%%, RMSE %+.1f%%",
            model,
            r2_diff,
            mae_pct,
            rmse_pct
        )

        results.append({
            "Model": model,
            "Retro_R2": r_r2,
            "Forward_R2": f_r2,
            "Performance_Drop_R2": r2_diff,
            "MAE_Change_%": mae_pct,
            "RMSE_Change_%": rmse_pct
        })
    df_results = pd.DataFrame(results)

    if df_results.empty:
        logger.warning("⚠️ No valid models to compare.")
        return

    save_dir = ProjectPaths.comparison_plots_dir(group)

    table_path = save_dir / "performance_drop.csv"
    df_results.to_csv(table_path, index=False)

    logger.info("✅ Saved performance drop table to %s", table_path)

    plt.figure()

    df_results = df_results.sort_values("Performance_Drop_R2")

    plt.bar(df_results["Model"], df_results["Performance_Drop_R2"])

    for i, v in enumerate(df_results["Performance_Drop_R2"]):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

    plt.title(f"{group.upper()} – Performance Drop (Forward vs Retrospective)")
    plt.xlabel("Model")
    plt.ylabel("ΔR² (Forward − Retrospective)")
    plt.xticks(rotation=30)

    plot_path = save_dir / "performance_drop.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()

    logger.info("✅ Saved performance drop plot to %s", plot_path)

    plt.figure()

    plt.bar(df_results["Model"], df_results["MAE_Change_%"])

    for i, v in enumerate(df_results["MAE_Change_%"]):
        plt.text(i, v, f"{v:.1f}%", ha='center', va='bottom')

    plt.title(f"{group.upper()} – MAE Change (Forward vs Retrospective)")
    plt.ylabel("MAE Change (%)")
    plt.xticks(rotation=30)

    plot_path = save_dir / "mae_Change.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()

    logger.info("✅ Saved MAE increase plot to %s", plot_path)

    plt.figure()

    plt.bar(df_results["Model"], df_results["RMSE_Change_%"])

    for i, v in enumerate(df_results["RMSE_Change_%"]):
        plt.text(i, v, f"{v:.1f}%", ha='center', va='bottom')

    plt.title(f"{group.upper()} – RMSE Change (Forward vs Retrospective)")
    plt.ylabel("RMSE Change (%)")
    plt.xticks(rotation=30)

    plot_path = save_dir / "rmse_Change.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()

    logger.info("✅ Saved RMSE increase plot to %s", plot_path)


def statistical_tests(group):

    forward = pd.read_csv(ProjectPaths.forward_results_dir(group) / "metrics.csv")

    logger.info("--- MODEL COMPARISON (%s) ---", group.upper())

    for _, row in forward.iterrows():
        logger.info("%s: R2 = %.3f, MAE = %.2f", row["Model"], row["R2"], row["MAE"])


def error_by_strength_level(group):

    path = ProjectPaths.forward_results_dir(group) / "ml_predictions.csv"
    df = pd.read_csv(path)

    plot_dir = ProjectPaths.comparison_plots_dir(group)

    df["Strength_Level"] = pd.qcut(df["Actual"], q=3, labels=["Low", "Medium", "High"])

    results = []

    models = {
        "Random Forest": "RF_Pred",
        "Gradient Boosting": "GB_Pred"
    }

    for level in ["Low", "Medium", "High"]:
        subset = df[df["Strength_Level"] == level]

        for model_name, col in models.items():

            if col not in subset.columns:
                continue

            mae = np.mean(np.abs(subset["Actual"] - subset[col]))

            results.append({
                "Strength_Level": level,
                "Model": model_name,
                "MAE": mae
            })

    df_results = pd.DataFrame(results)

    save_path = plot_dir / "error_by_strength.csv"
    df_results.to_csv(save_path, index=False)

    logger.info("✅ Saved strength-level error table → %s", save_path)

    plt.figure()

    for model in df_results["Model"].unique():
        subset = df_results[df_results["Model"] == model]
        plt.plot(subset["Strength_Level"], subset["MAE"], marker="o", label=model)

    plt.title(f"{group.upper()} – Error by Strength Level (MAE)")
    plt.xlabel("Strength Level")
    plt.ylabel("MAE")
    plt.legend()

    plot_path = plot_dir / "error_by_strength.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()

    logger.info("✅ Saved strength-level plot → %s", plot_path)


def plot_metric_comparison(group, metric):

    forward_df = load_forward_metrics(group)
    retro_df = load_retro_metrics(group)

    df = pd.concat([retro_df, forward_df])

    plt.figure()

    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        plt.plot(subset["Type"], subset[metric], marker="o", label=model)

    plt.title(f"{group.upper()} – {metric} Comparison (Retrospective vs Forward)")
    plt.xlabel("Evaluation Type")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()

    plot_dir = ProjectPaths.comparison_plots_dir(group)

    save_path = plot_dir / f"{metric.lower()}_comparison.png"
    plt.savefig(save_path)
    plt.clf()

    logger.info("✅ Saved plot to %s", save_path)


def main():
    for group in ["raw", "wraps"]:
        plot_metric_comparison(group, "R2")
        plot_metric_comparison(group, "MAE")
        plot_metric_comparison(group, "RMSE")
        plot_residuals(group)
        statistical_tests(group)
        compare_forward_vs_retro(group)
        error_by_strength_level(group)


if __name__ == "__main__":
    main()
