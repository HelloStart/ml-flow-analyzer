from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_PATH = Path(__file__).with_name("rssi_reference_metrics.png")


def main() -> None:
    rows = [
        ["Gaussian Process Regression", "0.282", "0.867", "0.931", "0.975"],
        ["K-Nearest Neighbors", "0.344", "1.333", "1.155", "0.961"],
        ["Decision Trees", "0.549", "1.729", "1.315", "0.949"],
        ["Support Vector Regression", "0.792", "9.770", "3.126", "0.715"],
        ["Linear Regression", "3.840", "31.677", "5.628", "0.075"],
    ]
    figure, axis = plt.subplots(figsize=(8.2, 3.65), dpi=140)
    figure.patch.set_facecolor("#fffdf8")
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Model", "MAE", "MSE", "RMSE", "R2 Score"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.45, 0.12, 0.14, 0.14, 0.15],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.65)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5dd")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(weight="bold", color="#172f45")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f6f8")
        else:
            cell.set_facecolor("#ffffff")
    figure.tight_layout(pad=0.25)
    figure.savefig(OUTPUT_PATH, facecolor=figure.get_facecolor(), bbox_inches="tight")


if __name__ == "__main__":
    main()