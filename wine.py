from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Callable

import joblib
import numpy as np
import yaml
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).parent
KNOWLEDGE_PATH = ROOT / "knowledge" / "wine.yaml"
ARTIFACTS_DIR = ROOT / "artifacts"

for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(font_name.lower() in font.name.lower() for font in font_manager.fontManager.ttflist):
        rcParams["font.sans-serif"] = [font_name]
        break
rcParams["axes.unicode_minus"] = False


class WineDataModel(QAbstractTableModel):
    def __init__(self, workflow: WineWorkflow) -> None:
        super().__init__()
        self.workflow = workflow
        self.headers = workflow.feature_names + ["真实标签（仅评估）"]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.workflow.features)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        if index.column() < len(self.workflow.feature_names):
            return f"{self.workflow.features[index.row(), index.column()]:.4f}"
        return str(int(self.workflow.true_labels[index.row()]))

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        return self.headers[section] if orientation == Qt.Orientation.Horizontal else section + 1


class WineWorkflow:
    def __init__(self) -> None:
        dataset = load_wine()
        self.features = dataset.data
        self.true_labels = dataset.target
        self.feature_names = list(dataset.feature_names)
        self.class_names = list(dataset.target_names)
        self.config = {
            "n_clusters": 3,
            "n_init": 10,
            "random_state": 42,
            "scaling": True,
        }
        self._prepare(self.config)

    def _prepare(self, config: dict) -> None:
        self.config = config
        self.scaler = StandardScaler().fit(self.features)
        self.scaled = self.scaler.transform(self.features)
        self.training_data = self.scaled if config["scaling"] else self.features
        self.pca = PCA(n_components=2).fit(self.scaled)
        self.pca_points = self.pca.transform(self.scaled)
        self.cluster_labels: np.ndarray | None = None

    def run(self, config: dict) -> None:
        self._prepare(config)
        started = perf_counter()
        self.model = KMeans(
            n_clusters=config["n_clusters"],
            n_init=config["n_init"],
            random_state=config["random_state"],
        )
        self.cluster_labels = self.model.fit_predict(self.training_data)
        self.fit_ms = (perf_counter() - started) * 1000
        self.silhouette = silhouette_score(self.training_data, self.cluster_labels)
        self.ari = adjusted_rand_score(self.true_labels, self.cluster_labels)
        self.nmi = normalized_mutual_info_score(self.true_labels, self.cluster_labels)
        self.inertia = float(self.model.inertia_)
        self.cluster_sizes = dict(sorted(Counter(self.cluster_labels).items()))

    @property
    def is_clustered(self) -> bool:
        return self.cluster_labels is not None

    def k_diagnostics(self) -> list[dict[str, float | int]]:
        values = []
        for clusters in range(1, 11):
            model = KMeans(
                n_clusters=clusters,
                n_init=self.config["n_init"],
                random_state=self.config["random_state"],
            ).fit(self.training_data)
            silhouette = (
                silhouette_score(self.training_data, model.labels_)
                if clusters >= 2 else float("nan")
            )
            values.append({"k": clusters, "inertia": float(model.inertia_), "silhouette": silhouette})
        return values

    def cluster_comparison(self) -> tuple[np.ndarray, dict[int, int], int]:
        assert self.cluster_labels is not None
        matrix = contingency_matrix(self.true_labels, self.cluster_labels)
        class_rows, cluster_columns = linear_sum_assignment(-matrix)
        mapping = {
            int(cluster_id): int(class_id)
            for class_id, cluster_id in zip(class_rows, cluster_columns)
        }
        matched = int(matrix[class_rows, cluster_columns].sum())
        return matrix, mapping, matched

    def save(self) -> tuple[Path, Path]:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        model_path = ARTIFACTS_DIR / "wine_kmeans.pkl"
        scaler_path = ARTIFACTS_DIR / "wine_scaler.pkl"
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        return model_path, scaler_path

    def load_and_group(self, sample: np.ndarray) -> tuple[int, np.ndarray]:
        model_path, scaler_path = self.save()
        model = joblib.load(model_path)
        if self.config["scaling"]:
            scaler = joblib.load(scaler_path)
            prepared = scaler.transform([sample])[0]
        else:
            prepared = sample
        distances = np.linalg.norm(model.cluster_centers_ - prepared, axis=1)
        return int(np.argmin(distances)), distances


class WineCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7, 5), dpi=100, facecolor="#fbfaf6")
        super().__init__(self.figure)
        self.setMinimumHeight(380)

    def show(self, step_id: str, workflow: WineWorkflow) -> None:
        self.figure.clear()
        if step_id in {"explore", "evaluate"}:
            left = self.figure.add_subplot(121)
            right = self.figure.add_subplot(122)
            axes = (left, right)
        else:
            axis = self.figure.add_subplot(111)
            axes = (axis,)
        for current in axes:
            current.set_facecolor("#fbfaf6")

        if step_id == "explore":
            left.scatter(
                workflow.pca_points[:, 0],
                workflow.pca_points[:, 1],
                s=30,
                alpha=0.65,
                color="#3d7ea6",
                edgecolors="none",
            )
            left.set_title("PCA 二维投影（不使用标签、不运行聚类）")
            left.set_xlabel("PC1")
            left.set_ylabel("PC2")
            ranges = np.ptp(workflow.features, axis=0)
            order = np.argsort(ranges)[::-1]
            right.barh(
                range(len(order)),
                ranges[order],
                color="#e07a5f",
            )
            right.set_yticks(
                range(len(order)),
                [workflow.feature_names[index] for index in order],
            )
            right.invert_yaxis()
            right.set_title("原始特征数值范围")
            right.set_xlabel("最大值 - 最小值")
        elif step_id == "choose-k":
            diagnostics = workflow.k_diagnostics()
            ks = [entry["k"] for entry in diagnostics]
            inertias = [entry["inertia"] for entry in diagnostics]
            axis.plot(ks, inertias, "o-", color="#3d7ea6")
            axis.axvline(workflow.config["n_clusters"], linestyle="--", color="#e07a5f")
            axis.set_title("肘部法：K 与 Inertia")
            axis.set_xlabel("簇数量 K")
            axis.set_ylabel("Inertia（越小越紧凑）")
            axis.set_xticks(ks)
        elif step_id == "experiment":
            assert workflow.cluster_labels is not None
            for cluster_id in sorted(workflow.cluster_sizes):
                mask = workflow.cluster_labels == cluster_id
                axis.scatter(workflow.pca_points[mask, 0], workflow.pca_points[mask, 1], s=36, alpha=0.8, label=f"Cluster {cluster_id}")
            axis.set_title(f"K-Means 聚类结果 · K={workflow.config['n_clusters']}")
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            axis.legend(frameon=False)
        elif step_id == "evaluate":
            assert workflow.cluster_labels is not None
            matrix, _, _ = workflow.cluster_comparison()
            image = left.imshow(matrix, cmap="YlGnBu")
            left.set_title("真实类别 × 聚类编号交叉表")
            left.set_xlabel("Cluster ID")
            left.set_ylabel("真实类别")
            left.set_xticks(range(matrix.shape[1]), [f"Cluster {i}" for i in range(matrix.shape[1])])
            left.set_yticks(range(matrix.shape[0]), workflow.class_names)
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    left.text(column, row, str(matrix[row, column]), ha="center", va="center")
            self.figure.colorbar(image, ax=left, fraction=0.046, pad=0.04)
            metrics = [workflow.silhouette, workflow.ari, workflow.nmi]
            bars = right.bar(["Silhouette", "ARI", "NMI"], metrics, color=["#3d7ea6", "#e07a5f", "#6a994e"])
            right.set_ylim(0, 1.05)
            right.set_title("聚类评估指标")
            for bar, value in zip(bars, metrics):
                right.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
        elif step_id == "select":
            diagnostics = workflow.k_diagnostics()
            axis.plot([d["k"] for d in diagnostics], [d["inertia"] for d in diagnostics], "o-", color="#3d7ea6")
            axis.axvline(workflow.config["n_clusters"], linestyle="--", color="#e07a5f", label=f"当前 K={workflow.config['n_clusters']}")
            axis.set_title("选择结果：肘部与当前 K")
            axis.set_xlabel("K")
            axis.set_ylabel("Inertia")
            axis.legend(frameon=False)
        elif step_id == "deploy":
            sample = workflow.features[0]
            cluster_id, distances = workflow.load_and_group(sample)
            bars = axis.bar([f"Cluster {i}" for i in range(len(distances))], distances, color="#3d7ea6")
            bars[cluster_id].set_color("#e07a5f")
            axis.set_title("样本 0 到已保存簇中心的距离")
            axis.set_xlabel("K-Means 簇编号")
            axis.set_ylabel("欧氏距离（越小越近）")
            axis.text(
                cluster_id,
                distances[cluster_id],
                "最近中心",
                ha="center",
                va="bottom",
                color="#b84e35",
            )
        else:
            axis.axis("off")
        for current in axes:
            current.grid(axis="y", alpha=0.18)
        self.figure.tight_layout()
        self.draw()


class WineTaskPage(QWidget):
    def __init__(self, on_back: Callable[[], None], on_ask_ai: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.on_back = on_back
        self.on_ask_ai = on_ask_ai
        self.content = yaml.safe_load(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        self.workflow = WineWorkflow()
        self.experiment_history: list[dict[str, str | float | int]] = []
        self.experiment_run_count = 0
        self._build_ui()
        self._apply_style()
        self._show_overview()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        header = QHBoxLayout()
        title = QLabel("Wine 无监督聚类")
        title.setObjectName("wineTitle")
        header.addWidget(title)
        header.addStretch()
        ask_ai = QPushButton("问 AI")
        ask_ai.setVisible(self.on_ask_ai is not None)
        if self.on_ask_ai is not None:
            ask_ai.clicked.connect(self.on_ask_ai)
        header.addWidget(ask_ai)
        back = QPushButton("返回实践目录")
        back.clicked.connect(self.on_back)
        header.addWidget(back)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame()
        left.setObjectName("winePanel")
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._heading("任务介绍"))
        self.overview_button = QPushButton("任务概览\n目标、数据与引出问题")
        self.overview_button.setObjectName("wineOverviewButton")
        self.overview_button.clicked.connect(self._show_overview)
        left_layout.addWidget(self.overview_button)
        left_layout.addWidget(self._heading("聚类学习流程"))
        self.step_list = QListWidget()
        for step in self.content["steps"]:
            self.step_list.addItem(QListWidgetItem(f"{step['title']}\n{step['short']}"))
        self.step_list.currentRowChanged.connect(self._step_changed)
        left_layout.addWidget(self.step_list)
        splitter.addWidget(left)

        center = QFrame()
        center.setObjectName("winePanel")
        center_layout = QVBoxLayout(center)
        control = QHBoxLayout()
        self.center_title = self._heading("任务概览")
        control.addWidget(self.center_title)
        control.addStretch()
        self.parameters = QFrame()
        parameter_layout = QHBoxLayout(self.parameters)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.k_input = QSpinBox(); self.k_input.setRange(2, 8); self.k_input.setValue(3); self.k_input.setPrefix("K ")
        self.n_init_input = QSpinBox(); self.n_init_input.setRange(1, 50); self.n_init_input.setValue(10); self.n_init_input.setPrefix("n_init ")
        self.seed_input = QSpinBox(); self.seed_input.setRange(0, 9999); self.seed_input.setValue(42); self.seed_input.setPrefix("seed ")
        self.scaling_input = QCheckBox("标准化"); self.scaling_input.setChecked(True)
        for widget in (self.k_input, self.n_init_input, self.seed_input, self.scaling_input):
            parameter_layout.addWidget(widget)
        reset_button = QPushButton("恢复默认")
        reset_button.clicked.connect(self._reset_experiment)
        parameter_layout.addWidget(reset_button)
        run_button = QPushButton("运行聚类"); run_button.clicked.connect(self._run)
        parameter_layout.addWidget(run_button)
        control.addWidget(self.parameters)
        center_layout.addLayout(control)
        self.history_title = self._heading("本次会话聚类记录（最多保留 8 次）")
        center_layout.addWidget(self.history_title)
        self.history_table = QTableWidget()
        self.history_table.setObjectName("wineHistoryTable")
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setWordWrap(False)
        self.history_table.setMaximumHeight(150)
        center_layout.addWidget(self.history_table)
        self._show_experiment_history()

        self.center_stack = QStackedWidget()
        self.data_view = QTableView()
        self.data_view.setAlternatingRowColors(True)
        self.data_model = WineDataModel(self.workflow)
        self.data_view.setModel(self.data_model)
        self.data_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.data_view.horizontalHeader().setDefaultSectionSize(120)
        self.center_stack.addWidget(self.data_view)
        self.canvas = WineCanvas()
        self.center_stack.addWidget(self.canvas)
        center_layout.addWidget(self.center_stack, 1)
        self.status = QLabel()
        self.status.setObjectName("wineStatus")
        self.status.setWordWrap(True)
        center_layout.addWidget(self.status)
        splitter.addWidget(center)

        right = QFrame()
        right.setObjectName("winePanel")
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._heading("知识讲解"))
        self.knowledge = QTextBrowser()
        right_layout.addWidget(self.knowledge)
        splitter.addWidget(right)
        splitter.setSizes([240, 700, 430])
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("wineHeading")
        return label

    def _show_overview(self) -> None:
        self.step_list.setCurrentRow(-1)
        self.overview_button.setProperty("active", True)
        self.overview_button.style().unpolish(self.overview_button)
        self.overview_button.style().polish(self.overview_button)
        self.parameters.hide()
        self.history_title.hide()
        self.history_table.hide()
        self.status.show()
        self.center_title.setText(self.content["task"]["title"])
        self._show_data()
        task = self.content["task"]
        self.status.setText("当前阶段 · 只认识任务与数据，尚未选择具体解决方法。")
        self.knowledge.setHtml(self._task_html())

    def _step_changed(self, row: int) -> None:
        if row >= 0:
            self._show_step(self.content["steps"][row])

    def ai_context(self) -> str:
        task = self.content["task"]
        step = self.content["steps"][self.step_list.currentRow()] if self.step_list.currentRow() >= 0 else None
        sections = [
            "[ML Flow Analyzer 当前任务]",
            f"任务: {task['title']}",
            f"当前页面: {step['title'] if step else '任务概览'}",
            f"任务说明: {task['subtitle']}",
            f"数据概况: {task['dataset_summary']}",
        ]
        if step is not None:
            sections.extend([f"学习目标: {step['goal']}", f"页面说明: {step['explanation']}"])
        if self.workflow.is_clustered:
            sections.append(
                f"当前聚类: K={self.workflow.config['n_clusters']}; 标准化={'开启' if self.workflow.config['scaling'] else '关闭'}; "
                f"Inertia={self.workflow.inertia:.2f}; Silhouette={self.workflow.silhouette:.4f}; "
                f"ARI={self.workflow.ari:.4f}; NMI={self.workflow.nmi:.4f}; 簇大小={self.workflow.cluster_sizes}"
            )
        return "\n".join(sections)

    def _run(self) -> None:
        self.status.setText("正在使用当前参数重新执行 K-Means 和聚类评估……")
        QApplication.processEvents()
        self.workflow.run({
            "n_clusters": self.k_input.value(),
            "n_init": self.n_init_input.value(),
            "random_state": self.seed_input.value(),
            "scaling": self.scaling_input.isChecked(),
        })
        self._record_experiment()
        self._step_changed(self.step_list.currentRow())

    def _reset_experiment(self) -> None:
        self.k_input.setValue(3)
        self.n_init_input.setValue(10)
        self.seed_input.setValue(42)
        self.scaling_input.setChecked(True)
        self.experiment_history.clear()
        self.experiment_run_count = 0
        self._show_experiment_history()

    def _show_step(self, step: dict) -> None:
        self.overview_button.setProperty("active", False)
        self.overview_button.style().unpolish(self.overview_button)
        self.overview_button.style().polish(self.overview_button)
        self.center_title.setText(step["title"])
        self.parameters.setVisible(step["id"] in {"choose-k", "experiment", "evaluate"})
        self.status.setVisible(step["id"] not in {"select", "deploy"})
        self.history_title.setVisible(step["id"] == "experiment")
        self.history_table.setVisible(step["id"] == "experiment")
        needs_clustering = step["id"] in {"experiment", "evaluate", "select", "deploy"}
        if needs_clustering and not self.workflow.is_clustered:
            self.status.setText("正在使用当前参数执行 K-Means 聚类……")
            QApplication.processEvents()
            self.workflow.run({
                "n_clusters": self.k_input.value(),
                "n_init": self.n_init_input.value(),
                "random_state": self.seed_input.value(),
                "scaling": self.scaling_input.isChecked(),
            })
        if step["id"] == "data":
            self._show_data()
        elif step["id"] == "choose-k":
            self.center_stack.setCurrentWidget(self.canvas)
            self.canvas.show(step["id"], self.workflow)
        else:
            self.center_stack.setCurrentWidget(self.canvas)
            self.canvas.show(step["id"], self.workflow)
        if step["id"] == "data":
            self.status.setText("任务与数据 · 阅读任务背景、数据来源和 13 个化学特征。")
        elif step["id"] == "explore":
            self.status.setText("数据探索 · 查看无标签 PCA 投影与各特征的原始数值范围。")
        elif step["id"] == "choose-k":
            self.status.setText(
                "K 值诊断 · 对比不同 K 对应的 Inertia 与 Silhouette，寻找合理候选值。"
            )
        elif step["id"] not in {"select", "deploy"}:
            self.status.setText(
                f"实际运行 · K={self.workflow.config['n_clusters']} · 标准化 {'开启' if self.workflow.config['scaling'] else '关闭'} · "
                f"fit {self.workflow.fit_ms:.2f} ms · Silhouette {self.workflow.silhouette:.4f} · "
                f"ARI {self.workflow.ari:.4f} · NMI {self.workflow.nmi:.4f}"
            )
        self.knowledge.setHtml(self._knowledge_html(step))

    def _record_experiment(self) -> None:
        self.experiment_run_count += 1
        self.experiment_history.append({
            "run": self.experiment_run_count,
            "k": self.workflow.config["n_clusters"],
            "scaling": "开启" if self.workflow.config["scaling"] else "关闭",
            "n_init": self.workflow.config["n_init"],
            "seed": self.workflow.config["random_state"],
            "inertia": self.workflow.inertia,
            "silhouette": self.workflow.silhouette,
            "sizes": ", ".join(f"C{cluster}: {size}" for cluster, size in self.workflow.cluster_sizes.items()),
            "ari": self.workflow.ari,
            "nmi": self.workflow.nmi,
        })
        self.experiment_history = self.experiment_history[-8:]
        self._show_experiment_history()

    def _show_experiment_history(self) -> None:
        headers = ["运行", "K", "标准化", "n_init", "seed", "Inertia", "Silhouette", "簇大小", "ARI*", "NMI*"]
        self.history_table.setColumnCount(len(headers))
        self.history_table.setHorizontalHeaderLabels(headers)
        self.history_table.setRowCount(len(self.experiment_history))
        for row, record in enumerate(reversed(self.experiment_history)):
            values = [
                str(record["run"]), str(record["k"]), str(record["scaling"]),
                str(record["n_init"]), str(record["seed"]), f"{record['inertia']:.2f}",
                f"{record['silhouette']:.4f}", str(record["sizes"]),
                f"{record['ari']:.4f}", f"{record['nmi']:.4f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.history_table.setItem(row, column, item)
        header = self.history_table.horizontalHeader()
        for column in (0, 1, 2, 3, 4, 5, 6, 8, 9):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.history_table.setToolTip("ARI*、NMI* 使用隐藏真实标签，仅用于聚类完成后的教学对照。")

    def _show_data(self) -> None:
        self.center_stack.setCurrentWidget(self.data_view)

    def _task_html(self) -> str:
        task = self.content["task"]
        rows = "".join(f"<tr><td>{item['feature']}</td><td>{item['meaning']}</td></tr>" for item in task["feature_table"])
        return (
            f"<h2>1. 当前任务</h2><p>{task['current_task']}</p>"
            f"<h2>2. 当前数据</h2><p>{task['current_data']}</p><p><b>来源：</b>{task['source']}</p>"
            "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
            f"<tr><th>特征</th><th>含义</th></tr>{rows}</table>"
            f"<h2>3. 引出的问题</h2><p>{task['method_intro']}</p><p>{task['question']}</p>"
        )

    def _knowledge_html(self, step: dict) -> str:
        task = self.content["task"]
        explanation = step.get("explanation", "")
        base = (
            f"<h2>{step['title']}：{step['short']}</h2>"
            f"<p><b>目标：</b>{step['goal']}</p>"
            + (f"<p>{explanation}</p>" if explanation else "")
        )
        if step["id"] == "data":
            return (
                self._task_html()
                + "<h3>本节采用的尝试</h3>"
                + f"<p>{step['explanation']}</p>"
                + "<p>此时只是决定尝试聚类，还没有选择 K，也没有运行 K-Means。"
                "下一步先探索特征量纲与数据形状。</p>"
            )
        if step["id"] == "explore":
            explained = self.workflow.pca.explained_variance_ratio_
            ranges = np.ptp(self.workflow.features, axis=0)
            widest = np.argsort(ranges)[::-1][:3]
            narrowest = np.argsort(ranges)[:3]
            return (
                base
                + f"<p>{step['pca_intro']}</p>"
                + "<h3>PCA 关键代码</h3>"
                + f"<pre style='background:#f1f3f2; padding:10px;'>{escape(step['pca_code'])}</pre>"
                + "<h3>本次实际输出</h3>"
                + f"<p>PCA 降维后形状：<b>{self.workflow.pca_points.shape}</b><br>"
                + f"两个主成分解释方差比例：<b>{np.round(explained, 4).tolist()}</b><br>"
                + f"累计解释方差：<b>{explained.sum():.4f}（{explained.sum():.1%}）</b></p>"
                + "<h3>左图：只看数据形状</h3>"
                + f"<p>PCA 将 13 维标准化数据投影到 2 维，PC1 和 PC2 分别解释 "
                f"{explained[0]:.1%}、{explained[1]:.1%} 的方差，累计 {explained.sum():.1%}。"
                "所有点使用同一种颜色，既没有使用真实标签，也没有显示聚类结果。"
                "它只能帮助观察数据是否可能存在结构，不能据此宣布已经发现了几个簇。</p>"
                + "<h3>右图：为什么后续需要标准化</h3>"
                + f"<p>原始范围最大的特征包括 {', '.join(self.workflow.feature_names[i] for i in widest)}；"
                f"范围较小的包括 {', '.join(self.workflow.feature_names[i] for i in narrowest)}。"
                "如果直接计算欧氏距离，大范围特征会占据主导。标准化让各特征以可比较的尺度"
                "参与后续距离计算。</p>"
                + "<p><b>阶段边界：</b>本步骤只完成数据观察。K 值选择在下一步进行，"
                "K-Means 分组结果从“聚类实验”开始显示。</p>"
                + f"<h3>探索结论</h3><p>{step['conclusion']}</p>"
            )
        if step["id"] == "choose-k":
            diagnostics = self.workflow.k_diagnostics()
            elbow_observations = "".join(
                f"<li>{item}</li>" for item in step["elbow_observation"]
            )
            algorithm_rows = "".join(
                f"<tr><td><b>{item['algorithm']}</b></td><td>{item['fit']}</td>"
                f"<td>{item['reason']}</td></tr>"
                for item in step["algorithm_choice"]
            )
            rows = "".join(
                f"<tr><td>{d['k']}</td><td>{d['inertia']:.2f}</td>"
                f"<td>{self._format_silhouette(float(d['silhouette']))}</td></tr>"
                for d in diagnostics
            )
            return (
                base
                + "<h3>为什么本节先选 K-Means</h3>"
                + "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
                + f"<tr><th>算法</th><th>本节选择</th><th>原因</th></tr>{algorithm_rows}</table>"
                + f"<p>{step['choice_conclusion']}</p>"
                + "<h3>选择 K：肘部与轮廓系数</h3>"
                + "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
                + f"<tr><th>K</th><th>Inertia</th><th>Silhouette</th></tr>{rows}</table>"
                + f"<h3>从图中可以看出</h3><ul>{elbow_observations}</ul>"
                + f"<p style='background:#fff8e8; padding:8px;'><b>标签边界：</b>"
                + f"{step['label_comparison']}</p>"
            )
        if step["id"] == "experiment":
            return base + f"<p><b>当前参数：</b>K={self.workflow.config['n_clusters']}，n_init={self.workflow.config['n_init']}，seed={self.workflow.config['random_state']}，标准化={'开启' if self.workflow.config['scaling'] else '关闭'}。</p><p><b>各簇样本数：</b>{self.workflow.cluster_sizes}</p><p><b>执行证据：</b>KMeans.fit_predict() 实测 {self.workflow.fit_ms:.2f} ms。</p>"
        if step["id"] == "evaluate":
            notes = self.content["metric_notes"]
            metric_rows = "".join(
                f"<tr><td><b>{item['name']}</b></td><td>{item['requires_labels']}</td>"
                f"<td>{item['measures']}</td><td>{item['range']}</td></tr>"
                for item in self.content["metric_table"]
            )
            result_rows = (
                f"<tr><td>Silhouette Score</td><td>{self.workflow.silhouette:.4f}</td>"
                f"<td>{self._silhouette_interpretation(self.workflow.silhouette)}</td></tr>"
                f"<tr><td>ARI</td><td>{self.workflow.ari:.4f}</td>"
                f"<td>{self._agreement_interpretation(self.workflow.ari)}</td></tr>"
                f"<tr><td>NMI</td><td>{self.workflow.nmi:.4f}</td>"
                f"<td>{self._agreement_interpretation(self.workflow.nmi)}</td></tr>"
            )
            matrix, mapping, matched = self.workflow.cluster_comparison()
            class_to_cluster = {class_id: cluster_id for cluster_id, class_id in mapping.items()}
            class_rows = []
            for class_id, class_name in enumerate(self.workflow.class_names):
                counts = ", ".join(
                    f"Cluster {cluster_id}: {int(matrix[class_id, cluster_id])}"
                    for cluster_id in range(matrix.shape[1])
                    if matrix[class_id, cluster_id] > 0
                )
                mapped_cluster = class_to_cluster.get(class_id)
                mapped_text = (
                    f"主要对应 Cluster {mapped_cluster}"
                    if mapped_cluster is not None else
                    "当前 K 下没有独立的一一映射簇"
                )
                class_rows.append(
                    f"<li><b>{class_name}（{int(matrix[class_id].sum())} 个）：</b>"
                    f"{counts}；{mapped_text}。</li>"
                )
            mapping_text = "，".join(
                f"Cluster {cluster_id} → {self.workflow.class_names[class_id]}"
                for cluster_id, class_id in sorted(mapping.items())
            )
            agreement = matched / len(self.workflow.true_labels)
            return (
                base
                + "<h3>评估指标</h3>"
                + "<p>与监督分类不同，聚类训练时没有可直接使用的正确答案。评估分为"
                "不依赖标签的内部指标，以及仅用于教学对照的外部指标。</p>"
                + "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
                + f"<tr><th>指标</th><th>需要真实标签</th><th>衡量内容</th><th>范围</th></tr>{metric_rows}</table>"
                + "<h3>当前结果解读</h3>"
                + "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
                + f"<tr><th>指标</th><th>当前值</th><th>含义</th></tr>{result_rows}</table>"
                + "<h3>为什么 ARI/NMI 很高，但 Silhouette 较低？</h3>"
                + f"<p>Silhouette 只看 13 维标准化特征中的几何结构，{notes['silhouette']}"
                "边界样本、非球形结构或簇内离散都会拉低该值。ARI/NMI 则比较聚类分组和"
                "隐藏真实标签，回答的是“分组与已知品种有多一致”。因此两类指标可以同时成立，"
                "并不矛盾。</p>"
                + "<h3>真实类别 × 聚类编号交叉表怎么读</h3>"
                + f"<ul>{''.join(class_rows)}</ul>"
                + f"<p><b>最佳一一映射：</b>{mapping_text}。</p>"
                + f"<p>按该映射，<b>{matched}/{len(self.workflow.true_labels)}"
                f"（{agreement:.1%}）</b>样本与真实类别一致。这个数值是聚类完成后的教学对照一致率，"
                "不是 K-Means 训练时优化的分类准确率。</p>"
                + "<p><b>关键边界：</b>簇编号是任意的。换一个随机种子后编号可能交换，"
                "ARI/NMI 不受这种编号置换影响。</p>"
            )
        if step["id"] == "select":
            return base + f"<p><b>当前选择：</b>标准化={'开启' if self.workflow.config['scaling'] else '关闭'}，K={self.workflow.config['n_clusters']}。</p><p>Silhouette {self.workflow.silhouette:.4f}，ARI {self.workflow.ari:.4f}，NMI {self.workflow.nmi:.4f}。</p><p><b>关键边界：</b>簇编号可以因初始化而变化，不能把 Cluster 0 直接解释为 class_0。</p>"
        sample_index = 0
        sample = self.workflow.features[sample_index]
        cluster_id, distances = self.workflow.load_and_group(sample)
        model_path, scaler_path = self.workflow.save()
        input_rows = "".join(
            f"<tr><td>{name}</td><td>{value:.4f}</td></tr>"
            for name, value in zip(self.workflow.feature_names, sample)
        )
        return (
            base
            + f"<p><b>实际产物：</b>{model_path.name}、{scaler_path.name}</p>"
            + f"<h3>本次输入：数据表第 {sample_index + 1} 条 Wine 样本</h3>"
            + "<p>输入不是品种名称，而是下面 13 项原始化学分析值。程序使用训练时保存的 "
            "Scaler 将它们变换到同一尺度，再交给已保存的 K-Means 模型。</p>"
            + "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
            + f"<tr><th>化学特征</th><th>输入值</th></tr>{input_rows}</table>"
            + "<h3>中间图：距离决定分组</h3>"
            + "<p>每根柱表示该输入样本到一个已保存簇中心的欧氏距离；橙色柱是最短距离，"
            "因此对应本次分组结果。距离不是类别概率，也不表示葡萄酒质量。</p>"
            + f"<p><b>各中心距离：</b>{np.round(distances, 4).tolist()}<br>"
            + f"<b>输出：</b>最近簇 Cluster {cluster_id}。</p>"
            + "<p>输出是聚类编号，不是葡萄酒品种名称。部署时只需保存 Scaler 与 K 个 13 维簇中心。</p>"
        )

    @staticmethod
    def _format_silhouette(value: float) -> str:
        return "-" if np.isnan(value) else f"{value:.4f}"

    @staticmethod
    def _silhouette_interpretation(value: float) -> str:
        if value >= 0.5:
            return "簇结构较清晰，簇内较紧凑、簇间分离较好"
        if value >= 0.25:
            return "存在一定簇结构，但边界与重叠仍较明显"
        return "簇结构较弱，需要重新检查 K、缩放方式或算法假设"

    @staticmethod
    def _agreement_interpretation(value: float) -> str:
        if value >= 0.8:
            return "与隐藏真实类别高度一致"
        if value >= 0.5:
            return "与隐藏真实类别有一定一致性"
        return "与隐藏真实类别一致性较低"

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget { color: #28323c; font-size: 13px; }
            WineTaskPage { background: #f3f1ea; }
            QLabel#wineTitle { color: #243b53; font-size: 24px; font-weight: 700; }
            QFrame#winePanel { background: #fbfaf6; border: 1px solid #d9d5ca; }
            QPushButton#wineOverviewButton { text-align: left; background: #edf2f5; color: #243b53; border-left: 4px solid #3d7ea6; padding: 10px; }
            QPushButton#wineOverviewButton[active="true"] { background: #dbe9ef; border-left: 4px solid #e07a5f; }
            QLabel#wineHeading { color: #243b53; font-size: 17px; font-weight: 700; }
            QLabel#wineStatus { background: #edf5f2; border: 1px solid #c9dfd7; padding: 9px; color: #276749; }
            QListWidget, QTextBrowser, QTableView { background: #fffdf8; border: none; }
            QListWidget::item { padding: 11px 8px; border-bottom: 1px solid #e6e1d7; }
            QListWidget::item:selected { background: #f2cc8f; color: #243b53; border-left: 4px solid #e07a5f; }
            QPushButton { background: #243b53; color: white; border: none; padding: 8px 13px; }
            QHeaderView::section { background: #e8eef0; color: #243b53; padding: 7px; font-weight: 600; }
        """)
