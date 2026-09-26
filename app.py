from __future__ import annotations

import sys
from collections import Counter
from html import escape
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import font_manager, rcParams
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from housing import HousingTaskPage
from wine import WineTaskPage
from ai_dialog import AiDialog

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).parent
KNOWLEDGE_PATH = ROOT / "knowledge" / "iris.yaml"
CATALOG_PATH = ROOT / "knowledge" / "catalog.yaml"
ARTIFACTS_DIR = ROOT / "artifacts"
APP_VERSION = "v0.5"
CLASSIFIER_IDS = ["knn", "logistic", "decision-tree", "svm", "random-forest"]


for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(font_name.lower() in font.name.lower() for font in font_manager.fontManager.ttflist):
        rcParams["font.sans-serif"] = [font_name]
        break
rcParams["axes.unicode_minus"] = False


class ResponsiveImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )

    def set_source(self, path: Path) -> None:
        self._source = QPixmap(str(path))
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source.isNull() or self.width() <= 0:
            return
        width = max(1, self.width() - 8)
        scaled = self._source.scaledToWidth(
            width,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setFixedHeight(scaled.height() + 8)


class IrisWorkflow:
    def __init__(self) -> None:
        iris = load_iris()
        self.features = iris.data
        self.labels = iris.target
        self.feature_names = iris.feature_names
        self.class_names = iris.target_names
        self.run_count = 0
        self.config = {
            "test_size": 0.2,
            "random_state": 42,
            "scaling": "auto",
            "stratify": True,
            "features": [0, 1, 2, 3],
            "model_params": {},
        }
        self.rerun(self.config)

    def rerun(self, config: dict) -> None:
        run_started = perf_counter()
        self.run_count += 1
        self.config = config
        selected_features = config["features"]
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.features,
            self.labels,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=self.labels if config["stratify"] else None,
        )
        train_model_x = self.train_x[:, selected_features]
        test_model_x = self.test_x[:, selected_features]
        self.scaler = StandardScaler().fit(train_model_x)
        self.train_scaled = self.scaler.transform(train_model_x)
        self.test_scaled = self.scaler.transform(test_model_x)
        params = config["model_params"]
        seed = config["random_state"]
        self.models = {
            "knn": (KNeighborsClassifier(**({"n_neighbors": 3} | params.get("knn", {}))), True),
            "logistic": (LogisticRegression(**({"max_iter": 2000} | params.get("logistic", {}))), True),
            "decision-tree": (
                DecisionTreeClassifier(random_state=seed, **params.get("decision-tree", {})),
                False,
            ),
            "svm": (
                SVC(probability=True, random_state=seed, **({"kernel": "rbf"} | params.get("svm", {}))),
                True,
            ),
            "random-forest": (
                RandomForestClassifier(
                    random_state=seed,
                    **({"n_estimators": 100} | params.get("random-forest", {})),
                ),
                False,
            ),
        }
        model_predictions = {}
        self.training_evidence = {}
        for model_id, (model, needs_scaling) in self.models.items():
            use_scaling = config["scaling"] == "on" or (
                config["scaling"] == "auto" and needs_scaling
            )
            train_features = self.train_scaled if use_scaling else train_model_x
            test_features = self.test_scaled if use_scaling else test_model_x
            fit_started = perf_counter()
            model.fit(train_features, self.train_y)
            fit_ms = (perf_counter() - fit_started) * 1000
            model_predictions[model_id] = model.predict(test_features)
            self.training_evidence[model_id] = {
                "model": type(model).__name__,
                "train_shape": train_features.shape,
                "fit_ms": fit_ms,
                "scaled": use_scaling,
            }
        rule_params = params.get("rules", {})
        self.rule_predictions = self._predict_rules(
            self.test_x,
            rule_params.get("petal_length_threshold", 2.45),
            rule_params.get("petal_width_threshold", 1.75),
        )
        kmeans_params = {"n_clusters": 3, "n_init": 10} | params.get("kmeans", {})
        kmeans_scaling = config["scaling"] != "off"
        kmeans_train = self.train_scaled if kmeans_scaling else train_model_x
        kmeans_test = self.test_scaled if kmeans_scaling else test_model_x
        kmeans_started = perf_counter()
        self.kmeans = KMeans(random_state=seed, **kmeans_params).fit(kmeans_train)
        kmeans_fit_ms = (perf_counter() - kmeans_started) * 1000
        self.training_evidence["kmeans"] = {
            "model": type(self.kmeans).__name__,
            "train_shape": kmeans_train.shape,
            "fit_ms": kmeans_fit_ms,
            "scaled": kmeans_scaling,
        }
        self.cluster_mapping = {
            cluster: Counter(self.train_y[self.kmeans.labels_ == cluster]).most_common(1)[0][0]
            for cluster in range(kmeans_params["n_clusters"])
        }
        self.kmeans_predictions = np.array([
            self.cluster_mapping[cluster]
            for cluster in self.kmeans.predict(kmeans_test)
        ])
        self.solution_predictions = {
            "rules": self.rule_predictions,
            **model_predictions,
            "kmeans": self.kmeans_predictions,
        }
        self.run_ms = (perf_counter() - run_started) * 1000

    @staticmethod
    def _predict_rules(
        features: np.ndarray,
        petal_length_threshold: float,
        petal_width_threshold: float,
    ) -> np.ndarray:
        return np.where(
            features[:, 2] < petal_length_threshold,
            0,
            np.where(features[:, 3] < petal_width_threshold, 1, 2),
        )

    def accuracy(self, solution_id: str) -> float:
        return accuracy_score(self.test_y, self.solution_predictions[solution_id])

    def metrics(self, solution_id: str) -> dict[str, float | int]:
        predictions = self.solution_predictions[solution_id]
        return {
            "accuracy": accuracy_score(self.test_y, predictions),
            "precision": precision_score(self.test_y, predictions, average="weighted"),
            "recall": recall_score(self.test_y, predictions, average="weighted"),
            "f1": f1_score(self.test_y, predictions, average="weighted"),
            "correct": int((predictions == self.test_y).sum()),
            "errors": int((predictions != self.test_y).sum()),
        }

    def save_best_model(self) -> tuple[Path, Path]:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        model_path = ARTIFACTS_DIR / "iris_best_model.pkl"
        scaler_path = ARTIFACTS_DIR / "iris_scaler.pkl"
        joblib.dump(self.models["svm"][0], model_path)
        joblib.dump(self.scaler, scaler_path)
        return model_path, scaler_path

    def load_and_predict(self, sample: list[float]) -> tuple[int, np.ndarray]:
        model_path, scaler_path = self.save_best_model()
        loaded_model = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)
        selected_sample = np.asarray(sample)[self.config["features"]]
        scaled_sample = loaded_scaler.transform([selected_sample])
        prediction = int(loaded_model.predict(scaled_sample)[0])
        probabilities = loaded_model.predict_proba(scaled_sample)[0]
        return prediction, probabilities


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(6, 4), dpi=100, facecolor="#fbfaf6")
        super().__init__(self.figure)
        self.setMinimumHeight(300)

    def show_step(
        self,
        step_id: str,
        workflow: IrisWorkflow,
        x_feature: int = 2,
        y_feature: int = 3,
        scaled: bool = False,
        solution_id: str = "logistic",
    ) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#fbfaf6")
        colors = ["#e07a5f", "#3d7ea6", "#6a994e"]

        if step_id == "explore":
            for index, name in enumerate(workflow.class_names):
                mask = workflow.labels == index
                axis.scatter(
                    workflow.features[mask, x_feature], workflow.features[mask, y_feature],
                    label=name, color=colors[index], alpha=0.78, s=42,
                )
            axis.set_title("选择不同维度，观察类别重叠")
            axis.set_xlabel(workflow.feature_names[x_feature])
            axis.set_ylabel(workflow.feature_names[y_feature])
            axis.legend(frameon=False)
        elif step_id == "experiment":
            predictions = workflow.solution_predictions[solution_id]
            for index, name in enumerate(workflow.class_names):
                mask = predictions == index
                axis.scatter(
                    workflow.test_x[mask, 2], workflow.test_x[mask, 3],
                    label=name, color=colors[index], alpha=0.75, s=36,
                )
            titles = {
                "rules": "手工阈值对测试样本的判断",
                "knn": "KNN 对测试样本的预测",
                "logistic": "逻辑回归对测试样本的预测",
                "decision-tree": "决策树对测试样本的预测",
                "svm": "SVM 对测试样本的预测",
                "random-forest": "随机森林对测试样本的预测",
                "kmeans": "K-Means 自然分组映射到类别后的结果",
            }
            axis.set_title(titles[solution_id])
            axis.set_xlabel(workflow.feature_names[2])
            axis.set_ylabel(workflow.feature_names[3])
            wrong = predictions != workflow.test_y
            axis.scatter(
                workflow.test_x[wrong, 2], workflow.test_x[wrong, 3],
                facecolors="none", edgecolors="#20262e", s=100, linewidths=1.6,
                label="预测错误",
            )
            axis.legend(frameon=False)
        elif step_id == "confusion":
            predictions = workflow.solution_predictions[solution_id]
            matrix = confusion_matrix(workflow.test_y, predictions)
            image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(10, matrix.max()))
            titles = {
                "knn": "KNN",
                "logistic": "逻辑回归",
                "decision-tree": "决策树",
                "svm": "SVM",
                "random-forest": "随机森林",
            }
            axis.set_title(f"{titles[solution_id]} · 混淆矩阵")
            axis.set_xlabel("预测类别")
            axis.set_ylabel("真实类别")
            axis.set_xticks(range(3), workflow.class_names)
            axis.set_yticks(range(3), workflow.class_names)
            for row in range(3):
                for column in range(3):
                    axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
            self.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        elif step_id in {"evaluate", "select"}:
            ids = [
                "rules", "knn", "logistic", "decision-tree",
                "svm", "random-forest", "kmeans",
            ]
            names = ["规则", "KNN", "逻辑回归", "决策树", "SVM", "随机森林", "K-Means"]
            values = [workflow.accuracy(item) for item in ids]
            bar_colors = ["#e07a5f", "#3d7ea6", "#5b8e7d", "#f2cc8f", "#7b6d8d", "#9c6644", "#6a994e"]
            if step_id == "select":
                bar_colors = ["#d4d1c8" if item != solution_id else "#e07a5f" for item in ids]
            bars = axis.bar(names, values, color=bar_colors, width=0.58)
            axis.set_ylim(0, 1.08)
            axis.set_ylabel("测试集准确率")
            axis.set_title("候选方案统一评估" if step_id == "evaluate" else "当前工程选择")
            axis.tick_params(axis="x", rotation=18)
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center")
        elif step_id == "deploy":
            predictions = workflow.solution_predictions[solution_id]
            matrix = confusion_matrix(workflow.test_y, predictions)
            image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(10, matrix.max()))
            axis.set_title(f"部署前确认 · {workflow.accuracy(solution_id):.1%}")
            axis.set_xlabel("预测类别")
            axis.set_ylabel("真实类别")
            axis.set_xticks(range(3), workflow.class_names)
            axis.set_yticks(range(3), workflow.class_names)
            for row in range(3):
                for column in range(3):
                    axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
            self.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        else:
            axis.axis("off")

        axis.grid(axis="y", alpha=0.18)
        self.figure.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"ML Flow Analyzer {APP_VERSION} · AI 与机器学习流程学习")
        self.resize(1420, 860)
        self.workflow = IrisWorkflow()
        self.content = self._load_content()
        self.catalog = self._load_catalog()
        self.model_parameter_values: dict[str, dict] = {}
        self.baseline_snapshots = {
            solution["id"]: self._capture_snapshot(solution["id"])
            for solution in self.content["solutions"]
        }
        self.current_snapshots = dict(self.baseline_snapshots)
        self.experiment_history: list[dict[str, str | float | int]] = []
        self._build_ui()
        self._apply_style()
        self._show_task_overview()

    def _load_content(self) -> dict:
        with KNOWLEDGE_PATH.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def _load_catalog(self) -> dict:
        with CATALOG_PATH.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("ML Flow Analyzer")
        title.setObjectName("title")
        version = QLabel(APP_VERSION)
        version.setObjectName("versionBadge")
        subtitle = QLabel("从一个具体任务出发，看懂机器学习如何从数据走到部署")
        subtitle.setObjectName("subtitle")
        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addWidget(version)
        title_row.addStretch()
        title_box.addLayout(title_row)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        return_button = QPushButton("返回学习中心")
        return_button.clicked.connect(self._return_to_practice)
        header.addWidget(return_button)
        task_label = QLabel("学习任务")
        task_label.setObjectName("taskLabel")
        header.addWidget(task_label)
        self.task_selector = QComboBox()
        self.task_selector.setObjectName("taskSelector")
        self.task_selector.addItem(self.content["task"]["title"], self.content["task"]["id"])
        self.task_selector.setMinimumWidth(230)
        self.task_selector.currentIndexChanged.connect(self._task_selected)
        header.addWidget(self.task_selector)
        ask_ai = QPushButton("问 AI")
        ask_ai.setToolTip("基于当前页面、任务和运行结果提问")
        ask_ai.clicked.connect(self._ask_ai)
        header.addWidget(ask_ai)
        outer.addLayout(header)

        intro = QFrame()
        intro.setObjectName("intro")
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(16, 13, 16, 13)
        intro_layout.addWidget(QLabel(self.content["task"]["objective"]))
        intro_layout.addWidget(QLabel(self.content["task"]["note"]))
        outer.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_step_panel())
        splitter.addWidget(self._build_visual_panel())
        splitter.addWidget(self._build_knowledge_panel())
        splitter.setSizes([220, 600, 430])
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)
        self.task_page = root
        self.page_stack = QStackedWidget()
        self.portal_page = self._build_portal_page()
        self.page_stack.addWidget(self.portal_page)
        self.page_stack.addWidget(self.task_page)
        self.setCentralWidget(self.page_stack)

    def _build_portal_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_row = QHBoxLayout()
        title = QLabel("ML Flow Analyzer")
        title.setObjectName("title")
        version = QLabel(APP_VERSION)
        version.setObjectName("versionBadge")
        title_row.addWidget(title)
        title_row.addWidget(version)
        title_row.addStretch()
        title_box.addLayout(title_row)
        subtitle = QLabel("AI 与机器学习流程可视化学习工具")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        header.addWidget(QLabel("学习路线"))
        self.route_selector = QComboBox()
        self.route_selector.setObjectName("taskSelector")
        for route in self.catalog["routes"]:
            self.route_selector.addItem(route["title"], route["id"])
        self.route_selector.setMinimumWidth(210)
        self.route_selector.currentIndexChanged.connect(self._route_changed)
        header.addWidget(self.route_selector)
        ask_ai = QPushButton("问 AI")
        ask_ai.setToolTip("基于当前路线和页面内容提问")
        ask_ai.clicked.connect(self._ask_ai)
        header.addWidget(ask_ai)
        outer.addLayout(header)

        intro = QFrame()
        intro.setObjectName("intro")
        intro_layout = QVBoxLayout(intro)
        self.route_title = QLabel()
        self.route_title.setObjectName("heading")
        self.route_subtitle = QLabel()
        self.route_subtitle.setWordWrap(True)
        intro_layout.addWidget(self.route_title)
        intro_layout.addWidget(self.route_subtitle)
        outer.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        route_panel = QFrame()
        route_panel.setObjectName("panel")
        route_layout = QVBoxLayout(route_panel)
        route_layout.addWidget(self._heading("主题目录"))
        self.route_items = QListWidget()
        self.route_items.currentRowChanged.connect(self._portal_item_selected)
        route_layout.addWidget(self.route_items)
        splitter.addWidget(route_panel)

        content_panel = QFrame()
        content_panel.setObjectName("panel")
        content_layout = QVBoxLayout(content_panel)
        content_header = QHBoxLayout()
        self.portal_content_title = self._heading("AI 全景")
        content_header.addWidget(self.portal_content_title)
        content_header.addStretch()
        self.portal_action = QPushButton("进入 Iris 实践任务")
        self.portal_action.clicked.connect(self._open_selected_task)
        self.portal_action.hide()
        content_header.addWidget(self.portal_action)
        content_layout.addLayout(content_header)
        self.portal_content = QTextBrowser()
        self.portal_content.setObjectName("portalContent")
        content_layout.addWidget(self.portal_content, 1)
        splitter.addWidget(content_panel)

        knowledge_panel = QFrame()
        knowledge_panel.setObjectName("panel")
        knowledge_layout = QVBoxLayout(knowledge_panel)
        knowledge_layout.addWidget(self._heading("范围与说明"))
        self.portal_overview_image = ResponsiveImageLabel()
        self.portal_overview_image.setObjectName("overviewImage")
        self.portal_overview_image.hide()
        knowledge_layout.addWidget(self.portal_overview_image)
        self.portal_knowledge = QTextBrowser()
        self.portal_knowledge.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        knowledge_layout.addWidget(self.portal_knowledge)
        splitter.addWidget(knowledge_panel)
        splitter.setSizes([260, 650, 410])
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        self._populate_route_items()
        return page

    def _current_route(self) -> dict:
        route_id = self.route_selector.currentData()
        return next(route for route in self.catalog["routes"] if route["id"] == route_id)

    def _route_changed(self, _index: int) -> None:
        self._populate_route_items()

    def _populate_route_items(self, selected_id: str | None = None) -> None:
        route = self._current_route()
        self.route_title.setText(route["title"])
        self.route_subtitle.setText(route["subtitle"])
        self.route_items.blockSignals(True)
        self.route_items.clear()
        selected_row = 0
        for row, item in enumerate(route["items"]):
            status = "  ·  规划中" if item["status"] != "available" else ""
            group = f"{item['group']}\n" if item.get("group") else ""
            list_item = QListWidgetItem(f"{group}{item['title']}{status}")
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            self.route_items.addItem(list_item)
            if item["id"] == (selected_id or route["default_item"]):
                selected_row = row
        self.route_items.blockSignals(False)
        self.route_items.setCurrentRow(selected_row)
        self._portal_item_selected(selected_row)

    def _portal_item_selected(self, row: int) -> None:
        if row < 0:
            return
        item = self.route_items.item(row).data(Qt.ItemDataRole.UserRole)
        route_id = self.route_selector.currentData()
        self.portal_overview_image.hide()
        task_titles = {
            "iris-classification": "进入 Iris 实践任务",
            "housing-regression": "进入 Housing 实践任务",
            "wine-clustering": "进入 Wine 聚类任务",
        }
        self.portal_action.setVisible(item["id"] in task_titles)
        self.portal_action.setText(task_titles.get(item["id"], "进入实践任务"))
        self.portal_action.setProperty("taskId", item["id"])
        self.portal_content_title.setText(item["title"])
        if route_id == "panorama" and item["id"] == "ml-example":
            self._show_ml_example(item)
        elif route_id == "panorama" and item["id"] == "genai-example":
            self._show_genai_example(item)
        elif route_id == "panorama" and item["id"] == "fusion-example":
            self._show_fusion_example(item)
        elif item["status"] != "available":
            self.portal_content.setHtml(
                f"<h2>{item['title']}</h2><p>{item['summary']}</p>"
                "<p><b>状态：规划中</b></p><p>当前版本尚未提供可运行内容。</p>"
            )
            self.portal_knowledge.setHtml(
                "<h2>为什么保留这个入口？</h2>"
                "<p>它用于展示完整学习路线和后续边界，但不会冒充已经实现的功能。</p>"
            )
        elif route_id == "overview":
            self._show_course_overview(item)
        elif route_id == "panorama":
            self._show_panorama(item)
        elif route_id == "foundations":
            self._show_foundations(item)
        else:
            self._show_practice_item(item)

    def _show_course_overview(self, item: dict) -> None:
        overview = self.catalog["course_overview"]
        route_rows = "".join(
            f"<tr><td><b>{route['title']}</b></td>"
            f"<td>{overview['route_notes'][route['id']]}</td>"
            f"<td>{self._route_status(route)}</td></tr>"
            for route in self.catalog["routes"]
            if route["id"] != "overview"
        )
        practice = next(route for route in self.catalog["routes"] if route["id"] == "practice")
        practice_rows = "".join(
            f"<tr><td>{entry.get('group', '1. Overview')}</td>"
            f"<td><b>{entry['title']}</b></td>"
            f"<td>{'可用' if entry['status'] == 'available' else '规划中'}</td></tr>"
            for entry in practice["items"]
        )
        self.portal_content.setHtml(
            f"<h2>{overview['title']}</h2><p>{overview['introduction']}</p>"
            "<h3>学习路线总览</h3>"
            "<table cellspacing='0' cellpadding='8' border='1' width='100%'>"
            f"<tr><th>路线</th><th>学习重点</th><th>状态</th></tr>{route_rows}</table>"
            "<h3>ML 实践课程目录</h3>"
            "<table cellspacing='0' cellpadding='8' border='1' width='100%'>"
            f"<tr><th>章节</th><th>内容</th><th>状态</th></tr>{practice_rows}</table>"
        )
        self.portal_knowledge.setHtml(
            "<h2>课程说明</h2>"
            f"<h3>适合谁</h3><p>{overview['audience']}</p>"
            f"<h3>怎么学习</h3><p>{overview['learning_method']}</p>"
            f"<h3>当前进度</h3><p>{overview['current_progress']}</p>"
            "<h3>建议顺序</h3>"
            "<p>AI 全景 → AI 入门 → ML 实践。已有基础的用户也可以直接进入 Iris 实践任务。</p>"
        )

    @staticmethod
    def _route_status(route: dict) -> str:
        available = sum(item["status"] == "available" for item in route["items"])
        total = len(route["items"])
        if available == total:
            return "可用"
        if available == 0:
            return "规划中"
        return f"部分可用（{available}/{total}）"

    def _show_panorama(self, item: dict) -> None:
        panorama = self.catalog["panorama"]
        source_path = (CATALOG_PATH.parent / panorama["source"]).resolve()
        source_text = source_path.read_text(encoding="utf-8")
        self.portal_content.setHtml(
            f"<h1>{panorama['title']}</h1><p>{panorama['summary']}</p>"
            "<pre style='font-family: Consolas, Microsoft YaHei UI, monospace; "
            "font-size: 13px; line-height: 1.45; white-space: pre-wrap; "
            f"background: #f7f8f6; padding: 14px;'>{escape(source_text)}</pre>"
        )
        image_path = (CATALOG_PATH.parent / panorama["overview_image"]).resolve()
        self.portal_overview_image.set_source(image_path)
        self.portal_overview_image.show()
        description = "".join(
            f"<p>{escape(paragraph)}</p>"
            for paragraph in panorama["overview_description"].split("\n\n")
        )
        self.portal_knowledge.setHtml(
            "<h2>AI、ML、DL 与 GenAI 的关系</h2>"
            f"{description}"
            f"<p style='background:#fff8e8; padding:8px;'><b>边界提示：</b>"
            f"{escape(panorama['overview_note'])}</p>"
        )

    def _show_genai_example(self, item: dict) -> None:
        examples = self.catalog["panorama"]["genai_examples"]
        rows = "".join(
            f"<tr><td><b>{escape(example['title'])}</b><br>{escape(example['model'])}</td>"
            f"<td>{escape(example['flow'])}</td><td>{escape(example['tools'])}</td>"
            f"<td>{escape(example['constraints'])}</td></tr>"
            for example in examples["items"]
        )
        notes = "".join(
            f"<li><b>{escape(example['title'])}：</b>{escape(example['role'])}</li>"
            for example in examples["items"]
        )
        self.portal_content.setHtml(
            f"<h2>{item['title']}</h2><p>{item['summary']}</p>"
            "<table cellspacing='0' cellpadding='7' border='1' width='100%'>"
            f"<tr><th>案例与模型</th><th>生成流程</th><th>常用工具</th><th>工程约束</th></tr>{rows}</table>"
        )
        self.portal_knowledge.setHtml(
            "<h2>在课程中的定位</h2>"
            f"<p>{escape(examples['intro'])}</p><ul>{notes}</ul>"
            "<h3>边界</h3><p>本地 LLM 与扩散模型的具体下载、量化、启动和设备测量，"
            "将在后续本地部署主题中实现。本页不提供运行入口。</p>"
        )

    def _show_ml_example(self, item: dict) -> None:
        example = self.catalog["panorama"]["ml_example"]
        model_rows = "".join(
            f"<li><b>{escape(model['name'])}：</b>{escape(model['detail'])}</li>"
            for model in example["models"]
        )
        knowledge_rows = "".join(
            f"<li>{escape(point)}</li>" for point in example["knowledge_points"]
        )
        self.portal_content.setHtml(
            f"<h2>{item['title']}</h2>"
            "<h3>问题类型</h3>"
            f"<p>{escape(example['problem'])}</p>"
            "<h3>项目流程</h3>"
            f"<p>{escape(example['workflow'])}</p>"
            "<h3>涉及的模型</h3>"
            f"<ul>{model_rows}</ul>"
            "<h3>评价指标覆盖的知识点</h3>"
            f"<p>{escape(example['knowledge_intro'])}</p>"
            f"<ul>{knowledge_rows}</ul>"
            f"<p>{escape(example['positioning'])}</p>"
        )
        image_path = (CATALOG_PATH.parent / example["knowledge"]["result_image"]).resolve()
        self.portal_overview_image.set_source(image_path)
        self.portal_overview_image.show()
        self.portal_knowledge.setHtml(
            "<h2>参考项目结果</h2>"
            f"<p><b>项目参考地址：</b>{escape(example['knowledge']['source'])}</p>"
            "<p>上图为参考项目给出的模型评价指标对比。</p>"
        )

    def _show_fusion_example(self, item: dict) -> None:
        example = self.catalog["panorama"]["fusion_example"]
        device_rows = "".join(f"<li>{escape(text)}</li>" for text in example["device_steps"])
        cloud_rows = "".join(f"<li>{escape(text)}</li>" for text in example["cloud_steps"])
        self.portal_content.setHtml(
            f"<h2>{item['title']}</h2><p>{item['summary']}</p>"
            f"<p>{escape(example['intro'])}</p>"
            "<table cellspacing='0' cellpadding='7' border='1' width='100%'>"
            "<tr><th>位置</th><th>负责内容</th><th>为什么放在这里</th></tr>"
            "<tr><td><b>端侧 ESP32-S3</b></td><td>音频前端、WakeNet 唤醒检测、Wi-Fi 上传与 PCM 播放</td>"
            "<td>持续运行，需要低延迟、低功耗，并避免持续上传原始音频。</td></tr>"
            "<tr><td><b>云端服务</b></td><td>ASR 语音转文本、LLM 理解与生成、TTS 合成音频</td>"
            "<td>模型较大，需要更强算力并可持续更新能力。</td></tr>"
            "</table>"
            "<h3>端侧 ML：持续感知与触发</h3>"
            f"<ul>{device_rows}</ul>"
            "<h3>云端 GenAI：理解与生成</h3>"
            f"<ul>{cloud_rows}</ul>"
        )
        image_path = (CATALOG_PATH.parent / example["image"]).resolve()
        self.portal_overview_image.set_source(image_path)
        self.portal_overview_image.show()
        self.portal_knowledge.setHtml(
            "<h2>项目来源</h2>"
            "<p>ESP-SR：https://github.com/espressif/esp-sr</p>"
            "<h3>ML + GenAI 如何联动</h3>"
            "<p>端侧轻量模型负责“现在是否需要响应”的分类判断；云端生成模型负责"
            "“用户说了什么、应如何回答”。前者强调实时性与资源受限，后者强调理解、"
            "知识与自然语言生成。</p>"
            "<h3>对应 ML 体系</h3>"
            f"<p>{escape(example['ml_system'])}</p>"
            "<h3>云端 GenAI 部分</h3>"
            f"<p>{escape(example['genai_system'])}</p>"
        )

    def _show_foundations(self, item: dict) -> None:
        foundations = self.catalog["foundations"]
        if item["id"] == "ml-problems":
            rows = "".join(
                f"<tr><td><b>{entry['name']}</b></td><td>{entry['question']}</td>"
                f"<td>{entry['example']}</td></tr>"
                for entry in foundations["problems"]
            )
            self.portal_content.setHtml(
                "<h2>机器学习问题类型</h2>"
                "<table cellspacing='0' cellpadding='8' border='1' width='100%'>"
                f"<tr><th>类型</th><th>核心问题</th><th>示例</th></tr>{rows}</table>"
            )
            self.portal_knowledge.setHtml(
                "<h2>先判断问题，再选择模型</h2>"
                "<p>分类和回归通常使用有标签数据；聚类常用于没有标签时寻找分组；降维用于压缩或观察高维信息。</p>"
            )
            return
        rows = "".join(
            f"<tr><td><b>{entry['short']}</b><br>{entry['name']}</td>"
            f"<td>{entry['meaning']}</td><td>{entry['typical']}</td>"
            f"<td>{entry['boundary']}</td></tr>"
            for entry in foundations["concepts"]
        )
        self.portal_content.setHtml(
            "<h2>ML、DL、TinyML 与 LLM</h2>"
            "<table cellspacing='0' cellpadding='8' border='1' width='100%'>"
            f"<tr><th>概念</th><th>是什么</th><th>常见场景</th><th>边界</th></tr>{rows}</table>"
        )
        self.portal_knowledge.setHtml(
            "<h2>关键边界</h2>"
            "<p>DL 属于 ML；LLM 是深度学习和生成式 AI 的重要分支；TinyML 主要描述在 MCU 等受限设备上的部署。</p>"
            "<p>TinyML 与本地 LLM 都强调本地运行，但它们面对的模型规模、内存和算力并不相同。</p>"
        )

    def _show_practice_item(self, item: dict) -> None:
        if item["id"] == "practice-overview":
            practice = next(route for route in self.catalog["routes"] if route["id"] == "practice")
            rows = "".join(
                f"<tr><td>{entry.get('group', '1. Overview')}</td><td><b>{entry['title']}</b></td>"
                f"<td>{'可用' if entry['status'] == 'available' else '规划中'}</td></tr>"
                for entry in practice["items"]
            )
            self.portal_content.setHtml(
                "<h2>ML 实践课程目录</h2><p>从传统机器学习任务开始，逐步进入深度学习、TinyML 和本地 LLM。</p>"
                "<table cellspacing='0' cellpadding='8' border='1' width='100%'>"
                f"<tr><th>章节</th><th>内容</th><th>状态</th></tr>{rows}</table>"
            )
            self.portal_knowledge.setHtml(
                "<h2>当前进度</h2><p><b>Iris 鸢尾花分类</b>已经完成，可从左侧目录进入。</p>"
                "<p>其他内容保留在课程结构中并明确标记为规划中。</p>"
            )
            return
        self.portal_content.setHtml(
            f"<h2>{item['title']}</h2><p>{item['summary']}</p>"
            "<p><b>状态：可用</b></p>"
            "<p>当前已实现任务定义、数据探索、多模型实验、参数重训、评估、模型保存与加载推理。</p>"
        )
        self.portal_knowledge.setHtml(
            "<h2>所在课程位置</h2>"
            f"<p>ML 实践 → {item.get('group', '分类模型应用')} → {item['title']}</p>"
            "<p>进入任务后可随时返回实践目录。</p>"
        )

    def _open_iris(self) -> None:
        self.setWindowTitle(f"ML Flow Analyzer {APP_VERSION} · Iris 分类学习")
        self.page_stack.setCurrentWidget(self.task_page)
        self._show_task_overview()

    def _ask_ai(self) -> None:
        AiDialog(self._ai_context, self).exec()

    def _ai_context(self) -> str:
        current_page = self.page_stack.currentWidget()
        if hasattr(self, "housing_page") and current_page is self.housing_page:
            return self.housing_page.ai_context()
        if hasattr(self, "wine_page") and current_page is self.wine_page:
            return self.wine_page.ai_context()
        if current_page is self.task_page:
            step_row = self.step_list.currentRow()
            step = self.content["steps"][step_row] if step_row >= 0 else None
            sections = [
                "[ML Flow Analyzer 当前任务]",
                f"任务: {self.content['task']['title']}",
                f"当前页面: {step['title'] if step else '任务概览'}",
                f"任务说明: {self.content['task']['subtitle']}",
                f"数据概况: {self.content['task']['dataset_summary']}",
            ]
            if step is not None:
                sections.extend([f"学习目标: {step['goal']}", f"页面说明: {step['explanation']}"])
            solution_id = self.solution_selector.currentData()
            if solution_id and self.workflow.solution_predictions:
                metrics = self.workflow.metrics(solution_id)
                sections.append(
                    f"当前方案: {solution_id}; 参数: {self.model_parameter_values.get(solution_id, {})}; "
                    f"Accuracy={metrics['accuracy']:.4f}; F1={metrics['f1']:.4f}; "
                    f"错误数={metrics['errors']}"
                )
            return "\n".join(sections)
        route = self._current_route()
        current_item = self.route_items.currentItem()
        item = current_item.data(Qt.ItemDataRole.UserRole) if current_item else {}
        return "\n".join([
            "[ML Flow Analyzer 当前课程页面]",
            f"当前路线: {route['title']}",
            f"路线说明: {route['subtitle']}",
            f"当前内容: {item.get('title', '无')}",
            f"内容摘要: {item.get('summary', '无')}",
            f"内容状态: {item.get('status', '无')}",
            f"页面正文: {self.portal_content.toPlainText()}",
            f"知识说明: {self.portal_knowledge.toPlainText()}",
        ])

    def _open_selected_task(self) -> None:
        task_id = self.portal_action.property("taskId")
        if task_id == "iris-classification":
            self._open_iris()
        elif task_id == "housing-regression":
            self._open_housing()
        elif task_id == "wine-clustering":
            self._open_wine()

    def _open_housing(self) -> None:
        if not hasattr(self, "housing_page"):
            self.housing_page = HousingTaskPage(
                lambda: self._return_to_practice("housing-regression"),
                self._ask_ai,
            )
            self.page_stack.addWidget(self.housing_page)
        self.setWindowTitle(f"ML Flow Analyzer {APP_VERSION} · California Housing 回归")
        self.page_stack.setCurrentWidget(self.housing_page)

    def _open_wine(self) -> None:
        if not hasattr(self, "wine_page"):
            self.wine_page = WineTaskPage(
                lambda: self._return_to_practice("wine-clustering"),
                self._ask_ai,
            )
            self.page_stack.addWidget(self.wine_page)
        self.setWindowTitle(f"ML Flow Analyzer {APP_VERSION} · Wine 无监督聚类")
        self.page_stack.setCurrentWidget(self.wine_page)

    def _return_to_practice(self, selected_id: str = "iris-classification") -> None:
        practice_index = self.route_selector.findData("practice")
        self.route_selector.setCurrentIndex(practice_index)
        self._populate_route_items(selected_id)
        self.page_stack.setCurrentWidget(self.portal_page)
        self.setWindowTitle(f"ML Flow Analyzer {APP_VERSION} · AI 与机器学习流程学习")

    def _build_step_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.addWidget(self._heading("任务介绍"))
        self.overview_button = QPushButton("任务概览\n目标、来源与原始数据")
        self.overview_button.setObjectName("overviewButton")
        self.overview_button.clicked.connect(self._show_task_overview)
        layout.addWidget(self.overview_button)
        layout.addSpacing(8)
        layout.addWidget(self._heading("学习流程"))
        self.progress = QLabel("第 1 / 7 步")
        self.progress.setObjectName("progress")
        layout.addWidget(self.progress)
        self.step_list = QListWidget()
        for step in self.content["steps"]:
            item = QListWidgetItem(f"{step['title']}\n{step['short']}")
            item.setData(Qt.ItemDataRole.UserRole, step["id"])
            self.step_list.addItem(item)
        self.step_list.currentRowChanged.connect(self._select_step)
        layout.addWidget(self.step_list, 1)
        self.back_button = QPushButton("上一步")
        self.next_button = QPushButton("下一步")
        self.back_button.clicked.connect(lambda: self._select_step(self.step_list.currentRow() - 1))
        self.next_button.clicked.connect(lambda: self._select_step(self.step_list.currentRow() + 1))
        buttons = QHBoxLayout()
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.next_button)
        layout.addLayout(buttons)
        return panel

    def _build_visual_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        self.visual_scroll = QScrollArea()
        self.visual_scroll.setWidgetResizable(True)
        self.visual_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.visual_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.visual_content = QWidget()
        layout = QVBoxLayout(self.visual_content)
        layout.setContentsMargins(14, 14, 14, 14)
        self.visual_scroll.setWidget(self.visual_content)
        panel_layout.addWidget(self.visual_scroll)
        self.visual_title = self._heading("当前步骤")
        layout.addWidget(self.visual_title)
        self.visual_summary = QLabel()
        self.visual_summary.setObjectName("summary")
        self.visual_summary.setWordWrap(True)
        layout.addWidget(self.visual_summary)

        self.visual_controls = QFrame()
        self.visual_controls.setObjectName("controls")
        controls = QHBoxLayout(self.visual_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        self.x_label = QLabel("X 轴")
        self.x_feature = QComboBox()
        self.y_label = QLabel("Y 轴")
        self.y_feature = QComboBox()
        for index, name in enumerate(self.workflow.feature_names):
            self.x_feature.addItem(name, index)
            self.y_feature.addItem(name, index)
        self.x_feature.setCurrentIndex(2)
        self.y_feature.setCurrentIndex(3)
        self.x_feature.currentIndexChanged.connect(self._exploration_changed)
        self.y_feature.currentIndexChanged.connect(self._exploration_changed)
        self.solution_label = QLabel("候选方案")
        self.solution_selector = QComboBox()
        experiment_solutions = [
            solution for solution in self.content["solutions"]
            if solution.get("experiment", False)
        ]
        for solution in experiment_solutions:
            self.solution_selector.addItem(solution["title"], solution["id"])
        logistic_index = self.solution_selector.findData("logistic")
        self.solution_selector.setCurrentIndex(max(0, logistic_index))
        self.solution_selector.currentIndexChanged.connect(self._solution_changed)
        controls.addWidget(self.x_label)
        controls.addWidget(self.x_feature, 1)
        controls.addWidget(self.y_label)
        controls.addWidget(self.y_feature, 1)
        controls.addWidget(self.solution_label)
        controls.addWidget(self.solution_selector, 1)
        layout.addWidget(self.visual_controls)

        self.experiment_parameters = self._build_experiment_parameters()
        layout.addWidget(self.experiment_parameters)

        self.visual_stack = QStackedWidget()
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.data_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.visual_stack.addWidget(self.data_table)
        self.canvas = PlotCanvas()
        self.visual_stack.addWidget(self.canvas)
        self.evaluation_view = self._build_evaluation_view()
        self.visual_stack.addWidget(self.evaluation_view)
        layout.addWidget(self.visual_stack, 1)
        self.runtime = QLabel()
        self.runtime.setObjectName("runtime")
        self.runtime.setWordWrap(True)
        layout.addWidget(self.runtime)
        return panel

    def _build_experiment_parameters(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("experimentParameters")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)

        flow_row = QHBoxLayout()
        self.flow_parameter_widget = QWidget()
        flow_parameter_layout = QHBoxLayout(self.flow_parameter_widget)
        flow_parameter_layout.setContentsMargins(0, 0, 0, 0)
        flow_parameter_layout.addWidget(QLabel("流程参数"))
        self.test_size_input = QDoubleSpinBox()
        self.test_size_input.setRange(0.1, 0.5)
        self.test_size_input.setSingleStep(0.05)
        self.test_size_input.setValue(0.2)
        self.test_size_input.setPrefix("测试比例 ")
        flow_parameter_layout.addWidget(self.test_size_input)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 9999)
        self.seed_input.setValue(42)
        self.seed_input.setPrefix("随机种子 ")
        flow_parameter_layout.addWidget(self.seed_input)
        self.scaling_input = QComboBox()
        self.scaling_input.addItem("缩放：自动", "auto")
        self.scaling_input.addItem("缩放：强制开启", "on")
        self.scaling_input.addItem("缩放：关闭", "off")
        flow_parameter_layout.addWidget(self.scaling_input)
        self.stratify_input = QCheckBox("分层划分")
        self.stratify_input.setChecked(True)
        flow_parameter_layout.addWidget(self.stratify_input)
        flow_row.addWidget(self.flow_parameter_widget)
        layout.addLayout(flow_row)

        self.feature_parameter_widget = QWidget()
        feature_row = QHBoxLayout(self.feature_parameter_widget)
        feature_row.setContentsMargins(0, 0, 0, 0)
        feature_row.addWidget(QLabel("使用特征"))
        self.feature_inputs = []
        feature_titles = ["花萼长", "花萼宽", "花瓣长", "花瓣宽"]
        for title in feature_titles:
            checkbox = QCheckBox(title)
            checkbox.setChecked(True)
            self.feature_inputs.append(checkbox)
            feature_row.addWidget(checkbox)
        feature_row.addStretch()
        layout.addWidget(self.feature_parameter_widget)

        self.model_parameter_row = QHBoxLayout()
        layout.addLayout(self.model_parameter_row)
        self.run_experiment_button = QPushButton("运行实验")
        self.run_experiment_button.clicked.connect(self._run_experiment)
        self.model_parameter_row.addWidget(self.run_experiment_button)
        self._build_model_parameter_controls()

        self.snapshot_table = QTableWidget()
        self.snapshot_table.setObjectName("snapshotTable")
        self.snapshot_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.snapshot_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.snapshot_table.verticalHeader().setVisible(False)
        self.snapshot_table.setWordWrap(False)
        self.snapshot_table.verticalHeader().setDefaultSectionSize(24)
        self.snapshot_table.horizontalHeader().setFixedHeight(30)
        self.snapshot_table.setMinimumHeight(204)
        self.snapshot_table.setMaximumHeight(204)
        self.snapshot_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        snapshot_title = QLabel("实验快照对比")
        snapshot_title.setObjectName("snapshotTitle")
        layout.addWidget(snapshot_title)
        layout.addWidget(self.snapshot_table)
        self._show_snapshot_comparison()
        history_title = QLabel("本次会话实验记录（最多保留 8 次）")
        history_title.setObjectName("snapshotTitle")
        layout.addWidget(history_title)
        self.history_table = QTableWidget()
        self.history_table.setObjectName("snapshotTable")
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setWordWrap(False)
        self.history_table.verticalHeader().setDefaultSectionSize(24)
        self.history_table.horizontalHeader().setFixedHeight(30)
        self.history_table.setMinimumHeight(150)
        self.history_table.setMaximumHeight(150)
        layout.addWidget(self.history_table)
        self._show_experiment_history()
        return panel

    def _clear_model_parameter_controls(self) -> None:
        while self.model_parameter_row.count():
            item = self.model_parameter_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_model_parameter_controls(self) -> None:
        if not hasattr(self, "model_parameter_row"):
            return
        self._clear_model_parameter_controls()
        solution_id = self.solution_selector.currentData()
        solution = next(item for item in self.content["solutions"] if item["id"] == solution_id)
        self.model_parameter_row.addWidget(QLabel(f"{solution['title']} 参数"))
        self.model_parameter_widgets: dict[str, QWidget] = {}
        saved = self.model_parameter_values.get(solution_id, {})
        for spec in solution["adjustable_parameters"]:
            if spec["type"] in {"int", "optional_int"}:
                widget = QSpinBox()
                widget.setRange(spec["min"], spec["max"])
                widget.setSingleStep(spec["step"])
                value = saved.get(spec["id"], spec["default"])
                widget.setValue(0 if value is None else value)
            elif spec["type"] == "float":
                widget = QDoubleSpinBox()
                widget.setRange(spec["min"], spec["max"])
                widget.setSingleStep(spec["step"])
                widget.setDecimals(2)
                widget.setValue(saved.get(spec["id"], spec["default"]))
            else:
                widget = QComboBox()
                for choice in spec["choices"]:
                    widget.addItem(choice, choice)
                value = saved.get(spec["id"], spec["default"])
                widget.setCurrentIndex(widget.findData(value))
            widget.setToolTip(f"默认值：{spec['default']}。{spec['description']}")
            self.model_parameter_widgets[spec["id"]] = widget
            self.model_parameter_row.addWidget(QLabel(spec["title"]))
            self.model_parameter_row.addWidget(widget)
        self.model_parameter_row.addStretch()
        reset_button = QPushButton("恢复默认")
        reset_button.clicked.connect(self._reset_experiment)
        self.model_parameter_row.addWidget(reset_button)
        run_button = QPushButton("运行实验")
        run_button.clicked.connect(self._run_experiment)
        self.model_parameter_row.addWidget(run_button)
        is_rules = solution_id == "rules"
        self.flow_parameter_widget.setVisible(not is_rules)
        self.feature_parameter_widget.setVisible(not is_rules)

    def _build_evaluation_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        self.evaluation_table = QTableWidget()
        self.evaluation_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.evaluation_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.evaluation_table.setAlternatingRowColors(True)
        layout.addWidget(self.evaluation_table, 2)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("查看混淆矩阵"))
        self.evaluation_selector = QComboBox()
        for solution_id in CLASSIFIER_IDS:
            solution = next(item for item in self.content["solutions"] if item["id"] == solution_id)
            self.evaluation_selector.addItem(solution["title"], solution_id)
        self.evaluation_selector.currentIndexChanged.connect(self._evaluation_changed)
        selector_row.addWidget(self.evaluation_selector, 1)
        layout.addLayout(selector_row)

        self.evaluation_canvas = PlotCanvas()
        self.evaluation_canvas.setMinimumHeight(250)
        layout.addWidget(self.evaluation_canvas, 3)
        return view

    def _task_selected(self, _index: int) -> None:
        self._show_task_overview()

    def _show_task_overview(self) -> None:
        self.step_list.clearSelection()
        self.step_list.setCurrentRow(-1)
        task = self.content["task"]
        self.progress.setText("开始学习前")
        self.visual_title.setText(task["title"])
        self.visual_summary.setText(task["subtitle"])
        self._show_data_table()
        self.visual_stack.setCurrentWidget(self.data_table)
        self.visual_controls.hide()
        self.experiment_parameters.hide()
        self.runtime.setText(f"数据来源 · {task['source']}\n数据概况 · {task['dataset_summary']}")
        self.knowledge.setHtml(
            f"<h2>这个任务要做什么？</h2><p>{task['introduction']}</p>"
            f"{self._data_source_html(task)}"
            f"<p><b>数据概况：</b>{task['dataset_summary']}</p>"
            f"{self._dataset_tables_html(task)}"
            f"<p><b>方案说明：</b>{task['note']}</p>"
        )
        self.overview_button.setProperty("active", True)
        self.overview_button.style().unpolish(self.overview_button)
        self.overview_button.style().polish(self.overview_button)
        self.back_button.setEnabled(False)
        self.next_button.setEnabled(True)
        self.visual_scroll.verticalScrollBar().setValue(0)

    def _show_data_table(self, row_count: int | None = None) -> None:
        headers = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度", "类别"]
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)
        for column in range(len(headers)):
            self.data_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch
            )
        self.data_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed
        )
        visible_rows = len(self.workflow.features) if row_count is None else min(row_count, len(self.workflow.features))
        self.data_table.setRowCount(visible_rows)
        for row in range(self.data_table.rowCount()):
            values = [f"{value:.1f}" for value in self.workflow.features[row]]
            values.append(str(self.workflow.class_names[self.workflow.labels[row]]))
            for column, value in enumerate(values):
                self.data_table.setItem(row, column, QTableWidgetItem(value))

    def _exploration_changed(self, _index: int) -> None:
        row = self.step_list.currentRow()
        if row >= 0 and self.content["steps"][row]["id"] == "explore":
            self.canvas.show_step(
                "explore",
                self.workflow,
                self.x_feature.currentData(),
                self.y_feature.currentData(),
            )
            x_name = self.workflow.feature_names[self.x_feature.currentData()]
            y_name = self.workflow.feature_names[self.y_feature.currentData()]
            self.runtime.setText(
                f"交互观察 · 当前比较 {x_name} 与 {y_name}。"
                "重叠越多，仅凭这两个维度越难分类。"
            )

    def _solution_changed(self, _index: int) -> None:
        self._build_model_parameter_controls()
        self._show_snapshot_comparison()
        row = self.step_list.currentRow()
        if row < 0:
            return
        step = self.content["steps"][row]
        if step["id"] in {"experiment", "select", "deploy"}:
            self._render_step(step)
            self._show_step_knowledge(step)

    @staticmethod
    def _parameter_widget_value(widget: QWidget):
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        raise TypeError(f"Unsupported parameter widget: {type(widget)!r}")

    def _run_experiment(self) -> None:
        selected_features = [
            index for index, checkbox in enumerate(self.feature_inputs)
            if checkbox.isChecked()
        ]
        if not selected_features:
            QMessageBox.warning(self, "无法运行实验", "请至少选择一个输入特征。")
            return
        solution_id = self.solution_selector.currentData()
        solution = next(item for item in self.content["solutions"] if item["id"] == solution_id)
        values = {
            spec["id"]: self._parameter_widget_value(self.model_parameter_widgets[spec["id"]])
            for spec in solution["adjustable_parameters"]
        }
        for spec in solution["adjustable_parameters"]:
            if spec["type"] == "optional_int" and values[spec["id"]] == 0:
                values[spec["id"]] = None
        self.model_parameter_values[solution_id] = values
        config = {
            "test_size": self.test_size_input.value(),
            "random_state": self.seed_input.value(),
            "scaling": self.scaling_input.currentData(),
            "stratify": self.stratify_input.isChecked(),
            "features": selected_features,
            "model_params": self.model_parameter_values,
        }
        try:
            self.workflow.rerun(config)
        except ValueError as error:
            QMessageBox.warning(self, "实验参数无效", str(error))
            return
        self.current_snapshots = {
            item["id"]: self._capture_snapshot(item["id"])
            for item in self.content["solutions"]
        }
        self._record_experiment(solution_id)
        self._show_snapshot_comparison()
        row = self.step_list.currentRow()
        if row >= 0:
            step = self.content["steps"][row]
            self._render_step(step)
            self._show_step_knowledge(step)

    def _reset_experiment(self) -> None:
        self.test_size_input.setValue(0.2)
        self.seed_input.setValue(42)
        self.scaling_input.setCurrentIndex(self.scaling_input.findData("auto"))
        self.stratify_input.setChecked(True)
        for checkbox in self.feature_inputs:
            checkbox.setChecked(True)
        self.model_parameter_values.clear()
        self._build_model_parameter_controls()
        self.workflow.rerun({
            "test_size": 0.2,
            "random_state": 42,
            "scaling": "auto",
            "stratify": True,
            "features": [0, 1, 2, 3],
            "model_params": {},
        })
        self.current_snapshots = dict(self.baseline_snapshots)
        self.experiment_history.clear()
        self._show_snapshot_comparison()
        self._show_experiment_history()
        row = self.step_list.currentRow()
        if row >= 0:
            step = self.content["steps"][row]
            self._render_step(step)
            self._show_step_knowledge(step)

    def _capture_snapshot(self, solution_id: str) -> dict[str, str | float | int]:
        solution = next(item for item in self.content["solutions"] if item["id"] == solution_id)
        configured = self.model_parameter_values.get(solution_id, {})
        parameters = ", ".join(
            f"{spec['title']}={configured.get(spec['id'], spec['default'])}"
            for spec in solution["adjustable_parameters"]
        )
        features = ", ".join(
            str(self.workflow.feature_names[index]).replace(" (cm)", "")
            for index in self.workflow.config["features"]
        )
        metrics = self.workflow.metrics(solution_id)
        if solution_id == "rules":
            fit_time = "无训练"
        else:
            fit_time = f"{self.workflow.training_evidence[solution_id]['fit_ms']:.2f} ms"
        return {
            "model": solution["title"],
            "parameters": parameters,
            "split": (
                f"测试 {self.workflow.config['test_size']:.0%} / seed {self.workflow.config['random_state']} / "
                f"{'分层' if self.workflow.config['stratify'] else '不分层'} / {self.workflow.config['scaling']}"
            ),
            "features": features,
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
            "errors": int(metrics["errors"]),
            "fit_time": fit_time,
        }

    def _record_experiment(self, solution_id: str) -> None:
        snapshot = dict(self.current_snapshots[solution_id])
        snapshot["run"] = self.workflow.run_count
        self.experiment_history.append(snapshot)
        self.experiment_history = self.experiment_history[-8:]
        self._show_experiment_history()

    def _show_experiment_history(self) -> None:
        if not hasattr(self, "history_table"):
            return
        headers = ["运行", "方案", "特征", "模型参数", "Accuracy", "F1", "错误数"]
        self.history_table.setColumnCount(len(headers))
        self.history_table.setHorizontalHeaderLabels(headers)
        self.history_table.setRowCount(len(self.experiment_history))
        for row, record in enumerate(reversed(self.experiment_history)):
            values = [
                str(record["run"]),
                str(record["model"]),
                str(record["features"]),
                str(record["parameters"]),
                f"{record['accuracy']:.2%}",
                f"{record['f1']:.4f}",
                str(record["errors"]),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.history_table.setItem(row, column, item)
        header = self.history_table.horizontalHeader()
        for column in (0, 1, 4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for column in (2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

    def _show_snapshot_comparison(self) -> None:
        if not hasattr(self, "snapshot_table"):
            return
        solution_id = self.solution_selector.currentData()
        baseline = self.baseline_snapshots[solution_id]
        current = self.current_snapshots[solution_id]
        rows = [
            ("模型", baseline["model"], current["model"]),
            ("模型参数", baseline["parameters"], current["parameters"]),
            ("流程配置", baseline["split"], current["split"]),
            ("使用特征", baseline["features"], current["features"]),
            ("Accuracy", f"{baseline['accuracy']:.2%}", f"{current['accuracy']:.2%}"),
            ("F1", f"{baseline['f1']:.4f}", f"{current['f1']:.4f}"),
            ("错误数", str(baseline["errors"]), str(current["errors"])),
            ("fit 耗时", baseline["fit_time"], current["fit_time"]),
        ]
        self.snapshot_table.setColumnCount(4)
        self.snapshot_table.setHorizontalHeaderLabels(["对比项", "默认基线", "当前实验", "变化"])
        self.snapshot_table.setRowCount(len(rows))
        for row, (label, before, after) in enumerate(rows):
            change = "未变化" if before == after else "已变化"
            if label == "Accuracy":
                delta = current["accuracy"] - baseline["accuracy"]
                change = f"{delta:+.2%}"
            elif label == "F1":
                delta = current["f1"] - baseline["f1"]
                change = f"{delta:+.4f}"
            elif label == "错误数":
                delta = current["errors"] - baseline["errors"]
                change = f"{delta:+d}"
            for column, value in enumerate((label, before, after, change)):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.snapshot_table.setItem(row, column, item)
        header = self.snapshot_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

    def _evaluation_changed(self, _index: int) -> None:
        row = self.step_list.currentRow()
        if row < 0 or self.content["steps"][row]["id"] != "evaluate":
            return
        self._show_evaluation_view()
        self._show_step_knowledge(self.content["steps"][row])

    def _build_knowledge_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.addWidget(self._heading("知识讲解"))
        self.knowledge = QTextBrowser()
        self.knowledge.setOpenExternalLinks(False)
        layout.addWidget(self.knowledge, 1)
        return panel

    def _heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("heading")
        return label

    def _select_step(self, row: int) -> None:
        if not self.content["steps"]:
            return
        if row < 0:
            return
        row = min(row, len(self.content["steps"]) - 1)
        if self.step_list.currentRow() != row:
            self.step_list.setCurrentRow(row)
            return
        step = self.content["steps"][row]
        self.overview_button.setProperty("active", False)
        self.overview_button.style().unpolish(self.overview_button)
        self.overview_button.style().polish(self.overview_button)
        self.progress.setText(f"第 {row + 1} / {len(self.content['steps'])} 步")
        self.visual_title.setText(step["title"])
        self.visual_summary.setText(step["goal"])
        self._render_step(step)
        self._show_step_knowledge(step)
        self.back_button.setEnabled(row > 0)
        self.next_button.setEnabled(row < len(self.content["steps"]) - 1)
        self.visual_scroll.verticalScrollBar().setValue(0)

    def _show_step_knowledge(self, step: dict) -> None:
        if step["id"] == "data":
            task = self.content["task"]
            self.knowledge.setHtml(
                f"<h2>1. 当前的任务</h2><p>{task['current_task']}</p>"
                f"<h2>2. 当前的数据以及特征</h2><p>{task['current_data']}</p>"
                f"{self._dataset_tables_html(task)}"
                f"{self._data_source_html(task)}"
                f"<h2>3. 引出的问题</h2><p>{task['question']}</p>"
            )
            return
        if step["id"] == "solutions":
            self.knowledge.setHtml(
                f"<h2>{step['title']}：{step['short']}</h2>"
                f"<p><b>这一步要做什么：</b>{step['goal']}</p>"
                f"<p><b>为什么来到这一步：</b>{step['explanation']}</p>"
                f"{self._knowledge_groups_html(step)}"
            )
            return
        if step["id"] == "experiment":
            self.knowledge.setHtml(self._experiment_knowledge_html(step))
            return
        if step["id"] == "evaluate":
            self.knowledge.setHtml(self._evaluation_knowledge_html(step))
            return
        if step["id"] == "select":
            self.knowledge.setHtml(self._selection_knowledge_html(step))
            return
        if step["id"] == "deploy":
            self.knowledge.setHtml(self._deployment_knowledge_html(step))
            return
        concept_titles = self.content.get("concept_titles", {})
        concept_html = "".join(
            f"<h3>{concept_titles.get(key, key)}</h3>"
            f"<p>{self.content['concepts'].get(key, '暂无说明。')}</p>"
            for key in step["concepts"]
        )
        self.knowledge.setHtml(
            f"<h2>{step['title']}：{step['short']}</h2>"
            f"<p><b>这一步要做什么：</b>{step['goal']}</p>"
            f"<p><b>当前任务中发生了什么：</b>{step['explanation']}</p>"
            f"{self._selected_solution_html(step['id'])}"
            f"{self._algorithm_guidance_html(step)}"
            f"{concept_html}"
        )

    def _experiment_knowledge_html(self, step: dict) -> str:
        solution_id = self.solution_selector.currentData()
        solution = next(item for item in self.content["solutions"] if item["id"] == solution_id)
        predictions = self.workflow.solution_predictions[solution_id]
        correct = int((predictions == self.workflow.test_y).sum())
        total = len(self.workflow.test_y)
        accuracy = self.workflow.accuracy(solution_id)
        preprocessing_code = escape(solution["preprocessing_code"])
        training_code = escape(solution["training_code"])
        feature_names = [
            self.workflow.feature_names[index]
            for index in self.workflow.config["features"]
        ]
        current_parameters = self.model_parameter_values.get(solution_id, {})
        parameter_rows = "".join(
            f"<tr><td>{spec['title']}</td><td>{spec['default']}</td>"
            f"<td>{current_parameters.get(spec['id'], spec['default'])}</td>"
            f"<td>{spec['description']}</td></tr>"
            for spec in solution["adjustable_parameters"]
        )
        train_counts = Counter(self.workflow.train_y)
        test_counts = Counter(self.workflow.test_y)
        if solution_id == "rules":
            execution_evidence = (
                f"第 {self.workflow.run_count} 次运行：手工规则不训练模型；"
                f"程序使用当前两个阈值重新判断 {total} 个测试样本。"
            )
        else:
            evidence = self.workflow.training_evidence[solution_id]
            execution_evidence = (
                f"第 {self.workflow.run_count} 次运行：已重新创建 {evidence['model']}，"
                f"使用输入形状 {evidence['train_shape']} 调用 fit()；"
                f"缩放 {'开启' if evidence['scaled'] else '关闭'}；"
                f"该模型 fit() 实测耗时 {evidence['fit_ms']:.2f} ms。"
            )
        return (
            f"<h2>方案实验：{solution['title']}</h2>"
            f"<p style='background:#e8f2ed; padding:8px;'><b>本次执行证据：</b>"
            f"{execution_evidence}</p>"
            "<h3>1. 方案实现介绍</h3>"
            f"<p>{step['explanation']}</p>"
            f"<p><b>当前流程参数：</b>测试比例 {self.workflow.config['test_size']:.0%}，"
            f"随机种子 {self.workflow.config['random_state']}，缩放策略 {self.workflow.config['scaling']}，"
            f"分层划分 {'开启' if self.workflow.config['stratify'] else '关闭'}。</p>"
            f"<p>本次训练集 <b>{len(self.workflow.train_y)}</b> 个，类别数量 {dict(train_counts)}；"
            f"测试集 <b>{len(self.workflow.test_y)}</b> 个，类别数量 {dict(test_counts)}。"
            "测试集不会参与训练。</p>"
            f"<p><b>当前使用特征：</b>{', '.join(feature_names)}</p>"
            "<h3>当前模型有哪些可调参数</h3>"
            "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
            "<tr><th>参数</th><th>默认值</th><th>当前值</th><th>作用</th></tr>"
            f"{parameter_rows}</table>"
            f"<h3>2. 当前方案如何处理数据</h3><p>{solution['process']}</p>"
            f"<p><b>是否缩放：</b>{solution['scaling']}</p>"
            f"<p><b>为什么：</b>{solution['scaling_reason']}</p>"
            f"<pre style='background:#f1f3f2; padding:10px;'>{preprocessing_code}</pre>"
            f"<h3>3. 模型、参数与训练代码</h3>"
            f"<p><b>实现：</b>{solution['model']}</p>"
            f"<p><b>关键参数：</b>{solution['parameters']}</p>"
            f"<pre style='background:#f1f3f2; padding:10px;'>{training_code}</pre>"
            f"<h3>4. 使用测试样本检查训练结果</h3>"
            f"<p>方案完成训练或规则设计后，使用此前保留且未参与训练的 {total} 个测试样本逐一预测，"
            "再把预测类别与真实类别比较。"
            f"本次预测正确 <b>{correct}</b> 个，预测错误 "
            f"<b>{total - correct}</b> 个，因此准确率为 "
            f"<b>{correct} / {total} = {accuracy:.1%}</b>。</p>"
            f"<p style='background:#edf5f2; padding:8px;'><b>左侧图说明：</b>"
            f"{solution['chart_note']}</p>"
        )

    def _evaluation_knowledge_html(self, step: dict) -> str:
        selected_id = self.evaluation_selector.currentData()
        selected = next(item for item in self.content["solutions"] if item["id"] == selected_id)
        matrix = confusion_matrix(
            self.workflow.test_y,
            self.workflow.solution_predictions[selected_id],
        )
        matrix_summary = (
            f"Setosa 正确 {matrix[0, 0]}/10；"
            f"Versicolor 被误判为 Virginica {matrix[1, 2]} 个；"
            f"Virginica 被误判为 Versicolor {matrix[2, 1]} 个。"
        )
        observations = "".join(
            f"<li style='margin-bottom:8px;'>{item}</li>"
            for item in step["observations"]
        )
        return (
            f"<h2>模型评估：五个分类模型</h2>"
            f"<p>{step['explanation']}</p>"
            "<h3>指标表怎么看</h3>"
            "<p><b>Accuracy</b> 是总体正确比例；<b>Precision</b> 关注预测为某类时有多少是真的；"
            "<b>Recall</b> 关注真实某类有多少被找到；<b>F1</b> 综合 Precision 与 Recall。"
            "当前表格使用 weighted 平均，使每个类别按测试样本数参与汇总。</p>"
            f"<h3>混淆矩阵：{selected['title']}</h3>"
            f"<p>{step['matrix_note']}</p>"
            f"<p><b>当前结果：</b>{matrix_summary}</p>"
            "<h3>关键观察</h3>"
            f"<ol>{observations}</ol>"
        )

    def _selection_knowledge_html(self, step: dict) -> str:
        model_path, scaler_path = self.workflow.save_best_model()
        code = escape(step["save_code"])
        return (
            "<h2>最终选择：SVM</h2>"
            f"<p>{step['explanation']}</p>"
            f"<p><b>选择依据：</b>{step['selection_reason']}</p>"
            f"<p><b>为什么保存两个文件：</b>{step['artifact_note']}</p>"
            "<h3>保存模型与 Scaler</h3>"
            f"<pre style='background:#f1f3f2; padding:10px;'>{code}</pre>"
            f"<p><b>实际生成：</b>{model_path.name}、{scaler_path.name}</p>"
        )

    def _deployment_knowledge_html(self, step: dict) -> str:
        sample = step["sample"]
        prediction, probabilities = self.workflow.load_and_predict(sample)
        scaled = self.workflow.scaler.transform([sample])[0]
        probability_rows = "".join(
            f"<tr><td>{name}</td><td>{value:.4f}</td></tr>"
            for name, value in zip(self.workflow.class_names, probabilities)
        )
        code = escape(step["inference_code"])
        return (
            "<h2>加载模型并执行推理</h2>"
            f"<p>{step['explanation']}</p>"
            f"<p><b>原始输入：</b>{sample}</p>"
            f"<p><b>Scaler 转换后：</b>{np.round(scaled, 4).tolist()}</p>"
            f"<pre style='background:#f1f3f2; padding:10px;'>{code}</pre>"
            f"<p><b>实际预测类别：</b>{self.workflow.class_names[prediction]}</p>"
            "<table cellspacing='0' cellpadding='6' border='1' width='100%'>"
            f"<tr><th>类别</th><th>概率</th></tr>{probability_rows}</table>"
            "<p>概率来自设置了 <b>probability=True</b> 的 SVC；三项概率之和为 1。"
            "部署时必须使用保存的 Scaler，而不是重新根据单个输入计算缩放参数。</p>"
        )

    def _knowledge_groups_html(self, step: dict) -> str:
        concept_titles = self.content.get("concept_titles", {})
        groups = []
        for index, group in enumerate(step.get("knowledge_groups", []), start=1):
            items = "".join(
                "<p style='margin-left: 12px;'>"
                f"<b>{concept_titles.get(key, key)}：</b>"
                f"{self.content['concepts'].get(key, '暂无说明。')}"
                "</p>"
                for key in group["concepts"]
            )
            groups.append(
                f"<h3>{index}. {group['title']}</h3>"
                f"<p>{group['description']}</p>{items}"
            )
        return "".join(groups)

    @staticmethod
    def _data_source_html(task: dict) -> str:
        return f"<h2>数据从哪里来？</h2><p>{task['source']}</p>"

    @staticmethod
    def _dataset_tables_html(task: dict) -> str:
        feature_rows = "".join(
            f"<tr><td>{item['feature']}</td><td>{item['meaning']}</td>"
            f"<td>{item['range']}</td></tr>"
            for item in task["feature_table"]
        )
        class_rows = "".join(
            f"<tr><td>{item['class']}</td><td>{item['samples']}</td></tr>"
            for item in task["class_table"]
        )
        table_style = "cellspacing='0' cellpadding='6' border='1' width='100%'"
        return (
            "<h3>特征说明</h3>"
            f"<table {table_style}><tr><th>特征</th><th>含义</th><th>范围（cm）</th></tr>"
            f"{feature_rows}</table>"
            "<h3>类别分布</h3>"
            f"<table {table_style}><tr><th>类别</th><th>样本数</th></tr>"
            f"{class_rows}</table>"
        )

    def _selected_solution_html(self, step_id: str) -> str:
        if step_id not in {"experiment", "select", "deploy"}:
            return ""
        solution_id = self.solution_selector.currentData()
        solution = next(item for item in self.content["solutions"] if item["id"] == solution_id)
        return (
            f"<h2>当前方案：{solution['title']}</h2>"
            f"<p><b>路线：</b>{solution['family']}</p>"
            f"<p><b>如何使用标签：</b>{solution['uses_labels']}</p>"
            f"<p><b>是否缩放：</b>{solution['scaling']}</p>"
            f"<p><b>部署方式：</b>{solution['deployment']}</p>"
        )

    def _render_step(self, step: dict) -> None:
        step_id = step["id"]
        is_table = step_id in {"data", "solutions"}
        if step_id == "evaluate":
            self.visual_stack.setCurrentWidget(self.evaluation_view)
        else:
            self.visual_stack.setCurrentWidget(self.data_table if is_table else self.canvas)
        self.experiment_parameters.setVisible(step_id == "experiment")
        self.visual_controls.setVisible(step_id in {"explore", "experiment"})
        is_explore = step_id == "explore"
        self.x_label.setVisible(is_explore)
        self.x_feature.setVisible(is_explore)
        self.y_label.setVisible(is_explore)
        self.y_feature.setVisible(is_explore)
        has_solution = step_id == "experiment"
        self.solution_label.setVisible(has_solution)
        self.solution_selector.setVisible(has_solution)

        if step_id == "data":
            self._show_data_table()
        elif step_id == "solutions":
            self._show_solutions_table()
        elif step_id == "evaluate":
            self._show_evaluation_view()
        else:
            solution_id = "svm" if step_id in {"select", "deploy"} else self.solution_selector.currentData()
            self.canvas.show_step(
                step_id,
                self.workflow,
                self.x_feature.currentData(),
                self.y_feature.currentData(),
                solution_id=solution_id,
            )
        self.runtime.setText(self._runtime_text(step_id))

    def _show_evaluation_view(self) -> None:
        headers = ["模型", "Accuracy", "Precision", "Recall", "F1", "正确", "错误"]
        self.evaluation_table.setColumnCount(len(headers))
        self.evaluation_table.setHorizontalHeaderLabels(headers)
        self.evaluation_table.setRowCount(len(CLASSIFIER_IDS))
        for row, solution_id in enumerate(CLASSIFIER_IDS):
            solution = next(item for item in self.content["solutions"] if item["id"] == solution_id)
            metrics = self.workflow.metrics(solution_id)
            values = [
                solution["title"],
                f"{metrics['accuracy']:.2%}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                str(metrics["correct"]),
                str(metrics["errors"]),
            ]
            for column, value in enumerate(values):
                self.evaluation_table.setItem(row, column, QTableWidgetItem(value))
        self.evaluation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.evaluation_table.verticalHeader().setVisible(False)
        self.evaluation_canvas.show_step(
            "confusion",
            self.workflow,
            solution_id=self.evaluation_selector.currentData(),
        )

    def _show_solutions_table(self) -> None:
        headers = ["候选方案", "路线", "标签", "缩放", "后续安排", "核心思路"]
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)
        self.data_table.setRowCount(len(self.content["solutions"]))
        self.data_table.setWordWrap(True)
        for row, solution in enumerate(self.content["solutions"]):
            values = [
                solution["title"], solution["family"], solution["uses_labels"],
                solution["scaling"], solution["role"], solution["summary"],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.data_table.setItem(row, column, item)
        header = self.data_table.horizontalHeader()
        for column in range(5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.data_table.resizeRowsToContents()

    def _algorithm_guidance_html(self, step: dict) -> str:
        guidance = step.get("algorithm_guidance", [])
        if not guidance:
            return ""
        rows = "".join(
            f"<tr><td><b>{item['algorithm']}</b></td><td>{item['scaling']}</td>"
            f"<td>{item['reason']}</td></tr>"
            for item in guidance
        )
        return (
            "<h2>哪些算法需要缩放？</h2>"
            "<table cellspacing='0' cellpadding='6' border='1'>"
            "<tr><th>算法</th><th>建议</th><th>原因</th></tr>"
            f"{rows}</table>"
        )

    def _runtime_text(self, step_id: str) -> str:
        selected_id = self.solution_selector.currentData()
        selected = next(item for item in self.content["solutions"] if item["id"] == selected_id)
        accuracy = self.workflow.accuracy(selected_id)
        svm = next(item for item in self.content["solutions"] if item["id"] == "svm")
        values = {
            "data": "当前状态 · 只有任务和数据，尚未选择解决方案。观察到 150 个样本、4 个特征和 3 个类别。",
            "explore": "观察到 · 当前数据没有缺失值，花瓣的两个特征呈现出明显的类别分离。",
            "solutions": "决策点 · 分类模型有多种选择；本章用逻辑回归演示后续流程，手工规则作为基线，K-Means 作为无监督对照。",
            "experiment": (
                f"第 {self.workflow.run_count} 次运行完成 · 全部监督模型已重新 fit，"
                f"总耗时 {self.workflow.run_ms:.2f} ms；{selected['title']} 准确率 {accuracy:.1%}。"
                if selected_id != "rules" else
                f"第 {self.workflow.run_count} 次运行完成 · 手工规则未训练模型，"
                f"已用当前阈值重新判断测试样本；准确率 {accuracy:.1%}。"
            ),
            "evaluate": (
                "统一结果 · 七种方案均使用相同的 30 个测试样本。"
                "柱高表示预测正确数量占测试样本总数的比例；还需结合稳定性和部署成本判断。"
            ),
            "select": f"最终选择 · SVM，测试集准确率 {self.workflow.accuracy('svm'):.1%}。模型和 Scaler 已保存。",
            "deploy": f"部署路径 · SVM：{svm['deployment']}",
        }
        return values[step_id]

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget { color: #28323c; font-size: 13px; }
            QMainWindow { background: #f3f1ea; }
            QLabel#title { color: #243b53; font-size: 28px; font-weight: 700; }
            QLabel#versionBadge { background: #3d7ea6; color: white; padding: 4px 8px; border-radius: 3px; font-weight: 700; }
            QLabel#subtitle { color: #6b7280; font-size: 14px; }
            QLabel#taskLabel { color: #59636e; font-weight: 600; }
            QComboBox#taskSelector { background: #fffdf8; border: 1px solid #b9b3a7; padding: 7px 10px; }
            QFrame#intro { background: #fff8e8; border: 1px solid #e8d8b5; border-left: 5px solid #e07a5f; }
            QFrame#panel { background: #fbfaf6; border: 1px solid #d9d5ca; }
            QLabel#heading { color: #243b53; font-size: 17px; font-weight: 700; }
            QLabel#progress { color: #e07a5f; font-weight: 600; }
            QLabel#summary { color: #4b5563; padding: 7px 0; }
            QLabel#runtime { background: #edf5f2; border: 1px solid #c9dfd7; padding: 10px; color: #276749; }
            QFrame#experimentParameters { background: #f4f7f5; border: 1px solid #cfdad5; }
            QListWidget { border: none; background: transparent; outline: none; }
            QListWidget::item { padding: 12px 10px; border-bottom: 1px solid #e6e1d7; }
            QListWidget::item:selected { background: #f2cc8f; color: #243b53; border-left: 4px solid #e07a5f; }
            QPushButton { background: #243b53; color: white; border: none; padding: 8px 13px; }
            QPushButton:disabled { background: #c4c7c9; }
            QPushButton#overviewButton { text-align: left; background: #edf2f5; color: #243b53; border-left: 4px solid #3d7ea6; padding: 10px; }
            QPushButton#overviewButton[active="true"] { background: #dbe9ef; border-left: 4px solid #e07a5f; }
            QTableWidget { background: #fffdf8; border: 1px solid #ddd8cc; gridline-color: #e7e2d8; alternate-background-color: #f6f3ec; }
            QHeaderView::section { background: #e8eef0; color: #243b53; padding: 8px; border: none; border-right: 1px solid #d0d7da; font-weight: 600; }
            QTextBrowser { background: #fffdf8; border: none; padding: 8px; }
        """)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
