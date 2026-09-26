from __future__ import annotations

from html import escape
from pathlib import Path
from time import perf_counter
from typing import Callable

import joblib
import numpy as np
import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import font_manager, rcParams
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QSpinBox,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).parent
KNOWLEDGE_PATH = ROOT / "knowledge" / "housing.yaml"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_IDS = ["linear", "ridge", "lasso", "decision-tree", "random-forest"]


for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(font_name.lower() in font.name.lower() for font in font_manager.fontManager.ttflist):
        rcParams["font.sans-serif"] = [font_name]
        break
rcParams["axes.unicode_minus"] = False


class HousingDataModel(QAbstractTableModel):
    def __init__(self, workflow: HousingWorkflow, limit: int) -> None:
        super().__init__()
        self.workflow = workflow
        self.limit = min(limit, len(workflow.features))
        self.headers = workflow.feature_names + ["MedHouseVal"]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self.limit

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        if index.column() < len(self.workflow.feature_names):
            value = self.workflow.features[index.row(), index.column()]
        else:
            value = self.workflow.targets[index.row()]
        return f"{value:.4f}"

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return section + 1


class HousingWorkflow:
    def __init__(self) -> None:
        dataset = fetch_california_housing(download_if_missing=False)
        self.features = dataset.data
        self.targets = dataset.target
        self.feature_names = list(dataset.feature_names)
        self.correlations = {
            name: float(np.corrcoef(self.features[:, index], self.targets)[0, 1])
            for index, name in enumerate(self.feature_names)
        }
        self.config = {"test_size": 0.2, "random_state": 42, "model_params": {}}
        self.configure(self.config)

    def configure(self, config: dict) -> None:
        self.config = config
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.features,
            self.targets,
            test_size=config["test_size"],
            random_state=config["random_state"],
        )
        self.scaler = StandardScaler().fit(self.train_x)
        self.train_scaled = self.scaler.transform(self.train_x)
        self.test_scaled = self.scaler.transform(self.test_x)
        params = config["model_params"]
        seed = config["random_state"]
        self.models = {
            "linear": (LinearRegression(), True),
            "ridge": (Ridge(**({"alpha": 1.0} | params.get("ridge", {}))), True),
            "lasso": (Lasso(**({"alpha": 0.01} | params.get("lasso", {}))), True),
            "decision-tree": (
                DecisionTreeRegressor(random_state=seed, **params.get("decision-tree", {})),
                False,
            ),
            "random-forest": (
                RandomForestRegressor(
                    random_state=seed,
                    n_jobs=-1,
                    **({"n_estimators": 50} | params.get("random-forest", {})),
                ),
                False,
            ),
        }
        self.predictions: dict[str, np.ndarray] = {}
        self.fit_times: dict[str, float] = {}

    def rerun(self, config: dict) -> None:
        self.configure(config)
        self.train_all()

    @property
    def is_trained(self) -> bool:
        return len(self.predictions) == len(self.models)

    def train_all(self) -> None:
        if self.is_trained:
            return
        for model_id, (model, needs_scaling) in self.models.items():
            train_x = self.train_scaled if needs_scaling else self.train_x
            test_x = self.test_scaled if needs_scaling else self.test_x
            started = perf_counter()
            model.fit(train_x, self.train_y)
            self.fit_times[model_id] = (perf_counter() - started) * 1000
            self.predictions[model_id] = model.predict(test_x)

    def metrics(self, model_id: str) -> dict[str, float]:
        prediction = self.predictions[model_id]
        mse = mean_squared_error(self.test_y, prediction)
        return {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(self.test_y, prediction)),
            "r2": float(r2_score(self.test_y, prediction)),
        }

    def save_best(self) -> tuple[Path, Path]:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        model_path = ARTIFACTS_DIR / "housing_best_model.pkl"
        scaler_path = ARTIFACTS_DIR / "housing_scaler.pkl"
        joblib.dump(self.models["random-forest"][0], model_path)
        joblib.dump(self.scaler, scaler_path)
        return model_path, scaler_path

    def load_and_predict(self, sample: list[float]) -> float:
        model_path, _ = self.save_best()
        model = joblib.load(model_path)
        return float(model.predict([sample])[0])


class HousingCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(6, 4), dpi=100, facecolor="#fbfaf6")
        super().__init__(self.figure)
        self.setMinimumHeight(340)

    def draw_step(
        self,
        step_id: str,
        workflow: HousingWorkflow,
        model_id: str,
        explore_mode: str = "target",
        feature_index: int = 0,
    ) -> None:
        self.figure.clear()
        if step_id == "select":
            importance_axis = self.figure.add_subplot(211)
            residual_axis = self.figure.add_subplot(212)
            axes = (importance_axis, residual_axis)
        else:
            axis = self.figure.add_subplot(111)
            axes = (axis,)
        for current_axis in axes:
            current_axis.set_facecolor("#fbfaf6")
        if step_id == "explore":
            if explore_mode == "target":
                axis.hist(workflow.targets, bins=50, color="#3d7ea6", edgecolor="#f7f5ef")
                axis.set_title("目标变量 MedHouseVal 分布")
                axis.set_xlabel("房价中位数（十万美元）")
                axis.set_ylabel("样本数量")
            else:
                name = workflow.feature_names[feature_index]
                correlation = workflow.correlations[name]
                axis.scatter(
                    workflow.features[:, feature_index],
                    workflow.targets,
                    alpha=0.12,
                    s=6,
                    color="#3d7ea6",
                    edgecolors="none",
                )
                axis.set_title(f"{name} 与房价 · 相关系数 {correlation:+.3f}")
                axis.set_xlabel(name)
                axis.set_ylabel("MedHouseVal（十万美元）")
        elif step_id == "experiment":
            prediction = workflow.predictions[model_id]
            axis.scatter(workflow.test_y, prediction, alpha=0.25, s=9, color="#3d7ea6")
            axis.plot([0, 5], [0, 5], "--", color="#e07a5f", label="理想预测")
            axis.set_title("预测值与真实值")
            axis.set_xlabel("真实房价（十万美元）")
            axis.set_ylabel("预测房价（十万美元）")
            axis.legend(frameon=False)
        elif step_id == "evaluate":
            prediction = workflow.predictions[model_id]
            residuals = workflow.test_y - prediction
            axis.hist(residuals, bins=50, color="#6a994e", edgecolor="#f7f5ef")
            axis.axvline(0, linestyle="--", color="#e07a5f")
            axis.set_title("残差分布（真实值 - 预测值）")
            axis.set_xlabel("残差（十万美元）")
            axis.set_ylabel("样本数量")
        elif step_id == "select":
            model = workflow.models["random-forest"][0]
            order = np.argsort(model.feature_importances_)[::-1]
            importance_axis.bar(
                range(len(order)),
                model.feature_importances_[order],
                color="#3d7ea6",
            )
            importance_axis.set_xticks(
                range(len(order)),
                [workflow.feature_names[index] for index in order],
                rotation=25,
            )
            importance_axis.set_title("Random Forest 特征重要性")
            importance_axis.set_ylabel("Importance")
            residuals = workflow.test_y - workflow.predictions["random-forest"]
            residual_axis.hist(
                residuals,
                bins=50,
                color="#6a994e",
                edgecolor="#f7f5ef",
            )
            residual_axis.axvline(0, linestyle="--", color="#e07a5f")
            residual_axis.set_title("Random Forest 残差分布")
            residual_axis.set_xlabel("残差（真实值 - 预测值，十万美元）")
            residual_axis.set_ylabel("样本数量")
        elif step_id == "deploy":
            sample = workflow.test_x[0].tolist()
            predicted_value = workflow.load_and_predict(sample)
            true_value = float(workflow.test_y[0])
            values = [true_value * 100000, predicted_value * 100000]
            bars = axis.bar(
                ["真实房价", "模型预测"],
                values,
                color=["#3d7ea6", "#e07a5f"],
                width=0.55,
            )
            axis.set_title("未参与训练的测试样本：真实值与预测值")
            axis.set_ylabel("美元")
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"${value:,.0f}",
                    ha="center",
                    va="bottom",
                )
        else:
            axis.axis("off")
        for current_axis in axes:
            current_axis.grid(axis="y", alpha=0.18)
        self.figure.tight_layout()
        self.draw()


class HousingTaskPage(QWidget):
    def __init__(self, on_back: Callable[[], None], on_ask_ai: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.on_back = on_back
        self.on_ask_ai = on_ask_ai
        self.content = yaml.safe_load(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        self.overview_html = ""
        self.data_step_html = ""
        self.knowledge_mode = ""
        self.model_parameter_values: dict[str, dict] = {}
        self.experiment_history: list[dict[str, str | float | int]] = []
        self.experiment_run_count = 0
        self.workflow: HousingWorkflow | None = None
        self.load_error = ""
        try:
            self.workflow = HousingWorkflow()
        except OSError as error:
            self.load_error = str(error)
        self._build_ui()
        self._apply_style()
        self.overview_html = self._task_data_html(overview=True)
        self.data_step_html = self._task_data_html(overview=False)
        self._show_overview()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        header = QHBoxLayout()
        title = QLabel("California Housing 房价回归")
        title.setObjectName("housingTitle")
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
        left.setObjectName("housingPanel")
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._heading("任务介绍"))
        self.overview_button = QPushButton("任务概览\n目标、来源与部分原始数据")
        self.overview_button.setObjectName("housingOverviewButton")
        self.overview_button.clicked.connect(self._show_overview)
        left_layout.addWidget(self.overview_button)
        left_layout.addSpacing(8)
        left_layout.addWidget(self._heading("回归学习流程"))
        self.step_list = QListWidget()
        for step in self.content["steps"]:
            item = QListWidgetItem(f"{step['title']}\n{step['short']}")
            self.step_list.addItem(item)
        self.step_list.currentRowChanged.connect(self._step_changed)
        left_layout.addWidget(self.step_list)
        splitter.addWidget(left)

        center = QFrame()
        center.setObjectName("housingPanel")
        center_layout = QVBoxLayout(center)
        control = QHBoxLayout()
        self.center_title = self._heading("任务与数据")
        control.addWidget(self.center_title)
        control.addStretch()
        self.model_controls = QWidget()
        model_controls_layout = QHBoxLayout(self.model_controls)
        model_controls_layout.setContentsMargins(0, 0, 0, 0)
        model_controls_layout.addWidget(QLabel("模型"))
        self.model_selector = QComboBox()
        for model in self.content["models"]:
            self.model_selector.addItem(model["title"], model["id"])
        self.model_selector.setCurrentIndex(self.model_selector.findData("random-forest"))
        self.model_selector.currentIndexChanged.connect(self._model_changed)
        model_controls_layout.addWidget(self.model_selector)
        control.addWidget(self.model_controls)
        self.explore_controls = QWidget()
        explore_layout = QHBoxLayout(self.explore_controls)
        explore_layout.setContentsMargins(0, 0, 0, 0)
        explore_layout.addWidget(QLabel("视图"))
        self.explore_mode = QComboBox()
        self.explore_mode.addItem("目标分布", "target")
        self.explore_mode.addItem("特征与房价", "feature")
        explore_layout.addWidget(self.explore_mode)
        self.explore_feature = QComboBox()
        if self.workflow is not None:
            for index, name in enumerate(self.workflow.feature_names):
                self.explore_feature.addItem(name, index)
        self.explore_mode.currentIndexChanged.connect(self._explore_changed)
        self.explore_feature.currentIndexChanged.connect(self._explore_changed)
        explore_layout.addWidget(self.explore_feature)
        control.addWidget(self.explore_controls)
        center_layout.addLayout(control)
        self.experiment_controls = QFrame()
        self.experiment_controls.setObjectName("housingExperimentControls")
        experiment_layout = QHBoxLayout(self.experiment_controls)
        experiment_layout.setContentsMargins(8, 6, 8, 6)
        self.test_size_input = QDoubleSpinBox()
        self.test_size_input.setRange(0.1, 0.5)
        self.test_size_input.setSingleStep(0.05)
        self.test_size_input.setValue(0.2)
        self.test_size_input.setPrefix("测试比例 ")
        experiment_layout.addWidget(self.test_size_input)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 9999)
        self.seed_input.setValue(42)
        self.seed_input.setPrefix("随机种子 ")
        experiment_layout.addWidget(self.seed_input)
        self.parameter_label = QLabel("模型参数")
        experiment_layout.addWidget(self.parameter_label)
        self.parameter_widgets: dict[str, QWidget] = {}
        self.parameter_layout = QHBoxLayout()
        experiment_layout.addLayout(self.parameter_layout)
        experiment_layout.addStretch()
        reset_button = QPushButton("恢复默认")
        reset_button.clicked.connect(self._reset_experiment)
        experiment_layout.addWidget(reset_button)
        run_button = QPushButton("运行实验")
        run_button.clicked.connect(self._run_experiment)
        experiment_layout.addWidget(run_button)
        center_layout.addWidget(self.experiment_controls)
        self._build_parameter_controls()
        history_title = QLabel("本次会话实验记录（最多保留 8 次）")
        history_title.setObjectName("housingHeading")
        self.history_title = history_title
        center_layout.addWidget(history_title)
        self.history_table = QTableWidget()
        self.history_table.setObjectName("housingHistoryTable")
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
        self.data_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        if self.workflow is not None:
            self.housing_data_model = HousingDataModel(
                self.workflow,
                self.content["task"]["display_limit"],
            )
            self.data_view.setModel(self.housing_data_model)
            self.data_view.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Interactive
            )
            self.data_view.horizontalHeader().setDefaultSectionSize(105)
        self.center_stack.addWidget(self.data_view)
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.data_table.setAlternatingRowColors(True)
        self.data_table_mode = ""
        self.center_stack.addWidget(self.data_table)
        self.canvas = HousingCanvas()
        self.center_stack.addWidget(self.canvas)
        self.metrics_table = QTableWidget()
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.setAlternatingRowColors(True)
        self.center_stack.addWidget(self.metrics_table)
        self.deploy_view = QWidget()
        deploy_layout = QVBoxLayout(self.deploy_view)
        deploy_layout.setContentsMargins(0, 0, 0, 0)
        self.deploy_input_table = QTableWidget()
        self.deploy_input_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.deploy_input_table.setAlternatingRowColors(True)
        self.deploy_input_table.verticalHeader().setVisible(False)
        self.deploy_input_table.setMaximumHeight(250)
        deploy_layout.addWidget(self.deploy_input_table)
        self.deploy_flow = QLabel("加载模型文件 → 准备 8 个输入特征 → 保持特征顺序 → 执行 predict() → 输出连续房价")
        self.deploy_flow.setObjectName("housingDeployFlow")
        self.deploy_flow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.deploy_flow.setWordWrap(True)
        deploy_layout.addWidget(self.deploy_flow)
        self.deploy_canvas = HousingCanvas()
        self.deploy_canvas.setMinimumHeight(300)
        deploy_layout.addWidget(self.deploy_canvas, 1)
        self.center_stack.addWidget(self.deploy_view)
        center_layout.addWidget(self.center_stack, 1)
        self.status = QLabel()
        self.status.setObjectName("housingStatus")
        self.status.setWordWrap(True)
        center_layout.addWidget(self.status)
        splitter.addWidget(center)

        right = QFrame()
        right.setObjectName("housingPanel")
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
        label.setObjectName("housingHeading")
        return label

    def _step_changed(self, row: int) -> None:
        if row < 0:
            return
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
        if self.workflow is not None and self.workflow.is_trained:
            model_id = "random-forest" if step and step["id"] in {"select", "deploy"} else self.model_selector.currentData()
            metrics = self.workflow.metrics(model_id)
            sections.append(
                f"当前模型: {model_id}; 参数: {self.workflow.config['model_params'].get(model_id, {})}; "
                f"RMSE={metrics['rmse']:.4f}; MAE={metrics['mae']:.4f}; R²={metrics['r2']:.4f}"
            )
        return "\n".join(sections)

    def _show_overview(self) -> None:
        self.step_list.clearSelection()
        self.step_list.setCurrentRow(-1)
        self.overview_button.setProperty("active", True)
        self.overview_button.style().unpolish(self.overview_button)
        self.overview_button.style().polish(self.overview_button)
        self.model_controls.hide()
        self.explore_controls.hide()
        self.experiment_controls.hide()
        self.history_title.hide()
        self.history_table.hide()
        self.center_title.setText(self.content["task"]["title"])
        if self.workflow is None:
            self.data_table.setRowCount(0)
            self.center_stack.setCurrentWidget(self.data_table)
            self.status.setText("数据不可用。首次使用需要联网下载 California Housing 数据集。")
        else:
            self._show_data()
            self.status.setText(
                "数据概览 · 共 20,640 条街区数据，界面展示前 200 条；"
                "当前尚未选择回归模型。"
            )
        if self.knowledge_mode != "overview":
            self.knowledge.setHtml(self.overview_html)
            self.knowledge_mode = "overview"

    def _refresh_current_step(self) -> None:
        self._step_changed(self.step_list.currentRow())

    def _model_changed(self) -> None:
        self._build_parameter_controls()
        self._refresh_current_step()

    def _clear_parameter_controls(self) -> None:
        while self.parameter_layout.count():
            item = self.parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_parameter_controls(self) -> None:
        if not hasattr(self, "parameter_layout"):
            return
        self._clear_parameter_controls()
        model_id = self.model_selector.currentData()
        model = next(item for item in self.content["models"] if item["id"] == model_id)
        saved = self.model_parameter_values.get(model_id, {})
        self.parameter_widgets = {}
        self.parameter_label.setVisible(bool(model["parameters"]))
        for spec in model["parameters"]:
            if spec["type"] == "float":
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setRange(spec["min"], spec["max"])
                widget.setSingleStep(spec["step"])
            else:
                widget = QSpinBox()
                widget.setRange(spec["min"], spec["max"])
                widget.setSingleStep(spec["step"])
            value = saved.get(spec["id"], spec["default"])
            widget.setValue(0 if value is None else value)
            widget.setToolTip(f"默认值：{spec['default']}。{spec['description']}")
            self.parameter_widgets[spec["id"]] = widget
            self.parameter_layout.addWidget(QLabel(spec["title"]))
            self.parameter_layout.addWidget(widget)

    def _run_experiment(self) -> None:
        model_id = self.model_selector.currentData()
        model_spec = next(item for item in self.content["models"] if item["id"] == model_id)
        values = {}
        for spec in model_spec["parameters"]:
            value = self.parameter_widgets[spec["id"]].value()
            if spec["type"] == "optional_int" and value == 0:
                value = None
            values[spec["id"]] = value
        self.model_parameter_values[model_id] = values
        self.status.setText(
            "正在使用当前参数重新划分数据并训练 5 个回归模型，请稍候。"
        )
        QApplication.processEvents()
        assert self.workflow is not None
        self.workflow.rerun({
            "test_size": self.test_size_input.value(),
            "random_state": self.seed_input.value(),
            "model_params": self.model_parameter_values,
        })
        self._record_experiment(model_id)
        self._refresh_current_step()

    def _reset_experiment(self) -> None:
        self.test_size_input.setValue(0.2)
        self.seed_input.setValue(42)
        self.model_parameter_values.clear()
        self.experiment_history.clear()
        self.experiment_run_count = 0
        self._build_parameter_controls()
        if self.workflow is not None:
            self.workflow.configure({
                "test_size": 0.2,
                "random_state": 42,
                "model_params": {},
            })
        self._refresh_current_step()

    def _record_experiment(self, model_id: str) -> None:
        assert self.workflow is not None
        model = next(item for item in self.content["models"] if item["id"] == model_id)
        self.experiment_run_count += 1
        parameters = self.model_parameter_values.get(model_id, {})
        parameter_text = ", ".join(
            f"{spec['title']}={parameters.get(spec['id'], spec['default'])}"
            for spec in model["parameters"]
        ) or "默认配置"
        metrics = self.workflow.metrics(model_id)
        self.experiment_history.append({
            "run": self.experiment_run_count,
            "model": model["title"],
            "split": f"测试 {self.workflow.config['test_size']:.0%} / seed {self.workflow.config['random_state']}",
            "parameters": parameter_text,
            "r2": metrics["r2"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "fit_time": self.workflow.fit_times[model_id],
        })
        self.experiment_history = self.experiment_history[-8:]
        self._show_experiment_history()

    def _show_experiment_history(self) -> None:
        headers = ["运行", "模型", "流程配置", "模型参数", "R²", "MAE", "RMSE", "fit 耗时"]
        self.history_table.setColumnCount(len(headers))
        self.history_table.setHorizontalHeaderLabels(headers)
        self.history_table.setRowCount(len(self.experiment_history))
        for row, record in enumerate(reversed(self.experiment_history)):
            values = [
                str(record["run"]), str(record["model"]), str(record["split"]),
                str(record["parameters"]), f"{record['r2']:.4f}",
                f"{record['mae']:.4f}", f"{record['rmse']:.4f}",
                f"{record['fit_time']:.1f} ms",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.history_table.setItem(row, column, item)
        header = self.history_table.horizontalHeader()
        for column in (0, 1, 4, 5, 6, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for column in (2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

    def _explore_changed(self) -> None:
        self.explore_feature.setVisible(self.explore_mode.currentData() == "feature")
        row = self.step_list.currentRow()
        if row >= 0 and self.content["steps"][row]["id"] == "explore":
            self._show_step(self.content["steps"][row])

    def _show_step(self, step: dict) -> None:
        self.overview_button.setProperty("active", False)
        self.overview_button.style().unpolish(self.overview_button)
        self.overview_button.style().polish(self.overview_button)
        self.center_title.setText(step["title"])
        self.model_controls.setVisible(step["id"] in {"experiment", "evaluate"})
        self.experiment_controls.setVisible(step["id"] == "experiment")
        self.history_title.setVisible(step["id"] == "experiment")
        self.history_table.setVisible(step["id"] == "experiment")
        self.explore_controls.setVisible(step["id"] == "explore")
        self.explore_feature.setVisible(
            step["id"] == "explore" and self.explore_mode.currentData() == "feature"
        )
        self.status.setVisible(step["id"] in {"data", "explore", "solutions", "experiment"})
        if self.workflow is None:
            self.center_stack.setCurrentWidget(self.data_table)
            self.data_table.setRowCount(0)
            self.status.setText("数据不可用。首次使用需要联网下载 California Housing 数据集。")
            self.knowledge.setHtml(
                f"<h2>{step['title']}</h2><p>{step['explanation']}</p>"
                f"<p><b>加载错误：</b>{escape(self.load_error)}</p>"
            )
            return
        needs_training = step["id"] in {"experiment", "evaluate", "select", "deploy"}
        if needs_training and not self.workflow.is_trained:
            self.status.setText(
                "正在训练 5 个回归模型，请稍候。California Housing 使用全部 20,640 条数据，"
                "Random Forest 通常需要约 1-3 秒。"
            )
            QApplication.processEvents()
            self.workflow.train_all()
        model_id = (
            "random-forest"
            if step["id"] in {"select", "deploy"}
            else self.model_selector.currentData()
        )
        model = next(item for item in self.content["models"] if item["id"] == model_id)
        if step["id"] == "data":
            self._show_data()
        elif step["id"] == "solutions":
            self._show_models()
        elif step["id"] == "evaluate":
            self._show_metrics()
        elif step["id"] == "deploy":
            self._show_deploy_view()
        else:
            self.center_stack.setCurrentWidget(self.canvas)
            self.canvas.draw_step(
                step["id"],
                self.workflow,
                model_id,
                self.explore_mode.currentData(),
                self.explore_feature.currentData() or 0,
            )
        if step["id"] == "data":
            self.status.setText(
                "数据事实 · 共 20,640 条，8 个输入特征，目标为连续数值 MedHouseVal；"
                "界面展示前 200 条，训练使用全部数据。"
            )
            self.knowledge.setHtml(self._knowledge_html(step, model, None))
            self.knowledge_mode = "data"
            return
        if step["id"] in {"explore", "solutions"}:
            if step["id"] == "explore":
                if self.explore_mode.currentData() == "target":
                    self.status.setText(
                        "数据探索 · 当前查看目标 MedHouseVal 的整体分布；此阶段不涉及模型。"
                    )
                else:
                    feature_index = self.explore_feature.currentData() or 0
                    feature_name = self.workflow.feature_names[feature_index]
                    correlation = self.workflow.correlations[feature_name]
                    self.status.setText(
                        f"数据探索 · {feature_name} 与 MedHouseVal 的 Pearson 相关系数为 "
                        f"{correlation:+.4f}；此阶段不涉及模型。"
                    )
            else:
                self.status.setText(
                    "方案选择 · 当前只比较候选方法的特点和适用边界，尚未选定或训练模型。"
                )
            self.knowledge.setHtml(self._knowledge_html(step, model, None))
            self.knowledge_mode = step["id"]
            return
        metrics = self.workflow.metrics(model_id)
        if needs_training:
            self.status.setText(
                f"实际运行 · {model['title']} · fit {self.workflow.fit_times[model_id]:.1f} ms · "
                f"RMSE {metrics['rmse']:.4f} · MAE {metrics['mae']:.4f} · R² {metrics['r2']:.4f}"
            )
        self.knowledge.setHtml(self._knowledge_html(step, model, metrics))
        self.knowledge_mode = step["id"]

    def _show_data(self) -> None:
        assert self.workflow is not None
        self.center_stack.setCurrentWidget(self.data_view)

    def _show_models(self) -> None:
        if self.data_table_mode == "models":
            self.center_stack.setCurrentWidget(self.data_table)
            return
        headers = ["候选模型", "类型", "缩放", "核心思路"]
        self.data_table.setUpdatesEnabled(False)
        self.data_table.clear()
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)
        self.data_table.setRowCount(len(self.content["models"]))
        for row, model in enumerate(self.content["models"]):
            values = [model["title"], model["family"], model["scaling"], model["summary"]]
            for column, value in enumerate(values):
                self.data_table.setItem(row, column, QTableWidgetItem(value))
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.data_table.setUpdatesEnabled(True)
        self.data_table_mode = "models"
        self.center_stack.setCurrentWidget(self.data_table)

    def _show_metrics(self) -> None:
        assert self.workflow is not None
        headers = ["模型", "MSE", "RMSE", "MAE", "R²", "fit 耗时"]
        self.metrics_table.setColumnCount(len(headers))
        self.metrics_table.setHorizontalHeaderLabels(headers)
        self.metrics_table.setRowCount(len(MODEL_IDS))
        for row, model_id in enumerate(MODEL_IDS):
            model = next(item for item in self.content["models"] if item["id"] == model_id)
            metrics = self.workflow.metrics(model_id)
            values = [
                model["title"],
                f"{metrics['mse']:.4f}",
                f"{metrics['rmse']:.4f}",
                f"{metrics['mae']:.4f}",
                f"{metrics['r2']:.4f}",
                f"{self.workflow.fit_times[model_id]:.1f} ms",
            ]
            for column, value in enumerate(values):
                self.metrics_table.setItem(row, column, QTableWidgetItem(value))
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        self.center_stack.setCurrentWidget(self.metrics_table)

    def _show_deploy_view(self) -> None:
        assert self.workflow is not None
        sample = self.workflow.test_x[0]
        feature_info = {
            item["feature"]: item
            for item in self.content["task"]["feature_table"]
        }
        headers = ["输入字段", "含义", "输入值", "单位/备注"]
        self.deploy_input_table.setColumnCount(len(headers))
        self.deploy_input_table.setHorizontalHeaderLabels(headers)
        self.deploy_input_table.setRowCount(len(self.workflow.feature_names))
        for row, (name, value) in enumerate(zip(self.workflow.feature_names, sample)):
            info = feature_info[name]
            values = [name, info["meaning"], f"{value:.4f}", info["note"]]
            for column, text in enumerate(values):
                self.deploy_input_table.setItem(row, column, QTableWidgetItem(text))
        header = self.deploy_input_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.deploy_canvas.draw_step(
            "deploy",
            self.workflow,
            "random-forest",
        )
        self.center_stack.setCurrentWidget(self.deploy_view)

    def _knowledge_html(
        self,
        step: dict,
        model: dict,
        metrics: dict[str, float] | None,
    ) -> str:
        task = self.content["task"]
        if step["id"] == "data":
            return self.data_step_html
        if step["id"] == "explore":
            assert self.workflow is not None
            findings = "".join(
                f"<li><b>{item['title']}：</b>{item['description']}</li>"
                for item in step["findings"]
            )
            ranking = sorted(
                self.workflow.correlations.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            rows = "".join(
                f"<tr><td>{name}</td><td>{value:+.4f}</td>"
                f"<td>{self._correlation_label(value)}</td></tr>"
                for name, value in ranking
            )
            if self.explore_mode.currentData() == "target":
                current = (
                    "当前显示目标变量分布。可以先观察大部分房价集中区间、右偏情况，"
                    "以及 5.0 附近的数据上限。"
                )
            else:
                feature_index = self.explore_feature.currentData() or 0
                name = self.workflow.feature_names[feature_index]
                value = self.workflow.correlations[name]
                current = (
                    f"当前选择 <b>{name}</b>，与房价的 Pearson 相关系数为 "
                    f"<b>{value:+.4f}</b>，属于{self._correlation_label(value)}。"
                )
            return (
                f"<h2>{step['title']}：{step['short']}</h2>"
                f"<p><b>探索目标：</b>{step['goal']}</p><p>{step['explanation']}</p>"
                f"<h3>关键发现</h3><ul>{findings}</ul>"
                f"<p>{current}</p>"
                "<h3>八个特征与房价的线性相关性</h3>"
                "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
                f"<tr><th>特征</th><th>相关系数</th><th>观察</th></tr>{rows}</table>"
                "<h3>如何理解</h3>"
                "<p><b>MedInc</b> 的线性关系最明显；AveRooms、Latitude、HouseAge 只有较弱关系；"
                "AveBedrms、Longitude、Population、AveOccup 的单变量线性相关性接近 0。</p>"
                "<p>相关性弱不代表没有预测价值。经纬度需要组合表达位置，树模型也可能从"
                "非线性关系和特征交互中提取信息。相关系数只能衡量单变量线性关系。</p>"
            )
        if step["id"] == "solutions":
            model_rows = "".join(
                f"<tr><td><b>{item['title']}</b></td><td>{item['family']}</td>"
                f"<td>{item['scaling']}</td><td>{item['summary']}</td></tr>"
                for item in self.content["models"]
            )
            return (
                f"<h2>{step['title']}：{step['short']}</h2>"
                f"<p><b>目标：</b>{step['goal']}</p><p>{step['explanation']}</p>"
                "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
                f"<tr><th>候选模型</th><th>类型</th><th>缩放</th><th>作用</th></tr>{model_rows}</table>"
                "<p>这一阶段只建立候选方案，不预先指定最终模型。进入方案实验后才执行训练和比较。</p>"
            )
        if step["id"] == "experiment":
            assert self.workflow is not None and metrics is not None
            current_parameters = self.model_parameter_values.get(model["id"], {})
            parameter_rows = "".join(
                f"<tr><td>{spec['title']}</td><td>{spec['default']}</td>"
                f"<td>{current_parameters.get(spec['id'], spec['default'])}</td>"
                f"<td>{spec['description']}</td></tr>"
                for spec in model["parameters"]
            )
            parameter_table = (
                "<p>Linear Regression 没有本页需要调整的核心超参数，"
                "仍可改变测试比例和随机种子。</p>"
                if not model["parameters"] else
                "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
                "<tr><th>参数</th><th>默认值</th><th>当前值</th><th>作用</th></tr>"
                f"{parameter_rows}</table>"
            )
            return (
                f"<h2>{step['title']}：{model['title']}</h2>"
                f"<p><b>实验目标：</b>{step['goal']}</p><p>{step['explanation']}</p>"
                f"<p><b>当前流程参数：</b>测试比例 {self.workflow.config['test_size']:.0%}，"
                f"随机种子 {self.workflow.config['random_state']}；训练集 "
                f"{len(self.workflow.train_y):,} 条，测试集 {len(self.workflow.test_y):,} 条。</p>"
                f"<h3>当前模型有哪些可调参数</h3>{parameter_table}"
                f"<h3>为什么选择这种预处理</h3><p>{model['summary']}</p>"
                f"<p><b>特征缩放：</b>{model['scaling']}。Scaler 只使用训练集拟合，"
                "避免测试集信息进入训练过程。</p>"
                f"<h3>本次真实训练结果</h3>"
                f"<p>已重新创建并调用 <b>{model['title']}.fit()</b>，"
                f"fit 耗时 {self.workflow.fit_times[model['id']]:.1f} ms。</p>"
                f"<p>MSE {metrics['mse']:.4f}，RMSE {metrics['rmse']:.4f}，"
                f"MAE {metrics['mae']:.4f}，R² {metrics['r2']:.4f}。</p>"
                "<p><b>左侧图：</b>横轴是真实房价，纵轴是预测房价。点越接近红色虚线，预测越准确。</p>"
            )
        if step["id"] == "select":
            assert self.workflow is not None and metrics is not None
            forest = self.workflow.models["random-forest"][0]
            importance_order = np.argsort(forest.feature_importances_)[::-1]
            importance_rows = "".join(
                f"<tr><td>{self.workflow.feature_names[index]}</td>"
                f"<td>{forest.feature_importances_[index]:.2%}</td></tr>"
                for index in importance_order
            )
            prediction = self.workflow.predictions["random-forest"]
            high_mask = self.workflow.test_y > 4.0
            high_count = int(high_mask.sum())
            high_bias = float((prediction[high_mask] - self.workflow.test_y[high_mask]).mean())
            high_mae = float(np.abs(prediction[high_mask] - self.workflow.test_y[high_mask]).mean())
            improvements = "".join(
                f"<li>{item}</li>" for item in step["improvements"]
            )
            model_path, scaler_path = self.workflow.save_best()
            return (
                "<h2>最终选择：Random Forest</h2>"
                f"<p>{step['explanation']}</p>"
                "<h3>1. 为什么选择它</h3>"
                f"<p>当前相同测试集上，Random Forest 的 RMSE 为 <b>{metrics['rmse']:.4f}</b>，"
                f"MAE 为 <b>{metrics['mae']:.4f}</b>，R² 为 <b>{metrics['r2']:.4f}</b>。"
                "它能表达收入、位置、居住特征之间的非线性关系和特征交互。</p>"
                "<h3>2. 哪些特征贡献较大</h3>"
                "<p>左侧图展示当前随机森林的特征重要性。重要性表示该特征在树分裂中带来的"
                "误差下降贡献，不表示因果关系。</p>"
                "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
                f"<tr><th>特征</th><th>重要性</th></tr>{importance_rows}</table>"
                "<p>MedInc 通常最重要；AveOccup 排名靠前；经纬度共同表达地理位置；"
                "AveBedrms 与 AveRooms 信息部分重叠，因此重要性通常较低。</p>"
                "<h3>3. 不能只看总体 R²</h3>"
                f"<p>{step['boundary']}</p>"
                "<p><b>残差 = 真实值 - 预测值。</b>残差接近 0 表示预测接近真实值；"
                "正残差表示模型低估，负残差表示模型高估。下方残差图如果以 0 为中心且两侧较均衡，"
                "说明总体系统性偏差较小；右侧长尾则表示部分样本被明显低估。</p>"
                f"<p>当前测试集中真实房价大于 4.0 的样本有 <b>{high_count}</b> 条。"
                f"该区域平均预测偏差为 <b>{high_bias:+.4f}</b>（预测值减真实值），"
                f"MAE 为 <b>{high_mae:.4f}</b>，即平均绝对误差约 ${high_mae * 100000:,.0f}。"
                "负偏差表示模型在高价区整体倾向低估。</p>"
                "<p>数据目标在 5.0 附近存在上限，这也会影响最高价区的学习与评估，"
                "不能把图中的边界简单理解为真实市场上限。</p>"
                "<h3>4. 如果业务关注高价房产</h3>"
                f"<ol>{improvements}</ol>"
                "<p><b>评估原则：</b>R²、RMSE/MAE、残差分布和预测值 vs 真实值应结合观察。</p>"
                "<h3>5. 保存部署产物</h3>"
                f"<p>已保存 <b>{model_path.name}</b> 和 <b>{scaler_path.name}</b>。"
                "当前 Random Forest 使用原始特征，不依赖缩放；保留 Scaler 是为了与其他回归模型"
                "和统一部署流水线兼容，实际部署时应明确当前模型是否需要它。</p>"
            )
        observations = "".join(f"<li>{item}</li>" for item in self.content["observations"])
        extra = ""
        if step["id"] == "evaluate":
            all_metrics = {
                model_id: self.workflow.metrics(model_id)  # type: ignore[union-attr]
                for model_id in MODEL_IDS
            }
            best_id = min(MODEL_IDS, key=lambda item: all_metrics[item]["rmse"])
            best_model = next(item for item in self.content["models"] if item["id"] == best_id)
            linear_r2 = [all_metrics[item]["r2"] for item in ("linear", "ridge", "lasso")]
            tree_r2 = all_metrics["decision-tree"]["r2"]
            forest_r2 = all_metrics["random-forest"]["r2"]
            dynamic_observations = (
                f"<li><b>{best_model['title']} 当前表现最好：</b>"
                f"RMSE {all_metrics[best_id]['rmse']:.4f}，R² {all_metrics[best_id]['r2']:.4f}。</li>"
                f"<li><b>三个线性模型表现接近：</b>R² 位于 "
                f"{min(linear_r2):.4f}-{max(linear_r2):.4f}，说明正则化在当前默认参数下没有改变整体结论。</li>"
                f"<li><b>树模型能捕捉非线性：</b>Decision Tree R² {tree_r2:.4f}，"
                f"Random Forest R² {forest_r2:.4f}；集成通常比单棵树更稳定。</li>"
                f"<li><b>R² {forest_r2:.2f} 的含义：</b>模型解释了约 {forest_r2:.0%} 的测试集房价变化，"
                "未解释部分可能来自学区、交通和房屋具体状况等未收录特征。</li>"
            )
            extra = (
                f"<h3>当前运行的关键观察</h3><ol>{dynamic_observations}</ol>"
                f"<h3>评估边界</h3><ul>{observations}</ul>"
            )
        if step["id"] == "deploy":
            assert self.workflow is not None
            sample = self.workflow.test_x[0].tolist()
            true_value = float(self.workflow.test_y[0])
            prediction = self.workflow.load_and_predict(sample)  # type: ignore[union-attr]
            error = prediction - true_value
            absolute_error = abs(error)
            relative_error = absolute_error / true_value if true_value else 0.0
            model_path = ARTIFACTS_DIR / "housing_best_model.pkl"
            return (
                "<h2>保存、加载与预测：验证一个未见样本</h2>"
                f"<p>{step['explanation']}</p>"
                "<h3>1. 输入是什么</h3>"
                "<p>中间表格列出一个来自测试集的真实街区样本。它包含 8 个输入特征，"
                "该样本没有参与模型训练，因此可以用于检验模型对未见数据的预测。</p>"
                "<p><b>特征顺序：</b>MedInc → HouseAge → AveRooms → AveBedrms → "
                "Population → AveOccup → Latitude → Longitude。</p>"
                "<h3>2. 加载并执行什么</h3>"
                f"<p>从 <b>{model_path.name}</b> 加载已保存的 Random Forest，保持与训练时"
                "相同的字段定义和顺序，然后调用 <b>predict()</b>。Random Forest 当前使用原始特征，"
                "不需要 StandardScaler 转换。</p>"
                "<h3>3. 输出是什么</h3>"
                "<p>模型输出一个连续数值，单位是十万美元。界面同时换算成美元显示。</p>"
                "<table cellspacing='0' cellpadding='6' border='1' width='100%'>"
                "<tr><th>比较项</th><th>十万美元</th><th>美元</th></tr>"
                f"<tr><td>真实房价</td><td>{true_value:.4f}</td><td>${true_value * 100000:,.0f}</td></tr>"
                f"<tr><td>模型预测</td><td>{prediction:.4f}</td><td>${prediction * 100000:,.0f}</td></tr>"
                f"<tr><td>预测误差</td><td>{error:+.4f}</td><td>{error * 100000:+,.0f}</td></tr>"
                "</table>"
                "<h3>4. 如何比较</h3>"
                f"<p>本次绝对误差为 <b>${absolute_error * 100000:,.0f}</b>，相对误差为 "
                f"<b>{relative_error:.1%}</b>。预测误差使用“预测值 - 真实值”："
                f"当前为 <b>{error:+.4f}</b>，因此模型对此样本"
                f"{'高估' if error > 0 else '低估' if error < 0 else '预测一致'}。</p>"
                "<p><b>中间图说明：</b>蓝柱是真实房价，橙柱是加载模型后的预测房价。"
                "两柱越接近，说明该样本预测越准确。单个样本只能验证一次推理链路，"
                "整体模型质量仍应参考完整测试集的 RMSE、MAE、R² 和残差分布。</p>"
            )
        return (
            f"<h2>{step['title']}：{step['short']}</h2>"
            f"<p><b>目标：</b>{step['goal']}</p><p>{step['explanation']}</p>"
            f"<h3>当前模型：{model['title']}</h3>"
            f"<p>{model['summary']}</p><p><b>缩放：</b>{model['scaling']}</p>"
            + (
                f"<p><b>实际结果：</b>RMSE {metrics['rmse']:.4f}，"
                f"MAE {metrics['mae']:.4f}，R² {metrics['r2']:.4f}</p>"
                if metrics is not None else
                "<p><b>当前阶段：</b>尚未训练模型。</p>"
            )
            + extra
        )

    @staticmethod
    def _correlation_label(value: float) -> str:
        strength = abs(value)
        if strength >= 0.6:
            return "较明显线性关系"
        if strength >= 0.1:
            return "较弱线性关系"
        return "线性关系不明显"

    def _task_data_html(self, overview: bool) -> str:
        task = self.content["task"]
        feature_rows = "".join(
            f"<tr><td>{item['feature']}</td><td>{item['meaning']}</td><td>{item['note']}</td></tr>"
            for item in task["feature_table"]
        )
        heading = "这个任务要做什么？" if overview else "1. 当前的任务"
        return (
            f"<h2>{heading}</h2><p>{task['current_task']}</p>"
            f"<h2>2. 当前的数据与目标</h2><p>{task['current_data']}</p>"
            f"<p><b>来源：</b>{task['source']}</p><p>{task['dataset_summary']}</p>"
            "<table cellspacing='0' cellpadding='5' border='1' width='100%'>"
            f"<tr><th>特征</th><th>含义</th><th>备注</th></tr>{feature_rows}</table>"
            f"<p><b>展示边界：</b>共 20,640 条，仅显示前 {task['display_limit']} 条；"
            "后续训练和评估使用全部数据。</p>"
            f"<h2>3. 这是哪类问题，接下来用什么方法？</h2><p>{task['method_intro']}</p>"
            f"<p><b>需要回答：</b>{task['question']}</p>"
        )

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget { color: #28323c; font-size: 13px; }
            HousingTaskPage { background: #f3f1ea; }
            QLabel#housingTitle { color: #243b53; font-size: 24px; font-weight: 700; }
            QFrame#housingPanel { background: #fbfaf6; border: 1px solid #d9d5ca; }
            QLabel#housingHeading { color: #243b53; font-size: 17px; font-weight: 700; }
            QLabel#housingStatus { background: #edf5f2; border: 1px solid #c9dfd7; padding: 9px; color: #276749; }
            QLabel#housingDeployFlow { background: #edf5f2; border: 1px solid #c9dfd7; padding: 9px; color: #276749; }
            QFrame#housingExperimentControls { background: #f4f7f5; border: 1px solid #cfdad5; }
            QListWidget, QTextBrowser, QTableWidget { background: #fffdf8; border: none; }
            QListWidget::item { padding: 11px 8px; border-bottom: 1px solid #e6e1d7; }
            QListWidget::item:selected { background: #f2cc8f; color: #243b53; border-left: 4px solid #e07a5f; }
            QPushButton { background: #243b53; color: white; border: none; padding: 8px 13px; }
            QPushButton#housingOverviewButton { text-align: left; background: #edf2f5; color: #243b53; border-left: 4px solid #3d7ea6; padding: 10px; }
            QPushButton#housingOverviewButton[active="true"] { background: #dbe9ef; border-left: 4px solid #e07a5f; }
            QHeaderView::section { background: #e8eef0; color: #243b53; padding: 7px; font-weight: 600; }
        """)
