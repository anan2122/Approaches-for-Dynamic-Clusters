"""PyQt5 GUI for selecting algorithms and visualizing clustering output."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QStandardItem, QStandardItemModel

from clustering.clustering_engine import ClusteringEngine
from clustering.clustering_utils import ClusteringResult
from data_loader import DataLoader
import time


def format_stats_for_output(stats: Dict[str, int]) -> str:
    """Render statistics into a compact key=value string."""
    ordered_keys = [
        "total_clusters",
        "target_cluster_size",
        "smallest_cluster_size",
        "outliers",
        "changed_cluster_points",
    ]
    return ", ".join(f"{key}={stats.get(key, 0)}" for key in ordered_keys)


def format_points_for_output(points: np.ndarray) -> str:
    """Render point arrays as tuples for plain-text output."""
    point_tuples = [tuple(float(value) for value in row) for row in points.tolist()]
    return str(point_tuples)


class ClusteringWorker(QObject):
    """Background worker that runs clustering pipeline and writes output."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        file_path: str,
        selected_algorithms: List[str],
        metric_name: str,
        algorithm_parameters: Dict[str, Dict[str, int | float | str]],
        output_file_path: str,
    ):
        super().__init__()
        self.file_path = file_path
        self.selected_algorithms = selected_algorithms
        self.metric_name = metric_name
        self.algorithm_parameters = algorithm_parameters
        self.output_file_path = Path(output_file_path)

    def run(self) -> None:
        """Execute full loading/clustering/output pipeline in worker thread."""
        try:
            self.progress.emit("Reading file...")
            data_loader = DataLoader()
            clustering_engine = ClusteringEngine()

            timestamp_points = data_loader.load_points(self.file_path)
            non_empty_timestamps = [arr for arr in timestamp_points if arr.size > 0]
            if not non_empty_timestamps:
                raise ValueError("No valid 3D points found in file.")

            algorithm_run_results: Dict[str, List[Tuple[np.ndarray, ClusteringResult]]] = {}
            timing_results: Dict[str, Dict[str, object]] = {}

            for algorithm_name in self.selected_algorithms:
                run_results: List[Tuple[np.ndarray, ClusteringResult]] = []
                parameters = self.algorithm_parameters.get(algorithm_name, {})
                if not isinstance(parameters, dict):
                    parameters = {}

                self.progress.emit(f"Running {algorithm_name}...")
                total_timestamps = len(non_empty_timestamps)
                per_timestamp_times: list[float] = []
                alg_start = time.perf_counter()
                for timestamp_index, timestamp_dataset in enumerate(
                    non_empty_timestamps,
                    start=1,
                ):
                    self.progress.emit(
                        f"Processing timestamp {timestamp_index}/{total_timestamps}..."
                    )

                    t0 = time.perf_counter()
                    result = clustering_engine.run_algorithm(
                        algorithm_name=algorithm_name,
                        points=timestamp_dataset,
                        metric_name=self.metric_name,
                        params=parameters,
                    )
                    t1 = time.perf_counter()
                    per_timestamp_times.append(t1 - t0)
                    run_results.append((timestamp_dataset, result))
                alg_end = time.perf_counter()
                timing_results[algorithm_name] = {
                    "per_timestamp_time": per_timestamp_times,
                    "total_time": float(alg_end - alg_start),
                }
                # Print timing summary to console
                total_time = timing_results[algorithm_name]["total_time"]
                avg_time = (
                    sum(per_timestamp_times) / len(per_timestamp_times)
                    if per_timestamp_times
                    else 0.0
                )
                print(
                    f"[TIMING] {algorithm_name}: total_time={total_time:.6f}s, avg_per_timestamp={avg_time:.6f}s"
                )
                algorithm_run_results[algorithm_name] = run_results

            self.progress.emit("Saving output...")
            self._append_results_to_output(algorithm_run_results, timing_results)

            finished_payload = {
                "selected_algorithms": self.selected_algorithms,
                "algorithm_run_results": algorithm_run_results,
            }
            if timing_results:
                finished_payload["timing_results"] = timing_results

            self.finished.emit(finished_payload)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _append_results_to_output(
        self,
        algorithm_run_results: Dict[str, List[Tuple[np.ndarray, ClusteringResult]]],
        timing_results: Dict[str, Dict[str, object]] | None = None,
    ) -> None:
        """Write all algorithm run results to output.txt by timestamp (overwrites on each run)."""
        with self.output_file_path.open("w", encoding="utf-8") as output_file:
            for algorithm_name, run_results in algorithm_run_results.items():
                output_file.write(f"Algorithm: {algorithm_name}\n")

                for timestamp_index, (points, result) in enumerate(run_results, start=1):
                    output_file.write(f"Timestamp: {timestamp_index}\n")
                    output_file.write(f"Points: {format_points_for_output(points)}\n")
                    output_file.write(f"Labels: {result.labels.tolist()}\n")
                    output_file.write(f"Stats: {format_stats_for_output(result.stats)}\n\n")

                output_file.write("\n")
                # Write timing if available for this algorithm
                if timing_results and algorithm_name in timing_results:
                    timing = timing_results[algorithm_name]
                    per_ts = timing.get("per_timestamp_time", [])
                    total = float(timing.get("total_time", 0.0))
                    # Write per-timestamp timings in a readable format
                    output_file.write("Timing:\n")
                    for idx, t in enumerate(per_ts, start=1):
                        output_file.write(f"Time per timestamp {idx}: {t:.6f} sec\n")
                    output_file.write(f"Total time: {total:.6f} sec\n\n")



class VisualizationPanel(QWidget):
    """A single result panel with 3D scatter and statistics labels."""

    def __init__(self, algorithm_name: str, parent=None):
        super().__init__(parent)
        self.algorithm_name = algorithm_name

        root_layout = QVBoxLayout(self)

        title_label = QLabel(f"Algorithm: {algorithm_name}")
        title_label.setStyleSheet("font-weight: bold;")
        root_layout.addWidget(title_label)

        # Embedded matplotlib figure for 3D cluster visualization.
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        root_layout.addWidget(self.canvas)

        # Statistics block below the chart.
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)
        self.total_clusters_label = QLabel("0")
        self.target_cluster_size_label = QLabel("0")
        self.smallest_cluster_size_label = QLabel("0")
        self.outliers_label = QLabel("0")
        self.changed_cluster_points_label = QLabel("0")

        stats_layout.addRow("Total clusters:", self.total_clusters_label)
        stats_layout.addRow("Target cluster size:", self.target_cluster_size_label)
        stats_layout.addRow("Smallest cluster size:", self.smallest_cluster_size_label)
        stats_layout.addRow("Number of outliers:", self.outliers_label)
        stats_layout.addRow(
            "Points changed cluster:", self.changed_cluster_points_label
        )

        root_layout.addWidget(stats_group)

    def render(self, points: np.ndarray, labels: np.ndarray, stats: Dict[str, int]) -> None:
        """Draw cluster scatter and update numeric statistics."""
        self.ax.clear()

        unique_labels = sorted(set(labels.tolist()))
        color_map = {
            -1: "black",
        }

        # Assign deterministic colors from matplotlib tab20 palette.
        palette = list(plt_color_list())
        color_index = 0
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue
            color_map[cluster_id] = palette[color_index % len(palette)]
            color_index += 1

        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_points = points[cluster_mask]
            if cluster_points.size == 0:
                continue

            label_name = "Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
            self.ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                cluster_points[:, 2],
                c=color_map.get(cluster_id, "gray"),
                s=18,
                alpha=0.8,
                label=label_name,
            )

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(self.algorithm_name)
        self.ax.legend(loc="best")
        self.canvas.draw_idle()

        self.total_clusters_label.setText(str(stats["total_clusters"]))
        self.target_cluster_size_label.setText(str(stats["target_cluster_size"]))
        self.smallest_cluster_size_label.setText(str(stats["smallest_cluster_size"]))
        self.outliers_label.setText(str(stats["outliers"]))
        self.changed_cluster_points_label.setText(
            str(stats["changed_cluster_points"])
        )


def plt_color_list() -> List[str]:
    """Provide a compact list of distinct plotting colors."""
    return [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "gold",
    ]


class AlgorithmDropdown(QComboBox):
    """Checkable multi-select dropdown for algorithm selection."""

    def __init__(self, algorithm_options: List[str], parent=None):
        super().__init__(parent)
        self._algorithm_options = algorithm_options

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select algorithms")
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setModel(QStandardItemModel(self))

        self.view().pressed.connect(self._toggle_item_check_state)

        for algorithm_name in algorithm_options:
            item = QStandardItem(algorithm_name)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setData(Qt.Unchecked, Qt.CheckStateRole)
            self.model().appendRow(item)

        self._update_label()

    def selected_algorithms(self) -> List[str]:
        """Return all checked algorithms in menu order."""
        selected: List[str] = []
        for row in range(self.count()):
            item = self.model().item(row)
            if item is not None and item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def _toggle_item_check_state(self, index) -> None:
        """Toggle the clicked item's check state without closing the popup."""
        item = self.model().itemFromIndex(index)
        if item is None:
            return

        item.setCheckState(
            Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
        )
        self._update_label()

    def hidePopup(self) -> None:
        """Keep the dropdown open for multiple selections until the user closes it."""
        self._update_label()
        super().hidePopup()

    def _update_label(self) -> None:
        """Reflect the current checked state in the button label."""
        selected = self.selected_algorithms()
        if not selected:
            self.setCurrentText("Select algorithms")
            return

        if len(selected) == 1:
            self.setCurrentText(selected[0])
            return

        self.setCurrentText(f"{len(selected)} algorithms selected")

    def set_selected_algorithms(self, selected_names: List[str]) -> None:
        """Programmatically set the checked algorithms."""
        selected_set = set(selected_names)
        for row in range(self.count()):
            item = self.model().item(row)
            if item is not None:
                item.setCheckState(Qt.Checked if item.text() in selected_set else Qt.Unchecked)
        self._update_label()


class MainWindow(QMainWindow):
    """Main application window combining controls, execution, and results."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clustering Visualizer (PyQt5)")
        self.resize(1400, 900)

        self.data_loader = DataLoader()
        self.clustering_engine = ClusteringEngine()
        self.result_panels: List[VisualizationPanel] = []
        self.algorithm_config = self._load_algorithm_config()
        self.metrics_config = self._load_metrics_config()
        self.parameter_widgets: Dict[str, Dict[str, QWidget]] = {}
        self.parameter_groups: List[QGroupBox] = []
        self.algorithm_run_results: Dict[str, List[Tuple[np.ndarray, ClusteringResult]]] = {}
        self.output_file_path = Path(__file__).resolve().parent / "output.txt"
        self.output_file_path.touch(exist_ok=True)
        self.is_running = False
        self.worker_thread: QThread | None = None
        self.worker: ClusteringWorker | None = None
        self.animation_interval = self._load_animation_config()
        self.animation_timer: QTimer | None = None
        self.current_timestamp_index = 0
        self.num_timestamps = 0

        self._build_ui()

    def _load_algorithm_config(self) -> List[Dict[str, object]]:
        """Load algorithm definitions and parameters from config.json."""
        config_path = Path(__file__).resolve().parent / "config.json"

        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(
                self,
                "Configuration Error",
                f"Failed to load config.json: {exc}",
            )
            return []

        algorithms = config_data.get("algorithms", [])
        if not isinstance(algorithms, list):
            QMessageBox.critical(
                self,
                "Configuration Error",
                "config.json must contain an 'algorithms' list.",
            )
            return []

        valid_algorithms: List[Dict[str, object]] = []
        for algorithm in algorithms:
            if not isinstance(algorithm, dict):
                continue

            algorithm_name = algorithm.get("name")
            parameters = algorithm.get("parameters", [])
            if not isinstance(algorithm_name, str):
                continue
            if not isinstance(parameters, list):
                parameters = []

            valid_algorithms.append(
                {
                    "name": algorithm_name,
                    "parameters": parameters,
                }
            )

        return valid_algorithms

    def _load_metrics_config(self) -> List[str]:
        """Load distance metrics from config.json with fallback defaults."""
        default_metrics = [
            ClusteringEngine.METRIC_EUCLIDEAN,
            ClusteringEngine.METRIC_MANHATTAN,
            ClusteringEngine.METRIC_SPEED,
            ClusteringEngine.METRIC_HEADING,
            ClusteringEngine.METRIC_COMBINATION,
        ]

        config_path = Path(__file__).resolve().parent / "config.json"

        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            return default_metrics

        metrics = config_data.get("metrics", [])
        if not isinstance(metrics, list) or not metrics:
            return default_metrics

        valid_metrics: List[str] = []
        for metric in metrics:
            if isinstance(metric, str):
                valid_metrics.append(metric)

        return valid_metrics if valid_metrics else default_metrics

    def _load_animation_config(self) -> float:
        """Load animation interval from config.json with fallback default."""
        default_interval = 2.0
        config_path = Path(__file__).resolve().parent / "config.json"

        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            return default_interval

        animation_interval = config_data.get("animation_interval")
        if isinstance(animation_interval, (int, float)) and animation_interval > 0:
            return float(animation_interval)

        return default_interval

    def _build_ui(self) -> None:
        """Create and connect all top-level GUI sections."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setSpacing(12)

        root_layout.addWidget(self._build_controls_section())
        root_layout.addWidget(self._build_dataset_section())
        root_layout.addWidget(self._build_bottom_section())
        root_layout.addWidget(self._build_visualization_section(), stretch=1)

    def _build_controls_section(self) -> QWidget:
        """Build left/right controls row for algorithms and parameter inputs."""
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)

        controls_layout.addWidget(self._build_top_section(), stretch=1)
        controls_layout.addWidget(self._build_parameter_section(), stretch=1)

        return controls_widget

    def _build_top_section(self) -> QGroupBox:
        """Build top section with algorithm selection controls."""
        top_group = QGroupBox("Top Section: Algorithm Selection")
        top_layout = QGridLayout(top_group)

        algorithm_options = [
            str(algorithm["name"])
            for algorithm in self.algorithm_config
            if isinstance(algorithm.get("name"), str)
        ]

        self.algorithm_dropdown = AlgorithmDropdown(algorithm_options)
        self.algorithm_dropdown.model().itemChanged.connect(
            lambda _: self._refresh_parameter_inputs()
        )

        top_layout.addWidget(QLabel("Algorithms:"), 0, 0)
        top_layout.addWidget(self.algorithm_dropdown, 0, 1)

        # Distance metric selector (stored and passed into clustering layer).
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(self.metrics_config)
        top_layout.addWidget(QLabel("Distance metric:"), 1, 0)
        top_layout.addWidget(self.metric_combo, 1, 1)

        return top_group

    def _build_parameter_section(self) -> QGroupBox:
        """Build right-side section with algorithm-specific parameter fields."""
        parameter_group = QGroupBox("Algorithm-Specific Parameters")
        self.parameter_panel_layout = QVBoxLayout(parameter_group)

        self.no_parameters_label = QLabel("No additional parameters for selected algorithms")
        self.no_parameters_label.setAlignment(Qt.AlignCenter)
        self.parameter_panel_layout.addWidget(self.no_parameters_label)
        self.parameter_panel_layout.addStretch(1)

        self._refresh_parameter_inputs()

        return parameter_group

    def _build_dataset_section(self) -> QGroupBox:
        """Build dataset file picker section below algorithm/parameter row."""
        dataset_group = QGroupBox("Dataset File")
        dataset_layout = QVBoxLayout(dataset_group)

        # File picker row.
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("Select dataset file...")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._select_file)

        file_picker_layout = QHBoxLayout()
        file_picker_layout.addWidget(self.file_path_input)
        file_picker_layout.addWidget(browse_button)
        dataset_layout.addLayout(file_picker_layout)

        return dataset_group

    def _build_bottom_section(self) -> QWidget:
        """Build bottom section with run action and animation controls."""
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("RUN")
        self.run_button.setMinimumHeight(36)
        self.run_button.clicked.connect(self._run_clustering)

        self.start_animation_button = QPushButton("Start Animation")
        self.start_animation_button.setMinimumHeight(36)
        self.start_animation_button.clicked.connect(self._start_animation)
        self.start_animation_button.setEnabled(False)

        self.stop_animation_button = QPushButton("Stop Animation")
        self.stop_animation_button.setMinimumHeight(36)
        self.stop_animation_button.clicked.connect(self._stop_animation)
        self.stop_animation_button.setEnabled(False)

        button_row.addStretch(1)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.start_animation_button)
        button_row.addWidget(self.stop_animation_button)
        bottom_layout.addLayout(button_row)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        bottom_layout.addWidget(self.status_label)

        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: #c62828; font-weight: 600;")
        self.error_label.setVisible(False)
        bottom_layout.addWidget(self.error_label)

        return bottom_widget

    def _build_visualization_section(self) -> QGroupBox:
        """Build visualization section with dynamic layout for result canvases."""
        visualization_group = QGroupBox("Visualization")
        visualization_layout = QVBoxLayout(visualization_group)

        # Scrollable visualization container to host 2-3 result panels.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QHBoxLayout(self.results_container)
        self.results_layout.setSpacing(12)
        self.scroll_area.setWidget(self.results_container)

        visualization_layout.addWidget(self.scroll_area)

        return visualization_group

    def _selected_algorithms(self) -> List[str]:
        """Return the selected algorithms from the dropdown menu."""
        return self.algorithm_dropdown.selected_algorithms()

    def _refresh_parameter_inputs(self) -> None:
        """Regenerate parameter fields for the currently selected algorithms."""
        selected_algorithms = set(self._selected_algorithms())

        for group in self.parameter_groups:
            self.parameter_panel_layout.removeWidget(group)
            group.deleteLater()

        self.parameter_groups.clear()
        self.parameter_widgets.clear()

        has_parameters = False
        algorithm_items = {
            str(item.get("name")): item
            for item in self.algorithm_config
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }

        for algorithm_name in self._selected_algorithms():
            algorithm_item = algorithm_items.get(algorithm_name)
            if algorithm_item is None:
                continue

            self.parameter_widgets[algorithm_name] = {}

            parameter_items = algorithm_item.get("parameters", [])
            if not isinstance(parameter_items, list) or not parameter_items:
                continue

            parameter_group = QGroupBox(f"{algorithm_name} Parameters")
            form_layout = QFormLayout(parameter_group)

            for parameter in parameter_items:
                if not isinstance(parameter, dict):
                    continue

                param_name = parameter.get("name")
                param_type = parameter.get("type")
                default_value = parameter.get("default")
                if not isinstance(param_name, str) or not isinstance(param_type, str):
                    continue

                widget = self._create_parameter_widget(
                    parameter_type=param_type,
                    default_value=default_value,
                )
                if widget is None:
                    continue

                self.parameter_widgets[algorithm_name][param_name] = widget
                form_layout.addRow(f"{param_name}:", widget)
                has_parameters = True

            if form_layout.rowCount() == 0:
                parameter_group.deleteLater()
                continue

            self.parameter_panel_layout.insertWidget(
                self.parameter_panel_layout.count() - 2,
                parameter_group,
            )
            self.parameter_groups.append(parameter_group)

        if selected_algorithms:
            self.no_parameters_label.setVisible(not has_parameters)
        else:
            self.no_parameters_label.setVisible(True)

    @staticmethod
    def _create_parameter_widget(parameter_type: str, default_value: object) -> QWidget | None:
        """Build an input widget matching the configured parameter type."""
        if parameter_type == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(6)
            widget.setRange(-1000000.0, 1000000.0)
            if isinstance(default_value, (int, float)):
                widget.setValue(float(default_value))
            return widget

        if parameter_type == "int":
            widget = QSpinBox()
            widget.setRange(-1000000, 1000000)
            if isinstance(default_value, int):
                widget.setValue(default_value)
            elif isinstance(default_value, float):
                widget.setValue(int(default_value))
            return widget

        if parameter_type == "string":
            widget = QLineEdit()
            widget.setText("" if default_value is None else str(default_value))
            return widget

        return None

    @staticmethod
    def _widget_value_or_default(widget: QWidget | None, default_value: object) -> object:
        """Read a parameter widget value with config default fallback."""
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QLineEdit):
            return widget.text()
        return default_value

    @staticmethod
    def _coerce_parameter_value(parameter_type: str, value: object) -> int | float | str:
        """Coerce dynamic parameter values to their configured type."""
        if parameter_type == "int":
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        if parameter_type == "float":
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        if parameter_type == "string":
            return "" if value is None else str(value)

        return "" if value is None else str(value)

    def _collect_selected_algorithm_parameters(
        self,
        selected_algorithms: List[str],
    ) -> Dict[str, Dict[str, int | float | str]]:
        """Build parameter values for selected algorithms from dynamic widgets."""
        algorithm_items = {
            str(item.get("name")): item
            for item in self.algorithm_config
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }

        collected: Dict[str, Dict[str, int | float | str]] = {}

        for algorithm_name in selected_algorithms:
            algorithm_item = algorithm_items.get(algorithm_name)
            if algorithm_item is None:
                collected[algorithm_name] = {}
                continue

            parameter_items = algorithm_item.get("parameters", [])
            if not isinstance(parameter_items, list):
                collected[algorithm_name] = {}
                continue

            algorithm_widget_map = self.parameter_widgets.get(algorithm_name, {})
            algorithm_values: Dict[str, int | float | str] = {}

            for parameter in parameter_items:
                if not isinstance(parameter, dict):
                    continue

                parameter_name = parameter.get("name")
                parameter_type = parameter.get("type")
                default_value = parameter.get("default")
                if not isinstance(parameter_name, str) or not isinstance(parameter_type, str):
                    continue

                widget = algorithm_widget_map.get(parameter_name)
                raw_value = self._widget_value_or_default(widget, default_value)
                algorithm_values[parameter_name] = self._coerce_parameter_value(
                    parameter_type=parameter_type,
                    value=raw_value,
                )

            collected[algorithm_name] = algorithm_values

        return collected

    def _select_file(self) -> None:
        """Open file chooser and store selected path in text field."""
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select dataset file",
            "",
            "Text files (*.txt *.csv);;All files (*.*)",
        )
        if selected_path:
            self.file_path_input.setText(selected_path)

    def _run_clustering(self) -> None:
        """Validate input and dispatch clustering execution to a worker thread."""
        if self.is_running:
            return

        self.is_running = True
        self.run_button.setEnabled(False)
        self.status_label.setText("Running...")
        self.error_label.clear()
        self.error_label.setVisible(False)

        selected_algorithms = self._selected_algorithms()
        if len(selected_algorithms) < 1:
            self.status_label.setText("Ready")
            self.error_label.setText("Please select at least 1 algorithm")
            self.error_label.setVisible(True)
            QMessageBox.warning(
                self,
                "Selection Error",
                "Please select at least 1 algorithm",
            )
            self._reset_run_state()
            return

        if len(selected_algorithms) > 3:
            self.status_label.setText("Ready")
            self.error_label.setText("Select no more than 3 algorithms")
            self.error_label.setVisible(True)
            QMessageBox.warning(
                self,
                "Selection Error",
                "Please select no more than 3 algorithms.",
            )
            self._reset_run_state()
            return

        file_path = self.file_path_input.text().strip()
        if not file_path:
            self.status_label.setText("Ready")
            self.error_label.setText("File not selected")
            self.error_label.setVisible(True)
            QMessageBox.warning(
                self,
                "Input Error",
                "Please choose a dataset file.",
            )
            self._reset_run_state()
            return

        self._rebuild_result_panels(selected_algorithms)

        metric_name = self.metric_combo.currentText()
        algorithm_parameters = self._collect_selected_algorithm_parameters(
            selected_algorithms=selected_algorithms,
        )

        self.worker_thread = QThread(self)
        self.worker = ClusteringWorker(
            file_path=file_path,
            selected_algorithms=selected_algorithms,
            metric_name=metric_name,
            algorithm_parameters=algorithm_parameters,
            output_file_path=str(self.output_file_path),
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def _on_worker_finished(self, payload: object) -> None:
        """Render results on UI thread after successful background execution."""
        if not isinstance(payload, dict):
            self._on_worker_failed("Invalid worker payload.")
            return

        selected_algorithms = payload.get("selected_algorithms", [])
        algorithm_run_results = payload.get("algorithm_run_results", {})
        if not isinstance(selected_algorithms, list) or not isinstance(
            algorithm_run_results, dict
        ):
            self._on_worker_failed("Invalid worker payload.")
            return

        self.algorithm_run_results = algorithm_run_results

        # Determine number of timestamps from first algorithm's results
        first_algorithm_results = next(
            (v for v in algorithm_run_results.values() if v), None
        )
        self.num_timestamps = len(first_algorithm_results) if first_algorithm_results else 0
        self.current_timestamp_index = 0

        # Enable animation controls if we have multiple timestamps
        has_multiple_timestamps = self.num_timestamps > 1
        self.start_animation_button.setEnabled(has_multiple_timestamps)

        # Render the first timestamp initially
        self._render_current_timestamp()

        self.status_label.setText("Completed")
        self._reset_run_state()

    def _on_worker_progress(self, status_message: str) -> None:
        """Update status label with worker progress messages."""
        self.status_label.setText(status_message)

    def _render_current_timestamp(self) -> None:
        """Render all result panels for the current timestamp."""
        for index, algorithm_name in enumerate(
            self.algorithm_dropdown.selected_algorithms()
        ):
            if index >= len(self.result_panels):
                break

            run_results = self.algorithm_run_results.get(algorithm_name, [])
            if not isinstance(run_results, list) or self.current_timestamp_index >= len(
                run_results
            ):
                continue

            render_points, render_result = run_results[self.current_timestamp_index]
            self.result_panels[index].render(
                points=render_points,
                labels=render_result.labels,
                stats=render_result.stats,
            )

        self.status_label.setText(
            f"Timestamp {self.current_timestamp_index + 1}/{self.num_timestamps}"
        )

    def _start_animation(self) -> None:
        """Start animating through timestamps."""
        if self.num_timestamps <= 1 or self.animation_timer is not None:
            return

        self.run_button.setEnabled(False)
        self.start_animation_button.setEnabled(False)
        self.stop_animation_button.setEnabled(True)

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._animate_next_timestamp)
        self.animation_timer.start(int(self.animation_interval * 1000))

    def _stop_animation(self) -> None:
        """Stop animation and clean up timer."""
        if self.animation_timer is None:
            return

        self.animation_timer.stop()
        self.animation_timer.deleteLater()
        self.animation_timer = None

        self.run_button.setEnabled(True)
        self.start_animation_button.setEnabled(True)
        self.stop_animation_button.setEnabled(False)

    def _animate_next_timestamp(self) -> None:
        """Update visualization to next timestamp and loop back to start."""
        self.current_timestamp_index = (self.current_timestamp_index + 1) % self.num_timestamps
        self._render_current_timestamp()

    def _on_worker_failed(self, error_message: str) -> None:
        """Handle background execution failures on UI thread."""
        self.status_label.setText("Ready")
        self.error_label.setText(error_message)
        self.error_label.setVisible(True)
        QMessageBox.critical(self, "Clustering Error", error_message)
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        """Reset execution flag and re-enable Run button."""
        self.is_running = False
        self.run_button.setEnabled(True)
        self.worker = None
        self.worker_thread = None

    def _rebuild_result_panels(self, selected_algorithms: List[str]) -> None:
        """Recreate visualization widgets to match selected algorithm count."""
        while self.results_layout.count() > 0:
            layout_item = self.results_layout.takeAt(0)
            widget = layout_item.widget()
            if widget is not None:
                widget.deleteLater()

        self.result_panels.clear()

        for algorithm_name in selected_algorithms:
            panel = VisualizationPanel(algorithm_name)
            self.results_layout.addWidget(panel)
            self.result_panels.append(panel)

        self.results_layout.addStretch(1)
