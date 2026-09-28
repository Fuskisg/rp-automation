from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from core.config import SettingsStore
from core.input import InputAdapter
from core.logger import create_logger
from modules.antiafk import AntiAfkTask, PATTERNS
from modules.macro import MacroTask, parse_script
from modules.vision_task import VisionMonitorTask
from vision.matcher import TemplateMatcher


STYLE = """
QWidget { background:#111722; color:#e9eef7; font-size:13px; }
QMainWindow { background:#0b1018; }
QFrame#side { background:#0d1521; border-right:1px solid #29364b; }
QFrame#card { background:#151f2e; border:1px solid #2a3b54; border-radius:12px; }
QLabel#brand { color:#78a9ff; font-size:20px; font-weight:bold; }
QLabel#title { font-size:24px; font-weight:bold; }
QLabel#muted { color:#8c9bb2; }
QPushButton { background:#1b2a40; border:1px solid #304764; border-radius:8px; padding:9px 12px; }
QPushButton:hover { background:#284264; }
QPushButton#primary { background:#3976d6; border-color:#6a9fff; font-weight:bold; }
QPushButton#danger { background:#7e3244; border-color:#b34a62; }
QPushButton#nav { text-align:left; background:transparent; border:0; padding:11px 14px; color:#aebbd0; }
QPushButton#nav:checked,QPushButton#nav:hover { background:#1b2a40; color:#fff; }
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QPlainTextEdit { background:#0d141e; border:1px solid #304057; border-radius:7px; padding:7px; }
QCheckBox { spacing:8px; }
QProgressBar { background:#0d141e; border:1px solid #304057; border-radius:6px; text-align:center; }
QProgressBar::chunk { background:#3976d6; border-radius:5px; }
"""


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "templates"
ASSET_GROUPS = {
    "cook": ("🍳 Готовка", "Ингредиенты, инструменты и кнопка запуска рецепта"),
    "cow": ("🐄 Ферма: коровы", "Подсказки направления и автоматическое действие"),
    "gym": ("🏋 Качалка", "Старт, успех и ошибка упражнения"),
    "heal": ("🏥 Медицина", "Символы и предметы лечебной мини‑игры"),
    "relogin": ("🔄 Переподключение", "Состояния подключения и повторного входа"),
    "shveika": ("🧵 Швейка", "Последовательность из 20 точек и состояния мини‑игры"),
    "spin": ("🎰 Казино", "Элементы колеса и кнопка запуска"),
    "stroyka": ("🏗 Стройка", "Картинки строительной мини‑игры"),
    "tokar": ("🔧 Токарь", "Шаблоны инструмента и рабочего элемента"),
}
PATTERN_LABELS = {
    "gentle": ("Мягкое движение", "A → D → W → S"),
    "walk": ("Прогулка вперёд/назад", "W → S"),
    "turn": ("Повороты", "A → D"),
}
SCRIPT_PRESETS = {
    "Проверка клавиш": "# безопасный пример\ntap f\nwait 0.5\ntap space",
    "Мягкое движение": "hold a 0.45\nhold d 0.45\nhold w 0.35\nhold s 0.35",
    "Открыть и подтвердить": "tap e\nwait 0.7\ntap enter",
}


class VisionScanThread(QtCore.QThread):
    result = QtCore.Signal(object, object, str)

    def __init__(self, image_path: Path, threshold: float):
        super().__init__()
        self.image_path = image_path
        self.threshold = threshold

    def run(self) -> None:
        try:
            template = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
            if template is None:
                raise ValueError(f"Не получилось открыть шаблон: {self.image_path.name}")
            with mss.mss() as capture:
                monitor = capture.monitors[1]
                bgra = np.asarray(capture.grab(monitor))
            frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            matcher = TemplateMatcher(self.threshold)
            matcher.add(self.image_path.stem, template)
            match = matcher.match_best(frame)
            preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.result.emit(preview, match, "")
        except Exception as exc:
            self.result.emit(None, None, str(exc))


class Card(QtWidgets.QFrame):
    def __init__(self, title: str, text: str = "") -> None:
        super().__init__(objectName="card")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(18, 16, 18, 16)
        heading = QtWidgets.QLabel(title)
        heading.setStyleSheet("font-size:16px;font-weight:bold")
        self.layout.addWidget(heading)
        if text:
            note = QtWidgets.QLabel(text, objectName="muted")
            note.setWordWrap(True)
            self.layout.addWidget(note)


class MainWindow(QtWidgets.QMainWindow):
    hotkey_signal = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RP Automation — самостоятельный помощник")
        self.resize(1080, 700)
        self.store = SettingsStore()
        self.settings = self.store.load()
        self.log = create_logger()
        self.input = InputAdapter(self.settings.dry_run, self.settings.key_delay, self.settings.click_pause)
        self.task = None
        self.vision_thread = None
        self._keyboard = None
        self._hotkey_handle = None
        self.started = None
        self._build()
        self.hotkey_signal.connect(self.stop)
        self.register_global_hotkey()
        self.write("Готово. Безопасный режим включён по умолчанию.")
        QtCore.QTimer.singleShot(400, self.refresh_status)

    def _build(self) -> None:
        root = QtWidgets.QWidget()
        shell = QtWidgets.QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(root)
        side = QtWidgets.QFrame(objectName="side")
        side.setFixedWidth(220)
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(18, 24, 18, 18)
        sl.addWidget(QtWidgets.QLabel("RP\nAUTOMATION", objectName="brand"))
        sl.addWidget(QtWidgets.QLabel("Собственная модульная панель", objectName="muted"))
        sl.addSpacing(22)
        self.pages = QtWidgets.QStackedWidget()
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for label, index in (("⌂  Обзор", 0), ("◌  Anti‑AFK", 1), ("☷  Сценарии", 2), ("◈  Мини‑игры", 3), ("⚙  Настройки", 4)):
            button = QtWidgets.QPushButton(label, objectName="nav")
            button.setCheckable(True)
            button.clicked.connect(lambda checked, i=index: self.pages.setCurrentIndex(i))
            group.addButton(button)
            sl.addWidget(button)
            if index == 0:
                button.setChecked(True)
        sl.addStretch()
        self.side_status = QtWidgets.QLabel("● Остановлено", objectName="muted")
        sl.addWidget(self.side_status)
        shell.addWidget(side)
        self.pages.addWidget(self.overview())
        self.pages.addWidget(self.afk_page())
        self.pages.addWidget(self.script_page())
        self.pages.addWidget(self.vision_page())
        self.pages.addWidget(self.settings_page())
        shell.addWidget(self.pages, 1)

    def page(self, title: str, subtitle: str) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(30, 26, 30, 26)
        layout.addWidget(QtWidgets.QLabel(title, objectName="title"))
        layout.addWidget(QtWidgets.QLabel(subtitle, objectName="muted"))
        layout.addSpacing(16)
        return widget, layout

    def overview(self) -> QtWidgets.QWidget:
        page, layout = self.page("Панель управления", "Быстрый запуск, состояние модулей и единая остановка.")
        row = QtWidgets.QHBoxLayout()
        self.state_value = QtWidgets.QLabel("Остановлено")
        self.mode_value = QtWidgets.QLabel("Проверка без ввода" if self.settings.dry_run else "Реальный ввод")
        self.time_value = QtWidgets.QLabel("—")
        for title, value in (("Состояние", self.state_value), ("Режим", self.mode_value), ("Сессия", self.time_value)):
            card = Card(title)
            value.setStyleSheet("font-size:20px;font-weight:bold;color:#78a9ff")
            card.layout.addWidget(value)
            row.addWidget(card)
        layout.addLayout(row)
        quick = Card("Быстрые действия", "Запускай только один модуль за раз. F8 всегда останавливает активный модуль.")
        quick_row = QtWidgets.QHBoxLayout()
        mini_games = QtWidgets.QPushButton("Открыть мини‑игры")
        anti_afk = QtWidgets.QPushButton("Запустить Anti‑AFK", objectName="primary")
        stop_all = QtWidgets.QPushButton("Остановить всё", objectName="danger")
        mini_games.clicked.connect(lambda: self.pages.setCurrentIndex(3))
        anti_afk.clicked.connect(self.start_afk)
        stop_all.clicked.connect(self.stop)
        quick_row.addWidget(mini_games)
        quick_row.addWidget(anti_afk)
        quick_row.addWidget(stop_all)
        quick.layout.addLayout(quick_row)
        layout.addWidget(quick)
        guide = Card("Быстрый старт", "1) Оставь безопасный режим.  2) Проверь шаблон на вкладке «Мини‑игры».  3) Только после этого включай реальный ввод в настройках.")
        layout.addWidget(guide)
        card = Card("Журнал", "Подробный файл: logs/automation.log")
        self.output = QtWidgets.QPlainTextEdit(readOnly=True)
        card.layout.addWidget(self.output)
        layout.addWidget(card, 1)
        return page

    def afk_page(self) -> QtWidgets.QWidget:
        page, layout = self.page("Anti‑AFK", "Циклические движения с общей кнопкой остановки.")
        card = Card("Параметры", "Сначала используй безопасный режим и проверь журнал.")
        form = QtWidgets.QFormLayout()
        self.interval = QtWidgets.QDoubleSpinBox(minimum=5, maximum=3600, value=self.settings.anti_afk_interval, suffix=" с")
        self.pattern = QtWidgets.QComboBox()
        for key, (label, detail) in PATTERN_LABELS.items():
            self.pattern.addItem(f"{label} · {detail}", key)
        pattern_keys = list(PATTERN_LABELS)
        self.pattern.setCurrentIndex(pattern_keys.index(self.settings.anti_afk_pattern) if self.settings.anti_afk_pattern in pattern_keys else 0)
        form.addRow("Пауза между циклами", self.interval)
        form.addRow("Рисунок движения", self.pattern)
        card.layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        start = QtWidgets.QPushButton("Запустить", objectName="primary")
        stop = QtWidgets.QPushButton("Остановить", objectName="danger")
        start.clicked.connect(self.start_afk)
        stop.clicked.connect(self.stop)
        buttons.addWidget(start)
        buttons.addWidget(stop)
        card.layout.addLayout(buttons)
        layout.addWidget(card)
        layout.addWidget(QtWidgets.QLabel("В безопасном режиме нажатия не отправляются — в журнал попадает только ход сценария.", objectName="muted"))
        layout.addStretch()
        return page

    def script_page(self) -> QtWidgets.QWidget:
        page, layout = self.page("Сценарии", "Свои последовательности действий без правки исходников.")
        card = Card("Редактор", "Команды: tap KEY, hold KEY SEC, wait SEC. Можно начать с готового пресета.")
        self.script = QtWidgets.QPlainTextEdit("# пример\ntap f\nwait 0.5\nhold w 1.2\ntap space")
        card.layout.addWidget(self.script, 1)
        line = QtWidgets.QHBoxLayout()
        self.preset = QtWidgets.QComboBox()
        self.preset.addItem("Выбрать пресет…", "")
        for name, script in SCRIPT_PRESETS.items():
            self.preset.addItem(name, script)
        load_preset = QtWidgets.QPushButton("Загрузить")
        load_preset.clicked.connect(lambda: self.script.setPlainText(self.preset.currentData() or self.script.toPlainText()))
        line.addWidget(self.preset)
        line.addWidget(load_preset)
        line.addSpacing(12)
        self.repeat = QtWidgets.QSpinBox(minimum=1, maximum=999, value=1)
        line.addWidget(QtWidgets.QLabel("Повторы:"))
        line.addWidget(self.repeat)
        line.addStretch()
        run = QtWidgets.QPushButton("Запустить", objectName="primary")
        stop = QtWidgets.QPushButton("Остановить", objectName="danger")
        run.clicked.connect(self.start_script)
        stop.clicked.connect(self.stop)
        line.addWidget(run)
        line.addWidget(stop)
        card.layout.addLayout(line)
        layout.addWidget(card, 1)
        return page

    def settings_page(self) -> QtWidgets.QWidget:
        page, layout = self.page("Настройки", "Безопасность, горячая клавиша и задержки ввода.")
        card = Card("Режим ввода")
        form = QtWidgets.QFormLayout()
        self.dry = QtWidgets.QCheckBox("Только журналировать действия")
        self.dry.setChecked(self.settings.dry_run)
        self.hotkey_enabled = QtWidgets.QCheckBox("Включить глобальную клавишу остановки")
        self.hotkey_enabled.setChecked(bool(getattr(self.settings, "hotkey_enabled", False)))
        self.hotkey = QtWidgets.QLineEdit(getattr(self.settings, "global_hotkey", "f8"))
        self.hotkey.setMaximumWidth(160)
        self.delay = QtWidgets.QDoubleSpinBox(minimum=0, maximum=2, singleStep=0.01, value=self.settings.key_delay, suffix=" с")
        self.pause = QtWidgets.QDoubleSpinBox(minimum=0, maximum=2, singleStep=0.01, value=self.settings.click_pause, suffix=" с")
        form.addRow(self.dry)
        form.addRow(self.hotkey_enabled)
        form.addRow("Клавиша остановки", self.hotkey)
        form.addRow("Задержка клавиш", self.delay)
        form.addRow("Пауза клика", self.pause)
        card.layout.addLayout(form)
        save = QtWidgets.QPushButton("Сохранить", objectName="primary")
        save.clicked.connect(self.save)
        card.layout.addWidget(save)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def vision_page(self) -> QtWidgets.QWidget:
        page, layout = self.page("Мини‑игры", "Выбери понятную категорию, проверь картинку и при необходимости запусти мониторинг.")
        card = Card("Шаблон мини‑игры", "Категории уже собраны по назначению. Сначала нажми «Проверить экран», затем запускай мониторинг.")
        form = QtWidgets.QFormLayout()
        self.asset_group = QtWidgets.QComboBox()
        self.asset_group.setMinimumWidth(270)
        self.asset_image = QtWidgets.QComboBox()
        self.asset_image.setMinimumWidth(270)
        self.vision_threshold = QtWidgets.QDoubleSpinBox(minimum=0.50, maximum=0.99, singleStep=0.01, value=0.88)
        self.vision_threshold.setSuffix(" порог")
        self.vision_interval = QtWidgets.QDoubleSpinBox(minimum=0.05, maximum=5.0, singleStep=0.05, value=0.20, suffix=" с")
        self.vision_action = QtWidgets.QComboBox()
        self.vision_action.addItem("Только журнал", "none")
        self.vision_action.addItem("Нажать клавишу", "key")
        self.vision_action.addItem("Кликнуть по центру", "click")
        self.vision_key = QtWidgets.QLineEdit()
        self.vision_key.setPlaceholderText("например: e или space")
        self.vision_hint = QtWidgets.QLabel(objectName="muted")
        self.vision_hint.setWordWrap(True)
        form.addRow("Категория", self.asset_group)
        form.addRow("Картинка", self.asset_image)
        form.addRow("Точность", self.vision_threshold)
        form.addRow("Проверять каждые", self.vision_interval)
        form.addRow("Действие", self.vision_action)
        form.addRow("Клавиша", self.vision_key)
        card.layout.addLayout(form)
        actions = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QPushButton("Обновить список")
        scan = QtWidgets.QPushButton("Проверить экран", objectName="primary")
        start = QtWidgets.QPushButton("Запустить мониторинг", objectName="primary")
        stop = QtWidgets.QPushButton("Остановить", objectName="danger")
        refresh.clicked.connect(self.refresh_asset_list)
        self.asset_group.currentTextChanged.connect(self.refresh_asset_images)
        scan.clicked.connect(self.scan_screen)
        start.clicked.connect(self.start_vision_monitor)
        stop.clicked.connect(self.stop_vision_monitor)
        actions.addWidget(refresh)
        actions.addWidget(scan)
        actions.addWidget(start)
        actions.addWidget(stop)
        card.layout.addLayout(actions)
        self.vision_status = QtWidgets.QLabel("Шаблоны ещё не проверялись", objectName="muted")
        self.vision_status.setWordWrap(True)
        card.layout.addWidget(self.vision_status)
        card.layout.addWidget(self.vision_hint)
        layout.addWidget(card)
        self.refresh_asset_list()
        note = QtWidgets.QLabel(
            "В безопасном режиме действия только записываются в журнал. Для реального ввода сначала проверь совпадение, затем отключи безопасный режим в настройках.",
            objectName="muted",
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        return page

    def start_vision_monitor(self) -> None:
        group_name = self.asset_group.currentData()
        if not group_name or not self.asset_image.currentText():
            self.vision_status.setText("Сначала выбери шаблон.")
            return
        self.stop()
        path = ASSET_DIR / group_name / self.asset_image.currentText()
        self.vision_thread = VisionMonitorTask(
            self.input,
            path,
            self.vision_threshold.value(),
            self.vision_interval.value(),
            self.vision_action.currentData(),
            self.vision_key.text(),
            self.write,
        )
        self.vision_thread.start()
        self.task = self.vision_thread
        self.set_running("Распознавание")
        self.vision_status.setText("Мониторинг запущен. Смотри журнал ниже на вкладке «Обзор».")

    def stop_vision_monitor(self) -> None:
        if self.vision_thread:
            self.vision_thread.stop()
            self.vision_thread = None
        if self.task and self.task.name == "Распознавание":
            self.task = None
            self.set_running(None)

    def refresh_asset_list(self) -> None:
        if not hasattr(self, "asset_group"):
            return
        current_group = self.asset_group.currentData()
        self.asset_group.blockSignals(True)
        self.asset_group.clear()
        groups = sorted(path.name for path in ASSET_DIR.iterdir() if path.is_dir()) if ASSET_DIR.exists() else []
        for group in groups:
            label, _ = ASSET_GROUPS.get(group, (group.title(), ""))
            self.asset_group.addItem(label, group)
        if current_group in groups:
            self.asset_group.setCurrentIndex(groups.index(current_group))
        self.asset_group.blockSignals(False)
        self.refresh_asset_images()

    def refresh_asset_images(self) -> None:
        if not hasattr(self, "asset_image"):
            return
        self.asset_image.clear()
        group_name = self.asset_group.currentData()
        group = ASSET_DIR / group_name if group_name else Path()
        if group.exists():
            self.asset_image.addItems(sorted(path.name for path in group.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}))
        self.vision_hint.setText(ASSET_GROUPS.get(group_name, ("", ""))[1])

    def scan_screen(self) -> None:
        group_name = self.asset_group.currentData()
        if not group_name or not self.asset_image.currentText():
            self.vision_status.setText("В assets/templates пока нет шаблонов. Добавь картинки и нажми «Обновить список».")
            return
        if getattr(self, "vision_thread", None) and self.vision_thread.isRunning():
            return
        path = ASSET_DIR / group_name / self.asset_image.currentText()
        self.vision_status.setText(f"Сканирую экран по шаблону {path.name}…")
        self.vision_thread = VisionScanThread(path, self.vision_threshold.value())
        self.vision_thread.result.connect(self.on_vision_result)
        self.vision_thread.start()

    def on_vision_result(self, _frame, match, error: str) -> None:
        if error:
            self.vision_status.setText(f"Ошибка распознавания: {error}")
            self.write(f"Распознавание: ошибка — {error}")
            return
        if match is None:
            self.vision_status.setText("Совпадение не найдено. Попробуй другой порог или сделай шаблон под своё разрешение.")
            self.write("Распознавание: совпадение не найдено")
            return
        x, y = match.center
        self.vision_status.setText(f"Найдено: {match.label}, точность {match.score:.3f}, центр экрана ({x}, {y})")
        self.write(f"Распознавание: {match.label}, точность {match.score:.3f}, координаты ({x}, {y})")

    def write(self, text: str) -> None:
        message = f"[{datetime.now():%H:%M:%S}] {text}"
        self.log.info(text)
        if hasattr(self, "output"):
            self.output.appendPlainText(message)

    def set_running(self, name: str | None) -> None:
        active = bool(name)
        self.side_status.setText(f"● {'Работает: ' + name if active else 'Остановлено'}")
        self.state_value.setText("Работает" if active else "Остановлено")
        if active:
            self.started = time.monotonic()
        else:
            self.started = None

    def refresh_status(self) -> None:
        if self.task and not self.task.is_alive():
            self.task = None
            self.set_running(None)
        if self.started is not None:
            seconds = int(time.monotonic() - self.started)
            self.time_value.setText(f"{seconds // 60:02d}:{seconds % 60:02d}")
        QtCore.QTimer.singleShot(500, self.refresh_status)

    def start_afk(self) -> None:
        self.stop()
        self.task = AntiAfkTask(self.input, self.interval.value(), self.pattern.currentData(), self.write)
        self.task.start()
        self.set_running("Anti‑AFK")

    def start_script(self) -> None:
        try:
            steps = parse_script(self.script.toPlainText())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Ошибка сценария", str(exc))
            return
        if not steps:
            QtWidgets.QMessageBox.warning(self, "Пустой сценарий", "Добавь хотя бы одну команду.")
            return
        self.stop()
        self.task = MacroTask(self.input, steps, self.repeat.value(), self.write)
        self.task.start()
        self.set_running("Сценарий")

    def stop(self) -> None:
        if self.task:
            self.task.stop()
            self.task = None
        self.set_running(None)

    def save(self) -> None:
        self.settings.dry_run = self.dry.isChecked()
        self.settings.key_delay = self.delay.value()
        self.settings.click_pause = self.pause.value()
        self.settings.anti_afk_interval = self.interval.value()
        self.settings.anti_afk_pattern = self.pattern.currentData()
        self.settings.global_hotkey = self.hotkey.text().strip() or "f8"
        self.settings.hotkey_enabled = self.hotkey_enabled.isChecked()
        self.store.save(self.settings)
        self.input.dry_run = self.settings.dry_run
        self.input.key_delay = self.settings.key_delay
        self.input.click_pause = self.settings.click_pause
        self.mode_value.setText("Проверка без ввода" if self.settings.dry_run else "Реальный ввод")
        self.register_global_hotkey()
        self.write("Настройки сохранены")

    def register_global_hotkey(self) -> None:
        if self._keyboard is not None and self._hotkey_handle is not None:
            try:
                self._keyboard.remove_hotkey(self._hotkey_handle)
            except Exception:
                pass
            self._hotkey_handle = None
        if not getattr(self.settings, "hotkey_enabled", False):
            return
        try:
            import keyboard

            self._keyboard = keyboard
            self._hotkey_handle = keyboard.add_hotkey(
                getattr(self.settings, "global_hotkey", "f8"),
                lambda: self.hotkey_signal.emit(),
            )
            self.write(f"Глобальная остановка: {self.settings.global_hotkey}")
        except Exception as exc:
            self._keyboard = None
            self.write(f"Не удалось включить глобальную клавишу: {exc}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop()
        self.stop_vision_monitor()
        if self._keyboard is not None and self._hotkey_handle is not None:
            try:
                self._keyboard.remove_hotkey(self._hotkey_handle)
            except Exception:
                pass
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
