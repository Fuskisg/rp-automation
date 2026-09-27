from __future__ import annotations

import sys
import threading
import time
from datetime import datetime

from PySide6 import QtCore, QtWidgets


STYLE = """
QWidget { background:#111722; color:#e8eef7; font-size:13px; }
QMainWindow { background:#0b1018; }
QFrame#card { background:#151f2e; border:1px solid #2a3b54; border-radius:12px; }
QLabel#title { font-size:24px; font-weight:bold; }
QLabel#muted { color:#8c9bb2; }
QPushButton { background:#1b2a40; border:1px solid #304764; border-radius:8px; padding:9px 12px; }
QPushButton:hover { background:#284264; }
QPushButton#primary { background:#3976d6; border-color:#6a9fff; font-weight:bold; }
QPushButton#danger { background:#7e3244; border-color:#b34a62; }
QPlainTextEdit,QSpinBox { background:#0d141e; border:1px solid #304057; border-radius:7px; padding:7px; }
"""


class Worker(threading.Thread):
    def __init__(self, script: list[tuple[str, str, float]], dry_run: bool, log):
        super().__init__(daemon=True)
        self.script, self.dry_run, self.log = script, dry_run, log
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            for kind, value, seconds in self.script:
                if self.stop_event.is_set():
                    self.log("Сценарий остановлен")
                    return
                if kind == "tap":
                    if not self.dry_run:
                        import pyautogui
                        pyautogui.press(value)
                    self.log(f"tap {value}" + (" [проверка]" if self.dry_run else ""))
                elif kind == "hold":
                    if not self.dry_run:
                        import pyautogui
                        pyautogui.keyDown(value)
                    self.stop_event.wait(seconds)
                    if not self.dry_run:
                        pyautogui.keyUp(value)
                    self.log(f"hold {value} {seconds:g}")
                else:
                    self.stop_event.wait(seconds)
                    self.log(f"wait {seconds:g}")
            self.log("Сценарий завершён")
        except Exception as exc:
            self.log(f"Ошибка: {exc}")


def parse_script(text: str) -> list[tuple[str, str, float]]:
    result = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "tap" and len(parts) == 2:
            result.append(("tap", parts[1], 0.0))
        elif parts[0] in ("hold", "wait") and len(parts) == (3 if parts[0] == "hold" else 2):
            result.append((parts[0], parts[1] if parts[0] == "hold" else "", float(parts[-1])))
        else:
            raise ValueError(f"Строка {line_no}: tap KEY, hold KEY SEC или wait SEC")
    return result


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RP Automation")
        self.resize(880, 620)
        self.worker = None
        self._build()
        self.write("Готово. Безопасный режим включён.")

    def _build(self):
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 24)
        self.setCentralWidget(root)
        layout.addWidget(QtWidgets.QLabel("RP Automation", objectName="title"))
        layout.addWidget(QtWidgets.QLabel("Сценарии и распознавание мини-игр", objectName="muted"))
        card = QtWidgets.QFrame(objectName="card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.addWidget(QtWidgets.QLabel("Редактор сценария"))
        card_layout.addWidget(QtWidgets.QLabel("Команды: tap KEY, hold KEY SEC, wait SEC", objectName="muted"))
        self.editor = QtWidgets.QPlainTextEdit("# тест без реального ввода\ntap f\nwait 0.5\nhold w 1")
        card_layout.addWidget(self.editor, 1)
        controls = QtWidgets.QHBoxLayout()
        self.dry = QtWidgets.QCheckBox("Безопасный режим: не нажимать клавиши")
        self.dry.setChecked(True)
        controls.addWidget(self.dry)
        controls.addStretch()
        start = QtWidgets.QPushButton("Запустить", objectName="primary")
        stop = QtWidgets.QPushButton("Остановить", objectName="danger")
        start.clicked.connect(self.start)
        stop.clicked.connect(self.stop)
        controls.addWidget(start)
        controls.addWidget(stop)
        card_layout.addLayout(controls)
        layout.addWidget(card, 1)
        log_card = QtWidgets.QFrame(objectName="card")
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.addWidget(QtWidgets.QLabel("Журнал"))
        self.output = QtWidgets.QPlainTextEdit(readOnly=True)
        log_layout.addWidget(self.output)
        layout.addWidget(log_card, 1)

    def write(self, text):
        self.output.appendPlainText(f"[{datetime.now():%H:%M:%S}] {text}")

    def start(self):
        self.stop()
        try:
            script = parse_script(self.editor.toPlainText())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Ошибка сценария", str(exc))
            return
        self.worker = Worker(script, self.dry.isChecked(), self.write)
        self.worker.start()
        self.write("Сценарий запущен")

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

    def closeEvent(self, event):
        self.stop()
        event.accept()


app = QtWidgets.QApplication(sys.argv)
app.setStyle("Fusion")
app.setStyleSheet(STYLE)
window = MainWindow()
window.show()
sys.exit(app.exec())
