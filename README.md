# RP Automation

Самостоятельный модуль анализа мини-игр. Проект использует захват экрана через `mss`, поиск шаблонов через OpenCV и анализ цветных сигналов через HSV. Исходные игровые ассеты другого проекта не копируются.

## Запуск

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Основной файл — `vision.py`. Он содержит `ScreenGrabber`, `TemplateMatcher` и `ColorSignalDetector`. Шаблоны должны быть сняты с конкретного разрешения экрана.
