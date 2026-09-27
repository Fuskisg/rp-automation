"""Computer-vision helpers for recognizing game mini-game prompts."""

from .matcher import Match, TemplateMatcher
from .screen import CaptureRegion, ScreenGrabber
from .signals import ColorSignal, ColorSignalDetector

__all__ = ["CaptureRegion", "ColorSignal", "ColorSignalDetector", "Match", "ScreenGrabber", "TemplateMatcher"]

