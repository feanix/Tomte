"""Starting the Tomte desktop app."""

import os
import sys
import traceback

import PySide6
from PySide6.QtCore import QObject
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMessageBox, QStyleFactory

from .main_window import APP_NAME, MainWindow


PYSIDE_FIX = 'python -m pip install "PySide6<6.12"'


def pyside_drops_none():
    """True if PySide6 has the bug (6.12.0 on Python 3.10 and 3.11) where every call that
    returns nothing loses a reference to None, so Python crashes after a while."""
    if sys.version_info >= (3, 12):
        return False                      # None can't be freed from Python 3.12 on
    probe = QObject()
    before = sys.getrefcount(None)
    for _ in range(20):
        probe.setObjectName("probe")
    return sys.getrefcount(None) < before - 10


def apply_dark_theme(app):
    """Fusion style with a dark palette, to match DCC tools."""
    app.setStyle(QStyleFactory.create("Fusion"))
    p = QPalette()
    base, window, text = QColor(35, 35, 35), QColor(45, 45, 45), QColor(220, 220, 220)
    roles = QPalette.ColorRole
    for role, colour in ((roles.Window, window), (roles.WindowText, text), (roles.Base, base),
                         (roles.AlternateBase, window), (roles.ToolTipBase, QColor(25, 25, 25)),
                         (roles.ToolTipText, text), (roles.Text, text), (roles.Button, window),
                         (roles.ButtonText, text), (roles.Highlight, QColor(47, 111, 191)),
                         (roles.HighlightedText, QColor(255, 255, 255)),
                         (roles.PlaceholderText, QColor(120, 120, 120))):
        p.setColor(role, colour)
    for role in (roles.Text, roles.ButtonText, roles.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(120, 120, 120))
    app.setPalette(p)


def main(argv=None):
    """Open the window. FBX, image and setup (.tomte) paths in `argv` are loaded. Returns the exit code."""
    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName(APP_NAME)
    if pyside_drops_none():
        python = ".".join(map(str, sys.version_info[:2]))
        text = (f"PySide6 {PySide6.__version__} has a bug on Python {python} that would make {APP_NAME} "
                f"crash. Fix it by running:\n\n    {PYSIDE_FIX}\n\nor use Python 3.12 or later.")
        if sys.__stderr__:
            sys.__stderr__.write(text + "\n")
        QMessageBox.critical(None, APP_NAME, text)
        return 1
    apply_dark_theme(app)
    window = MainWindow(paths=[os.path.abspath(a) for a in argv])

    def show_crash(etype, value, tb):
        # pythonw has no console, so unexpected errors would otherwise vanish
        text = "".join(traceback.format_exception(etype, value, tb))
        if sys.__stderr__:
            sys.__stderr__.write(text)
        QMessageBox.critical(window, APP_NAME, f"Something went wrong:\n\n{value}\n\n{text[-1500:]}")

    sys.excepthook = show_crash
    window.show()
    return app.exec()
