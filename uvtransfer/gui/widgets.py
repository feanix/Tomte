"""Reusable widgets for the Tomte window."""

import os

from PySide6.QtCore import QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
                               QLineEdit, QProgressBar, QPushButton, QTableWidget, QWidget)

from ..images import IMAGE_EXTENSIONS
from .rows import Status

GREY = "#9a9a9a"
UV_LINE_COLOUR = QColor(255, 210, 60, 150)


def local_paths(mime):
    """Local file paths in drag-and-drop data."""
    return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()] if mime.hasUrls() else []


def is_image(path):
    """True for an existing file with a supported image extension."""
    return os.path.isfile(path) and path.lower().endswith(IMAGE_EXTENSIONS)


def is_fbx(path):
    """True for an existing .fbx file."""
    return os.path.isfile(path) and path.lower().endswith(".fbx")


class FileField(QWidget):
    """Path box with a Browse button. Accepts a dropped file (or folder in folder mode).

    Signals:
        changed: the path changed for any reason.
        userChanged: the user typed, browsed or dropped a path.
    """

    changed = Signal()
    userChanged = Signal()

    def __init__(self, placeholder, settings, file_filter="", folder=False, title="Choose file"):
        super().__init__()
        self.settings, self.file_filter, self.folder, self.title = settings, file_filter, folder, title
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setAcceptDrops(False)           # the whole field handles drops
        self.edit.textChanged.connect(self.changed)
        self.edit.textEdited.connect(self.userChanged)
        self.button = QPushButton("Browse…")
        self.button.clicked.connect(self.browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)
        self.setAcceptDrops(True)

    def path(self):
        return self.edit.text().strip().strip('"')

    def set_path(self, path, user=False):
        self.edit.setText(os.path.normpath(path) if path else "")
        if user:
            self.userChanged.emit()

    def browse(self):
        """Open a file (or folder) dialog starting at the current path."""
        start = self.path()
        if start and not os.path.isdir(start):
            start = os.path.dirname(start)
        start = start or self.settings.value("last_dir", "")
        if self.folder:
            chosen = QFileDialog.getExistingDirectory(self, self.title, start)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, self.title, start, self.file_filter)
        if chosen:
            self.settings.setValue("last_dir", chosen if self.folder else os.path.dirname(chosen))
            self.set_path(chosen, user=True)

    def _accepts(self, path):
        return os.path.isdir(path) if self.folder else is_fbx(path)

    def dragEnterEvent(self, e):
        if any(self._accepts(p) for p in local_paths(e.mimeData())):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for p in local_paths(e.mimeData()):
            if self._accepts(p):
                self.set_path(p, user=True)
                e.acceptProposedAction()
                return


class TextureTable(QTableWidget):
    """Table of textures (name, type, size, status) that accepts dropped image files."""

    filesDropped = Signal(list)
    COLUMNS = ("Texture", "Type", "Size", "Status")

    def __init__(self):
        super().__init__(0, len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(3, 200)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(30)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setMinimumHeight(140)

    def _image_paths(self, e):
        return [p for p in local_paths(e.mimeData()) if is_image(p)]

    def dragEnterEvent(self, e):
        if self._image_paths(e):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if self._image_paths(e):
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = self._image_paths(e)
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.rowCount() == 0:
            p = QPainter(self.viewport())
            p.setPen(QColor(GREY))
            p.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter,
                       "Drop colour and normal map textures here\nor click Add…")


class NoWheelComboBox(QComboBox):
    """Combo box that ignores the mouse wheel unless focused, so scrolling a table can't change it."""

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class StatusCell(QProgressBar):
    """Progress bar showing a texture's `Status`, progress and details tooltip."""

    def __init__(self):
        super().__init__()
        self.setRange(0, 1000)
        self.setTextVisible(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._colour = None
        self.show_status(Status.READY)

    def show_status(self, status, fraction=0.0, text=None, tooltip=""):
        """Display `status` (text defaults to the status's label)."""
        self.setValue(int(fraction * 1000))
        self.setFormat(text or status.value)
        self.setToolTip(tooltip)
        if status.colour != self._colour:            # re-styling is slow; only on change
            self._colour = status.colour
            self.setStyleSheet(
                "QProgressBar { border: 1px solid #444; border-radius: 3px; background: #232323;"
                " color: #eee; }"
                f"QProgressBar::chunk {{ background: {status.colour}; }}")


class ImageView(QWidget):
    """Texture scaled to fit, with an optional UV wireframe drawn on top.

    Double-click opens the full image in the system viewer.
    """

    MARGIN, TITLE_HEIGHT = 8, 28

    def __init__(self, title):
        super().__init__()
        self.title = title
        self.image = None
        self.path = ""
        self.message = ""
        self.wireframe = None      # QPainterPath in UV units
        self.show_uvs = True
        self._scaled = None        # image scaled to the current size
        self._overlay = None       # wireframe rendered at the current size
        self.setMinimumSize(220, 240)
        self.setToolTip("Double-click to open the full image")

    def set_image(self, qimage, path="", message=""):
        """Show `qimage` (or `message` when it is None)."""
        self.image = qimage
        self.path = path if qimage is not None else ""
        self.message = message
        self._scaled = None
        self.update()

    def set_wireframe(self, path):
        """UV wireframe to draw (a QPainterPath in UV units), or None."""
        self.wireframe = path
        self._overlay = None
        self.update()

    def set_show_uvs(self, on):
        """Show or hide the UV wireframe."""
        self.show_uvs = on
        self.update()

    def _image_rect(self):
        area = self.rect().adjusted(self.MARGIN, self.TITLE_HEIGHT, -self.MARGIN, -self.MARGIN)
        aspect = 1.0
        if self.image is not None and not self.image.isNull():
            aspect = self.image.width() / self.image.height()
        w = min(area.width(), area.height() * aspect)
        h = w / aspect
        return QRectF(area.x() + (area.width() - w) / 2, area.y() + (area.height() - h) / 2, w, h)

    def resizeEvent(self, e):
        self._scaled = None
        self._overlay = None
        super().resizeEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1b1b1b"))
        p.setPen(QColor("#d0d0d0"))
        p.drawText(10, 19, self.title)
        rect = self._image_rect()
        p.fillRect(rect, QColor("#262626"))
        if self.image is not None and not self.image.isNull():
            size = rect.size().toSize()
            if self._scaled is None or self._scaled.size() != size:
                self._scaled = QPixmap.fromImage(self.image).scaled(
                    size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            p.drawPixmap(rect.topLeft(), self._scaled)
        else:
            p.setPen(QColor(GREY))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.message)
        if self.show_uvs and self.wireframe is not None:
            if self._overlay is None or self._overlay.deviceIndependentSize() != rect.size():
                self._overlay = self._render_wireframe(rect)
            p.drawPixmap(rect.topLeft(), self._overlay)
        p.end()

    def _render_wireframe(self, rect):
        """The wireframe drawn into a transparent pixmap the size of `rect` (cached by paintEvent)."""
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap((rect.size() * ratio).toSize())
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        p.translate(0, rect.height())             # UV (0,0) at bottom-left, v up
        p.scale(rect.width(), -rect.height())
        pen = QPen(UV_LINE_COLOUR)
        pen.setCosmetic(True)                     # 1 px wide whatever the scale
        p.setPen(pen)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.drawPath(self.wireframe)
        p.end()
        return pixmap

    def mouseDoubleClickEvent(self, e):
        if self.path and os.path.isfile(self.path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.path))
