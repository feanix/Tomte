"""The Tomte main window: pick two FBX files, add textures, transfer, preview."""

import html
import os
import shlex
import threading
import time

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QPushButton,
                               QSpinBox, QSplitter, QTableWidgetItem, QVBoxLayout, QWidget, QComboBox,
                               QFileDialog)

from ..errors import TransferError
from ..images import IMAGE_EXTENSIONS, image_info
from ..setupfile import EXTENSION as SETUP_EXTENSION, Setup, TextureEntry
from ..system import available_memory
from ..tiles import describe as describe_tiles
from ..transfer import Options, check_map_name, classify, default_output_dir, estimate_memory
from . import placement
from .preview import PreviewCache
from .rows import PLAIN, Status, TextureRow, TileTexture
from .widgets import (GREY, FileField, FillButton, ImageView, NoWheelComboBox, StatusCell, TextureTable,
                      is_fbx, is_image, local_paths)
from .worker import Bridge, Job, TransferWorker, load_meshes

APP_NAME = "Tomte"
KIND_CHOICES = [
    ("srgb", "Colour", "Colour texture in sRGB (base colour, albedo, emissive)"),
    ("normal", "Normal map", "Tangent-space normal map: directions are re-oriented for the new UVs"),
    ("linear", "Data", "Linear data (roughness, metallic, AO, masks, height...)"),
]
SIZE_CHOICES = [("Same as input", 0), ("512", 512), ("1024", 1024), ("2048", 2048),
                ("4096", 4096), ("8192", 8192)]
QUALITY_CHOICES = [("Draft (1 sample per pixel)", 1), ("Normal (2×2 samples)", 2),
                   ("High (3×3 samples)", 3), ("Best (4×4 samples)", 4)]
CONVENTION_CHOICES = [("Auto-detect", "auto"), ("DirectX (Unreal)", "dx"),
                      ("OpenGL (Unity, Blender)", "gl")]
IMAGE_FILTER = "Images (" + " ".join("*" + e for e in IMAGE_EXTENSIONS) + ")"
SETUP_FILTER = f"Tomte setups (*{SETUP_EXTENSION})"
CLI_SCRIPT = "tomte_cli.py"    # next to the uvtransfer package in a downloaded copy
PLAIN_HINT = "Types are guessed from file names; change them if needed."
TILED_HINT = ("Each row is one output. Drop textures onto a tile's column; double-click a name to "
              "rename it.")

GREEN, AMBER, RED = "#6cc070", "#e0b050", "#e06060"
MESH_RELOAD_DELAY_MS = 300   # wait for typing to pause before re-reading the FBX files
MEMORY_HEADROOM = 0.8        # ask before a transfer that needs more than 80% of free memory
CLOSE_WAIT_SECONDS = 30      # how long closing waits for a cancelled transfer to stop
ROW_ID = Qt.ItemDataRole.UserRole


def _size_text(width, height, high_precision):
    """Text for a texture's Size column, e.g. "2048 × 2048  16-bit"."""
    return f"{width} × {height}" + ("  16-bit" if high_precision else "")


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def is_setup(path):
    """True for an existing setup file."""
    return os.path.isfile(path) and path.lower().endswith(SETUP_EXTENSION)


def _select(combo, value, label):
    """Select the item whose data is `value`, adding it (as `label`) if it isn't offered."""
    index = combo.findData(value)
    if index < 0:
        combo.addItem(label, value)
        index = combo.count() - 1
    combo.setCurrentIndex(index)


def run_command(setup_path):
    """A shell command that runs `setup_path` with this copy of Tomte.

    Starts with a bare `python` so it works in cmd, PowerShell and POSIX shells.
    """
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(os.path.dirname(package), CLI_SCRIPT)
    quote = (lambda p: f'"{p}"') if os.name == "nt" else shlex.quote
    if os.path.isfile(script):
        return f"python {quote(script)} --setup {quote(setup_path)}"
    return f"uvtransfer --setup {quote(setup_path)}"        # installed with pip


class MainWindow(QMainWindow):
    """Main window.

    Mesh loading: whenever the FBX paths or UV channel change, the pair is
    re-read in the background. Each load is tagged with a token (paths, UV
    channel, reload count and file times); results for a token that is no
    longer current are ignored. Pressing Transfer while the meshes are
    loading, or after the files changed on disk, loads them first and then
    starts. Reload (F5) re-reads the meshes and every texture on demand.

    Args:
        settings: QSettings to remember folders and options in (default: per user).
        paths: FBX and image files to open (e.g. from the command line).
    """

    def __init__(self, settings=None, paths=()):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.settings = settings if settings is not None else QSettings("Tomte", "Tomte")
        self.bridge = Bridge()
        self.previews = PreviewCache()

        self.transfer = None          # Transfer for the current FBX pair, once loaded
        self.mesh_token = None        # token of the loaded (or failed) pair
        self.loading_token = None     # token of the load in progress
        self.mesh_error = ""
        self.mesh_generation = 0      # bumped by Reload so the same files are read again
        self.setup_path = ""          # setup file last opened or saved
        self.run_after_load = False   # Transfer was pressed while a load was needed
        self.worker = None            # TransferWorker while a batch runs
        self.rows = {}                # row id -> TextureRow
        self.next_row_id = 1
        self.preview_request = 0
        self.out_auto = True          # output folder follows the target FBX
        self.src_wireframe = None     # source UVs, all of them
        self.tile_wireframes = {}     # tile -> source UVs in that tile, moved into 0 to 1

        self._build_ui()
        self._connect_signals()
        self._restore_settings()
        self._open_paths(paths)
        self._schedule_mesh_load()
        self._update_enabled()

    @property
    def running(self):
        """True while a batch of transfers is in progress."""
        return self.worker is not None

    # -- dialogs (separate so tests can replace them) ---------------------------

    def ask(self, text):
        """Yes/No question. Returns True for Yes."""
        answer = QMessageBox.question(self, APP_NAME, text,
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.Yes)
        return answer == QMessageBox.StandardButton.Yes

    def warn(self, text):
        QMessageBox.warning(self, APP_NAME, text)

    def pick_colour(self, initial):
        """Colour dialog. Returns a tuple of 0..1 values (RGB, or RGBA if not opaque), or None if cancelled."""
        start = QColor.fromRgbF(*(list(initial[:3]) + [initial[3] if len(initial) > 3 else 1.0])) \
            if initial and len(initial) >= 3 else QColor(128, 128, 128)
        colour = QColorDialog.getColor(start, self, "Fill colour",
                                       QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if not colour.isValid():
            return None
        values = [round(c / 255, 3) for c in (colour.red(), colour.green(), colour.blue(), colour.alpha())]
        return tuple(values if values[3] < 1 else values[:3])

    def pick_texture(self, title):
        """File dialog for one image. Returns its path, or "" if cancelled."""
        start = self.settings.value("last_tex_dir", "") or os.path.dirname(self.src_field.path())
        path, _ = QFileDialog.getOpenFileName(self, title, start, IMAGE_FILTER)
        if path:
            self.settings.setValue("last_tex_dir", os.path.dirname(path))
        return path

    def inform(self, text):
        QMessageBox.information(self, APP_NAME, text)

    # -- building the window -----------------------------------------------------

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")

        def action(text, slot, shortcut=None, tip=""):
            a = QAction(text, self)
            if shortcut is not None:
                a.setShortcut(QKeySequence(shortcut))
            a.setStatusTip(tip)
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        self.open_action = action("Open setup…", self.open_setup_dialog, QKeySequence.StandardKey.Open,
                                  "Load meshes, textures and settings saved in a setup file")
        self.save_action = action("Save setup", self.save_setup, QKeySequence.StandardKey.Save,
                                  "Save the meshes, textures and settings to a setup file")
        self.save_as_action = action("Save setup as…", self.save_setup_as, QKeySequence.StandardKey.SaveAs)
        menu.addSeparator()
        self.copy_command_action = action(
            "Copy command line", self.copy_command_line, None,
            "Save the setup and copy a command that runs it without the app")
        menu.addSeparator()
        self.reload_action = action("Reload files", self.reload, QKeySequence.StandardKey.Refresh,
                                    "Read the FBX files and textures again from disk")

    def _build_ui(self):
        self._build_menu()
        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(10, 10, 6, 10)
        column.addWidget(self._build_mesh_box())
        column.addWidget(self._build_texture_box(), 1)
        column.addWidget(self._build_output_box())
        column.addLayout(self._build_actions())
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(2000)
        self.log_box.setFixedHeight(110)
        column.addWidget(self.log_box)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self._build_preview_panel())
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.setCentralWidget(self.splitter)
        self.setAcceptDrops(True)
        self.resize(1300, 860)

        self.mesh_timer = QTimer(self)
        self.mesh_timer.setSingleShot(True)
        self.mesh_timer.setInterval(MESH_RELOAD_DELAY_MS)

    def _build_mesh_box(self):
        self.src_field = FileField("FBX the textures were made for", self.settings,
                                   "FBX files (*.fbx)", title="Choose source FBX")
        self.src_field.setToolTip("The mesh whose UV layout the textures were painted on")
        self.dst_field = FileField("The same mesh with the new UVs", self.settings,
                                   "FBX files (*.fbx)", title="Choose target FBX")
        self.dst_field.setToolTip("The same mesh (same vertices and faces) with the new UV layout")
        self.mesh_status = QLabel()
        self.mesh_status.setWordWrap(True)
        self.mesh_status.setTextFormat(Qt.TextFormat.RichText)
        box = QGroupBox("Meshes")
        form = QFormLayout(box)
        form.addRow("Source FBX", self.src_field)
        form.addRow("Target FBX", self.dst_field)
        form.addRow("", self.mesh_status)
        return box

    def _build_texture_box(self):
        self.table = TextureTable()
        self.add_btn = QPushButton("Add…")
        self.remove_btn = QPushButton("Remove")
        self.clear_btn = QPushButton("Clear")
        self.texture_hint = QLabel(PLAIN_HINT)
        self.texture_hint.setStyleSheet(f"color: {GREY}")
        self.texture_hint.setWordWrap(True)
        buttons = QHBoxLayout()
        for b in (self.add_btn, self.remove_btn, self.clear_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.texture_hint)
        box = QGroupBox("Textures")
        layout = QVBoxLayout(box)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
        return box

    def _build_output_box(self):
        """Output folder and transfer settings."""
        self.out_field = FileField("Folder for the transferred textures", self.settings,
                                   folder=True, title="Choose output folder")
        self.size_combo = self._combo(SIZE_CHOICES, "Longest side of the output textures")
        self.quality_combo = self._combo(
            QUALITY_CHOICES, "More samples give smoother results where islands were scaled down")
        self.conv_combo = self._combo(
            CONVENTION_CHOICES, "Which way the green channel points. Unreal uses DirectX; Unity and "
            "Blender use OpenGL. Auto-detect works it out from the texture.")
        self.pad_spin = QSpinBox()
        self.pad_spin.setRange(0, 512)
        self.pad_spin.setSuffix(" px")
        self.pad_spin.setFixedWidth(90)
        self.pad_spin.setToolTip("How far colour bleeds past island edges, to hide seams in mipmaps")
        self.fill_check = QCheckBox("Fill all empty space")
        padding = QHBoxLayout()
        padding.setContentsMargins(0, 0, 0, 0)
        padding.addWidget(self.pad_spin)
        padding.addWidget(self.fill_check)
        padding.addStretch(1)
        padding_row = QWidget()
        padding_row.setLayout(padding)
        self.uv_spin = QSpinBox()
        self.uv_spin.setRange(0, 7)
        self.uv_spin.setFixedWidth(70)
        self.uv_spin.setToolTip("Which UV channel to read from both meshes (0 = first)")
        self.sixteen_check = QCheckBox("Save as 16-bit PNG")
        self.sixteen_check.setToolTip("Recommended for normal maps. High-precision inputs are always "
                                      "saved as 16-bit.")
        box = QGroupBox("Output")
        form = QFormLayout(box)
        form.addRow("Folder", self.out_field)
        form.addRow("Size", self.size_combo)
        form.addRow("Quality", self.quality_combo)
        form.addRow("Edge padding", padding_row)
        form.addRow("Normal maps", self.conv_combo)
        form.addRow("UV channel", self.uv_spin)
        form.addRow("", self.sixteen_check)
        return box

    @staticmethod
    def _combo(choices, tooltip):
        combo = QComboBox()
        for label, value in choices:
            combo.addItem(label, value)
        combo.setToolTip(tooltip)
        combo.setMaximumWidth(320)
        return combo

    def _build_actions(self):
        self.run_btn = QPushButton("Transfer")
        self.run_btn.setStyleSheet(
            "QPushButton { background: #2f6fbf; color: white; font-weight: bold; padding: 0 22px; }"
            "QPushButton:disabled { background: #3a3a3a; color: #777; }")
        self.cancel_btn = QPushButton("Cancel")
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setToolTip("Read the FBX files and textures again from disk (F5)")
        self.open_btn = QPushButton("Open output folder")
        row = QHBoxLayout()
        for b in (self.run_btn, self.cancel_btn, self.reload_btn):
            b.setMinimumHeight(34)
            row.addWidget(b)
        row.addStretch(1)
        self.open_btn.setMinimumHeight(34)
        row.addWidget(self.open_btn)
        return row

    def _build_preview_panel(self):
        title = QLabel("Preview")
        title.setStyleSheet("font-weight: bold")
        self.preview_name = QLabel("Select a texture to preview it")
        self.preview_name.setStyleSheet(f"color: {GREY}")
        self.uv_check = QCheckBox("Show UVs")
        header = QHBoxLayout()
        header.addWidget(title)
        header.addWidget(self.preview_name, 1)
        header.addWidget(self.uv_check)
        self.before_view = ImageView("Before  (source UVs)")
        self.after_view = ImageView("After  (target UVs)")
        views = QSplitter(Qt.Orientation.Vertical)
        views.addWidget(self.before_view)
        views.addWidget(self.after_view)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 10, 10, 10)
        layout.addLayout(header)
        layout.addWidget(views, 1)
        return panel

    def _connect_signals(self):
        """Wire widgets and worker signals to their handlers."""
        b = self.bridge
        b.meshLoaded.connect(self._on_mesh_loaded)
        b.progress.connect(self._on_progress)
        b.rowDone.connect(self._on_row_done)
        b.rowFailed.connect(self._on_row_failed)
        b.batchDone.connect(self._on_batch_done)
        b.previewReady.connect(self._on_preview_ready)

        self.mesh_timer.timeout.connect(self._start_mesh_load)
        self.src_field.changed.connect(self._schedule_mesh_load)
        self.dst_field.changed.connect(self._schedule_mesh_load)
        self.dst_field.changed.connect(self._update_auto_output)
        self.uv_spin.valueChanged.connect(self._schedule_mesh_load)
        self.out_field.userChanged.connect(self._output_chosen)
        self.out_field.changed.connect(self._update_enabled)
        self.out_field.changed.connect(self._refresh_preview)
        self.sixteen_check.toggled.connect(self._refresh_preview)
        self.fill_check.toggled.connect(self._update_enabled)

        self.add_btn.clicked.connect(self._browse_textures)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear_textures)
        self.table.filesDropped.connect(self._files_dropped)
        self.table.itemSelectionChanged.connect(self._refresh_preview)
        self.table.itemSelectionChanged.connect(self._update_enabled)
        self.table.itemChanged.connect(self._item_changed)
        self.table.cellDoubleClicked.connect(self._cell_double_clicked)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        self.delete_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.table)
        self.delete_shortcut.activated.connect(self._remove_selected)

        self.uv_check.toggled.connect(self.before_view.set_show_uvs)
        self.uv_check.toggled.connect(self.after_view.set_show_uvs)
        self.run_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self.cancel)
        self.reload_btn.clicked.connect(self.reload)
        self.open_btn.clicked.connect(self._open_output)

    # -- settings ---------------------------------------------------------------

    def _restore_settings(self):
        s = self.settings
        if s.value("geometry") is not None:
            self.restoreGeometry(s.value("geometry"))
        if s.value("splitter") is not None:
            self.splitter.restoreState(s.value("splitter"))
        self.src_field.set_path(s.value("source", ""))
        self.dst_field.set_path(s.value("target", ""))
        self.out_auto = s.value("out_auto", True, type=bool)
        if not self.out_auto:
            self.out_field.set_path(s.value("out", ""))
        self._update_auto_output()
        self._restore_index(self.size_combo, "size_idx", 0)
        self._restore_index(self.quality_combo, "quality_idx", 1)
        self.pad_spin.setValue(s.value("pad", 16, type=int))
        self.fill_check.setChecked(s.value("fill_all", False, type=bool))
        self._restore_index(self.conv_combo, "conv_idx", 0)
        self.uv_spin.setValue(s.value("uv_set", 0, type=int))
        self.sixteen_check.setChecked(s.value("sixteen", False, type=bool))
        self.uv_check.setChecked(s.value("show_uvs", True, type=bool))

    def _restore_index(self, combo, key, default):
        index = self.settings.value(key, default, type=int)
        combo.setCurrentIndex(index if 0 <= index < combo.count() else default)

    def save_settings(self):
        """Remember paths, options and window layout for next time."""
        s = self.settings
        s.setValue("geometry", self.saveGeometry())
        s.setValue("splitter", self.splitter.saveState())
        s.setValue("source", self.src_field.path())
        s.setValue("target", self.dst_field.path())
        s.setValue("out", self.out_field.path())
        s.setValue("out_auto", self.out_auto)
        s.setValue("size_idx", self.size_combo.currentIndex())
        s.setValue("quality_idx", self.quality_combo.currentIndex())
        s.setValue("pad", self.pad_spin.value())
        s.setValue("fill_all", self.fill_check.isChecked())
        s.setValue("conv_idx", self.conv_combo.currentIndex())
        s.setValue("uv_set", self.uv_spin.value())
        s.setValue("sixteen", self.sixteen_check.isChecked())
        s.setValue("show_uvs", self.uv_check.isChecked())

    def _open_paths(self, paths):
        setups = [p for p in paths if is_setup(p)]
        if setups:
            self.open_setup(setups[0])
            return
        fbx = [p for p in paths if is_fbx(p)]
        if fbx:
            self.src_field.set_path(fbx[0])
        if len(fbx) > 1:
            self.dst_field.set_path(fbx[1])
        images = [p for p in paths if is_image(p)]
        if images:
            self.add_textures(images)

    def options(self):
        """Current settings as `Options`."""
        return Options(
            size=self.size_combo.currentData(),
            supersample=self.quality_combo.currentData(),
            pad=-1 if self.fill_check.isChecked() else self.pad_spin.value(),
            normal_convention=self.conv_combo.currentData(),
            sixteen_bit=self.sixteen_check.isChecked(),
        )

    def log(self, text, colour=None):
        """Add a time-stamped line to the log box."""
        stamp = time.strftime("%H:%M:%S")
        body = html.escape(text)
        if colour:
            body = f"<span style='color:{colour}'>{body}</span>"
        self.log_box.appendHtml(f"<span style='color:#777'>{stamp}</span>&nbsp; {body}")

    # -- meshes -----------------------------------------------------------------

    def _schedule_mesh_load(self):
        self.mesh_timer.start()

    def _mesh_token(self):
        """Identifies the current FBX pair and UV channel, or None if a path isn't an FBX file."""
        src, dst = self.src_field.path(), self.dst_field.path()
        if not (is_fbx(src) and is_fbx(dst)):
            return None
        try:
            times = (os.path.getmtime(src), os.path.getmtime(dst))
        except OSError:
            return None
        return (_norm(src), _norm(dst), self.uv_spin.value(), self.mesh_generation) + times

    def _start_mesh_load(self):
        token = self._mesh_token()
        if token is None:
            self._clear_meshes()
            return
        if token in (self.mesh_token, self.loading_token):
            return                                # already loaded or loading
        self.loading_token = token
        self.transfer = None
        self.mesh_status.setText(f"<span style='color:{GREY}'>Reading meshes…</span>")
        self._update_enabled()
        load_meshes(self.bridge, token, token[0], token[1], token[2])

    def _clear_meshes(self):
        self.transfer = self.mesh_token = self.loading_token = None
        missing = [name for name, field in (("source", self.src_field), ("target", self.dst_field))
                   if field.path() and not is_fbx(field.path())]
        if missing:
            self.mesh_status.setText(f"<span style='color:{RED}'>✗ The {' and '.join(missing)} path "
                                     "isn't an existing .fbx file.</span>")
        else:
            self.mesh_status.setText(f"<span style='color:{GREY}'>Choose the source and target FBX "
                                     "files (or drop them onto the fields).</span>")
        self._set_wireframes(None, None)
        self._update_enabled()

    def _on_mesh_loaded(self, token, load):
        if token != self.loading_token:
            return                                # superseded by a newer load
        self.loading_token = None
        self.mesh_token = token
        self.transfer = load.transfer
        self.mesh_error = load.error
        if load.transfer is None:
            self.mesh_status.setText(f"<span style='color:{RED}'>✗ {html.escape(load.error)}</span>")
            self.log(load.error, RED)
            self.run_after_load = False
        else:
            lines = [f"<span style='color:{GREEN}'>✓ Meshes match:</span> "
                     f"{html.escape(load.transfer.summary())}"]
            lines += [f"<span style='color:{AMBER}'>⚠ {html.escape(w)}</span>"
                      for w in load.transfer.warnings]
            self.mesh_status.setText("<br>".join(lines))
            self.log(f"Meshes loaded: {load.transfer.summary()}")
            for w in load.transfer.warnings:
                self.log(w, AMBER)
        self.tile_wireframes = load.tile_wireframes or {}
        self._set_wireframes(load.src_wireframe, load.dst_wireframe)
        if load.transfer is not None:
            self._apply_mesh_tiles(load.transfer.tiles)
        self._update_enabled()
        if self.run_after_load and load.transfer is not None:
            self.run_after_load = False
            self.start()

    def _set_wireframes(self, src, dst):
        self.src_wireframe = src
        self.before_view.set_wireframe(src)
        self.after_view.set_wireframe(dst)

    def _apply_mesh_tiles(self, mesh_tiles):
        """Show a column per tile when the source mesh uses several, else the plain list.

        The list only goes back to plain when no row would lose anything
        (several textures, a fill or a chosen name).
        """
        used = {t for r in self.rows.values() if r.tiled for t in r.textures}
        keep = any(r.tiled and not r.is_plain_equivalent() for r in self.rows.values())
        tiles = sorted(set(mesh_tiles) | used) if len(mesh_tiles) > 1 or keep else []
        if tiles == self.table.tiles:
            return
        if tiles and not self.table.tiles:
            self.log(f"The source mesh uses tiles {describe_tiles(tiles)}: each output can take a texture "
                     "per tile.")
        self._convert_rows(tiles)

    # -- output folder ------------------------------------------------------------

    def _update_auto_output(self):
        if self.out_auto:
            dst = self.dst_field.path()
            self.out_field.set_path(default_output_dir(dst) if dst else "")

    def _output_chosen(self):
        self.out_auto = not self.out_field.path()     # clearing the field turns auto back on
        self._update_auto_output()

    def _open_output(self):
        out = self.out_field.path()
        if out and os.path.isdir(out):
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            self.inform("The output folder doesn't exist yet. It's created when you transfer.")

    # -- texture list -------------------------------------------------------------

    @property
    def tiled(self):
        """True when the list shows outputs put together from a texture per UDIM tile."""
        return bool(self.table.tiles)

    def _browse_textures(self):
        start = self.settings.value("last_tex_dir", "") or os.path.dirname(self.src_field.path())
        paths, _ = QFileDialog.getOpenFileNames(self, "Add textures", start, IMAGE_FILTER)
        if paths:
            self.settings.setValue("last_tex_dir", os.path.dirname(paths[0]))
            self.add_textures(paths, tile=self._current_tile())

    def _current_tile(self):
        """The tile of the current cell's column, if it is a tile column."""
        return self.table.tile_at(self.table.currentColumn()) if self.tiled else None

    def _files_dropped(self, paths, row, column):
        self.add_textures(paths, tile=self.table.tile_at(column) if self.tiled else None)

    def _read_texture(self, path):
        """A `TileTexture` for an image file, or None (and a log line) if it can't be read."""
        try:
            mtime = os.path.getmtime(path)
            width, height, high_precision = image_info(path)
        except OSError as e:
            self.log(f"Couldn't read {os.path.basename(path)}: {e}", RED)
            return None
        return TileTexture(path, width, height, high_precision, mtime)

    def add_textures(self, paths, tile=None):
        """Add image files to the list.

        Plain list: a row per texture (duplicates and unreadable files are
        skipped). Tiled list: files go into rows and tiles as `placement`
        decides; files without a tile number go into `tile` (default: the
        first tile).
        """
        if self.running:
            return
        paths = [p for p in paths if is_image(p)]
        if self.tiled:
            self._place_textures(paths, tile or self.table.tiles[0])
            return
        have = {_norm(r.path) for r in self.rows.values()}
        first_new = None
        for path in paths:
            if _norm(path) in have:
                continue
            texture = self._read_texture(path)
            if texture is None:
                continue
            have.add(_norm(path))
            table_row = self._insert_table_row(self._new_row({PLAIN: texture}, classify(path)))
            if first_new is None:
                first_new = table_row
        if first_new is not None and not self.table.selectedItems():
            self.table.selectRow(first_new)
        self._update_enabled()

    def _place_textures(self, paths, tile):
        """Put textures into the tiled list (see `placement`), adding rows and tile columns as needed."""
        textures = {p: t for p in paths if (t := self._read_texture(p)) is not None}
        rows = list(self.rows.values())
        slots = [placement.Slot(r.name, {t: x.path for t, x in r.textures.items()}) for r in rows]
        new_rows = {}
        for index, file_tile, path in placement.place(slots, list(textures), tile):
            if index < len(rows):
                row = rows[index]
                row.textures[file_tile] = textures[path]
                self._mark_edited(row)
            elif index in new_rows:
                new_rows[index].textures[file_tile] = textures[path]
            else:
                new_rows[index] = self._new_row({file_tile: textures[path]}, classify(path), slots[index].name)
        used = {t for r in self.rows.values() for t in r.textures}
        self._rebuild_table(sorted(set(self.table.tiles) | used))
        if textures:
            self.log(f"Added {len(textures)} texture(s)")

    def _new_row(self, textures, kind, name="", fill=None):
        """Create a `TextureRow` (not yet in the table)."""
        row = TextureRow(self.next_row_id, textures, name=name, fill=fill, kind_hint=kind)
        self.next_row_id += 1
        self.rows[row.id] = row
        return row

    def _insert_table_row(self, row):
        """Add `row` at the end of the table with new widgets. Returns the table row number."""
        table = self.table
        r = table.rowCount()
        table.blockSignals(True)                  # no itemChanged while filling in the row
        try:
            table.insertRow(r)
            first = QTableWidgetItem(row.label)
            first.setData(ROW_ID, row.id)
            if row.tiled:
                first.setToolTip("Output file name. Double-click to rename it.")
            else:
                first.setFlags(first.flags() & ~Qt.ItemFlag.ItemIsEditable)
                first.setToolTip(row.path)
            table.setItem(r, 0, first)

            kind = row.kind
            row.type_combo = NoWheelComboBox()
            for i, (value, label, tip) in enumerate(KIND_CHOICES):
                row.type_combo.addItem(label, value)
                row.type_combo.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
            row.type_combo.setCurrentIndex([k for k, _, _ in KIND_CHOICES].index(kind))
            row.type_combo.currentIndexChanged.connect(lambda _=0, rid=row.id: self._kind_changed(rid))
            table.setCellWidget(r, table.type_column, row.type_combo)

            if self.tiled:
                for tile in table.tiles:
                    table.setItem(r, table.tile_column(tile), self._tile_item(row, tile))
                row.fill_button = FillButton()
                row.fill_button.show_fill(row.fill, kind == "normal")
                row.fill_button.clicked.connect(lambda _=False, rid=row.id: self._fill_clicked(rid))
                table.setCellWidget(r, table.fill_column, row.fill_button)

            size = QTableWidgetItem(_size_text(row.width, row.height, row.high_precision))
            size.setFlags(size.flags() & ~Qt.ItemFlag.ItemIsEditable)
            size.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(r, table.size_column, size)

            row.status_cell = StatusCell()
            done = row.status in (Status.DONE, Status.FAILED, Status.MISSING)
            row.status_cell.show_status(row.status, 1.0 if done else 0.0)
            table.setCellWidget(r, table.status_column, row.status_cell)
        finally:
            table.blockSignals(False)
        return r

    def _tile_item(self, row, tile):
        """The cell showing which texture `row` uses for `tile`."""
        texture = row.textures.get(tile)
        item = QTableWidgetItem(os.path.basename(texture.path) if texture else "")
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if texture:
            item.setToolTip(f"{texture.path}\n{_size_text(texture.width, texture.height, texture.high_precision)}")
        else:
            item.setToolTip("No texture: parts in this tile get the row's fill. Drop a texture here, or "
                            "double-click to choose one.")
        return item

    def _rebuild_table(self, tiles):
        """Lay the table out for `tiles` ([] = plain) and fill it with every row again."""
        current = self.table.currentRow(), self.table.currentColumn()
        for row in self.rows.values():
            row.kind_hint = row.kind              # the type dropdowns are about to be replaced
            row.type_combo = row.status_cell = row.fill_button = None
        self.table.set_tiles(tiles)
        for row in self.rows.values():
            self._insert_table_row(row)
        self.texture_hint.setText(TILED_HINT if tiles else PLAIN_HINT)
        if 0 <= current[0] < self.table.rowCount():
            self.table.setCurrentCell(current[0], min(max(current[1], 0), self.table.columnCount() - 1))
        self._update_enabled()
        self._refresh_preview()

    def _convert_rows(self, tiles):
        """Switch the list between plain ([]) and tiled, turning the rows into the other kind."""
        rows = list(self.rows.values())
        if tiles and not self.tiled:              # plain -> tiled: textures are placed like dropped files
            self.rows.clear()
            slots = []
            made = {}
            for old in rows:
                for index, tile, _ in placement.place(slots, [old.path], tiles[0]):
                    if index in made:
                        made[index].textures[tile] = old.first
                    else:
                        made[index] = self._new_row({tile: old.first}, old.kind, slots[index].name)
        elif not tiles and self.tiled:            # tiled -> plain: a row per texture
            self.rows.clear()
            seen = set()
            for old in rows:
                for texture in old.textures.values():
                    if _norm(texture.path) not in seen:
                        seen.add(_norm(texture.path))
                        self._new_row({PLAIN: texture}, old.kind)
        self._rebuild_table(tiles)

    def _row_ids(self, selected_only=False):
        """Row ids in table order (optionally only rows with a selected cell)."""
        if selected_only:
            numbers = sorted({i.row() for i in self.table.selectedIndexes()})
        else:
            numbers = range(self.table.rowCount())
        return [self.table.item(r, 0).data(ROW_ID) for r in numbers]

    def _remove_selected(self):
        """Remove the selected rows; in the tiled list, selecting only tile cells removes just those textures."""
        if self.running:
            return
        selected = self.table.selectedIndexes()
        cells = [(self.table.item(i.row(), 0).data(ROW_ID), self.table.tile_at(i.column())) for i in selected]
        if self.tiled and cells and all(tile is not None for _, tile in cells):
            for row_id, tile in cells:
                row = self.rows[row_id]
                if row.textures.pop(tile, None) is not None:
                    self._mark_edited(row)
            for row_id in {rid for rid, _ in cells}:
                if not self.rows[row_id].textures:
                    del self.rows[row_id]
            self._rebuild_table(self.table.tiles)
            return
        for r in sorted({i.row() for i in selected}, reverse=True):
            self.rows.pop(self.table.item(r, 0).data(ROW_ID), None)
            self.table.removeRow(r)
        self._update_enabled()
        self._refresh_preview()

    def _clear_textures(self):
        if self.running:
            return
        self.table.setRowCount(0)
        self.rows.clear()
        self._update_enabled()
        self._refresh_preview()

    def _kind_changed(self, row_id):
        row = self.rows.get(row_id)
        if row is None:
            return
        if row.fill_button is not None:
            row.fill_button.show_fill(row.fill, row.kind == "normal")
        if row.result_path:
            row.result_path = ""
            self._set_status(row, Status.STALE)
            self._refresh_preview()

    def _mark_edited(self, row):
        """A row's textures, name or fill changed: its last result no longer matches."""
        if row.result_path or row.status in (Status.DONE, Status.FAILED):
            row.result_path = ""
            row.status = Status.EDITED
            if row.status_cell is not None:
                self._set_status(row, Status.EDITED)

    def _table_row(self, row_id):
        """Table row number of a row id, or -1."""
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(ROW_ID) == row_id:
                return r
        return -1

    # -- tiled rows: names, textures per tile, fills ---------------------------------

    def _item_changed(self, item):
        """The user renamed an output (column 0 of the tiled list)."""
        if not self.tiled or item.column() != 0:
            return
        row = self.rows.get(item.data(ROW_ID))
        name = item.text().strip()
        if row is None or name == row.name:
            return
        problem = ""
        try:
            check_map_name(name)
        except TransferError as e:
            problem = str(e)
        if not problem and any(r is not row and r.name.lower() == name.lower() for r in self.rows.values()):
            problem = f"Another output is already called {name}."
        if problem or self.running:
            self.table.blockSignals(True)
            item.setText(row.name)
            self.table.blockSignals(False)
            if problem:
                self.warn(problem)
            return
        row.name = name
        self._mark_edited(row)
        self._refresh_preview()

    def rename_output(self, row_id, name):
        """Rename a tiled row's output, as if typed into the table. Returns True if accepted."""
        r = self._table_row(row_id)
        if r < 0:
            return False
        self.table.item(r, 0).setText(name)
        return self.rows[row_id].name == name

    def _cell_double_clicked(self, r, column):
        tile = self.table.tile_at(column)
        if tile is None or self.running:
            return
        path = self.pick_texture(f"Texture for tile {tile}")
        if path:
            self.set_tile_texture(self.table.item(r, 0).data(ROW_ID), tile, path)

    def set_tile_texture(self, row_id, tile, path):
        """Use `path` for `tile` in a tiled row (None or "" removes the row's texture for that tile)."""
        row = self.rows.get(row_id)
        if row is None or self.running:
            return
        if path:
            texture = self._read_texture(path)
            if texture is None:
                return
            row.textures[tile] = texture
        else:
            row.textures.pop(tile, None)
            if not row.textures:
                del self.rows[row_id]
        self._mark_edited(row)
        self._rebuild_table(self.table.tiles)

    def _fill_clicked(self, row_id):
        row = self.rows.get(row_id)
        if row is None or self.running:
            return
        menu = QMenu(self)
        choose = menu.addAction("Choose colour…")
        empty = menu.addAction("Leave empty")
        chosen = menu.exec(row.fill_button.mapToGlobal(row.fill_button.rect().bottomLeft()))
        if chosen is choose:
            colour = self.pick_colour(row.fill)
            if colour is not None:
                self.set_fill(row_id, colour)
        elif chosen is empty:
            self.set_fill(row_id, None)

    def set_fill(self, row_id, fill):
        """Set a tiled row's fill colour (a tuple of 0..1 values) or None to leave those parts empty."""
        row = self.rows.get(row_id)
        if row is None or row.fill == fill:
            return
        row.fill = tuple(fill) if fill is not None else None
        if row.fill_button is not None:
            row.fill_button.show_fill(row.fill, row.kind == "normal")
        self._mark_edited(row)

    def _table_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid() or self.running:
            return
        row_id = self.table.item(index.row(), 0).data(ROW_ID)
        tile = self.table.tile_at(index.column())
        menu = QMenu(self)
        choose = remove_texture = None
        if tile is not None:
            choose = menu.addAction(f"Choose texture for tile {tile}…")
            if tile in self.rows[row_id].textures:
                remove_texture = menu.addAction(f"Remove the tile {tile} texture")
            menu.addSeparator()
        remove_row = menu.addAction("Remove this output" if self.tiled else "Remove this texture")
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is choose:
            self._cell_double_clicked(index.row(), index.column())
        elif chosen is remove_texture:
            self.set_tile_texture(row_id, tile, None)
        elif chosen is remove_row:
            self.table.clearSelection()
            self.table.selectRow(index.row())
            self.rows.pop(row_id, None)
            self.table.removeRow(index.row())
            self._update_enabled()
            self._refresh_preview()

    # -- setup files ----------------------------------------------------------------

    def current_setup(self):
        """The meshes, textures and settings in the window, as a `Setup`."""
        rows = [self.rows[rid] for rid in self._row_ids()]
        return Setup(
            source=self.src_field.path(), target=self.dst_field.path(),
            textures=[TextureEntry(r.path, r.kind) for r in rows if not r.tiled],
            maps=[r.to_map() for r in rows if r.tiled],
            output="" if self.out_auto else self.out_field.path(),
            uv_set=self.uv_spin.value(), options=self.options())

    def apply_setup(self, setup):
        """Fill the window from `setup` (replacing the texture list)."""
        self.src_field.set_path(setup.source)
        self.dst_field.set_path(setup.target)
        self.out_auto = not setup.output
        if setup.output:
            self.out_field.set_path(setup.output)
        self._update_auto_output()
        self.uv_spin.setMaximum(max(self.uv_spin.maximum(), setup.uv_set))
        self.uv_spin.setValue(setup.uv_set)
        o = setup.options
        _select(self.size_combo, o.size, str(o.size))
        _select(self.quality_combo, o.supersample, f"{o.supersample}×{o.supersample} samples")
        _select(self.conv_combo, o.normal_convention, o.normal_convention)
        self.fill_check.setChecked(o.pad == -1)
        if o.pad >= 0:
            self.pad_spin.setMaximum(max(self.pad_spin.maximum(), o.pad))
            self.pad_spin.setValue(o.pad)
        self.sixteen_check.setChecked(o.sixteen_bit)
        self._clear_textures()
        missing = [t.path for t in setup.textures if not is_image(t.path)]
        if setup.maps:
            tiles = {t for m in setup.maps for t in m.tiles}
            if self.transfer is not None and self._mesh_token() == self.mesh_token:
                tiles |= set(self.transfer.tiles)
            for m in setup.maps:
                textures = {}
                for tile, path in m.tiles.items():
                    texture = self._read_texture(path) if is_image(path) else None
                    if texture is None:
                        missing.append(path)
                    else:
                        textures[tile] = texture
                if textures:
                    self._new_row(textures, m.kind, m.name, m.fill)
            self._rebuild_table(sorted(tiles))
            if setup.textures:                    # plain textures too: each becomes an output
                self._place_textures([t.path for t in setup.textures if is_image(t.path)], min(tiles))
        else:
            if self.tiled:
                self._rebuild_table([])
            self.add_textures([t.path for t in setup.textures])
            kinds = {_norm(t.path): t.kind for t in setup.textures}
            choices = [k for k, _, _ in KIND_CHOICES]
            for row in self.rows.values():
                row.type_combo.setCurrentIndex(choices.index(kinds[_norm(row.path)]))
        if self.transfer is not None and self._mesh_token() == self.mesh_token:
            self._apply_mesh_tiles(self.transfer.tiles)      # same meshes as before: no reload to do it
        return missing

    def open_setup_dialog(self):
        start = os.path.dirname(self.setup_path) or self.settings.value("last_setup_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Open setup", start, SETUP_FILTER)
        if path:
            self.open_setup(path)

    def open_setup(self, path):
        """Load a setup file into the window. Returns True on success."""
        if self.running:
            return False
        try:
            setup = Setup.load(path)
        except TransferError as e:
            self.warn(str(e))
            return False
        missing = self.apply_setup(setup)
        self._set_setup_path(path)
        self.log(f"Opened setup {path}")
        if missing:
            self.warn("These textures in the setup weren't found and were left out:\n\n"
                      + "\n".join(missing))
        return True

    def save_setup(self):
        """Save to the current setup file (asking for a name the first time). Returns True if saved."""
        if not self.setup_path:
            return self.save_setup_as()
        return self._write_setup(self.setup_path)

    def save_setup_as(self):
        """Save to a new setup file. Returns True if saved."""
        if self.setup_path:
            start = self.setup_path
        else:
            target = self.dst_field.path()
            name = (os.path.splitext(os.path.basename(target))[0] if target else "Untitled") + SETUP_EXTENSION
            start = os.path.join(os.path.dirname(target) or self.settings.value("last_setup_dir", ""), name)
        path, _ = QFileDialog.getSaveFileName(self, "Save setup", start, SETUP_FILTER)
        if not path:
            return False
        if not path.lower().endswith(SETUP_EXTENSION):
            path += SETUP_EXTENSION
        return self._write_setup(path)

    def _write_setup(self, path):
        if not (self.src_field.path() and self.dst_field.path()):
            self.warn("Choose the source and target FBX files before saving a setup.")
            return False
        try:
            self.current_setup().save(path)
        except TransferError as e:
            self.warn(str(e))
            return False
        self._set_setup_path(path)
        self.log(f"Saved setup {path}", GREEN)
        return True

    def _set_setup_path(self, path):
        self.setup_path = os.path.abspath(path)
        self.settings.setValue("last_setup_dir", os.path.dirname(self.setup_path))
        self.setWindowTitle(f"{os.path.basename(path)} - {APP_NAME}")

    def copy_command_line(self):
        """Save the setup, then copy a command that runs it from a terminal or script."""
        if not self.save_setup():
            return
        command = run_command(self.setup_path)
        QApplication.clipboard().setText(command)
        self.log(f"Copied: {command}")
        self.statusBar().showMessage("Command copied. Paste it into a terminal or script to rerun "
                                     "this setup.", 8000)

    # -- reload -------------------------------------------------------------------

    def reload(self):
        """Read the FBX files and every texture in the list again from disk."""
        if self.running:
            return
        self.mesh_generation += 1
        self.mesh_timer.stop()
        self._start_mesh_load()
        self.previews.clear()
        outcomes = [self._reload_row(self.rows[rid], r) for r, rid in enumerate(self._row_ids())]
        changed, missing = outcomes.count("changed"), outcomes.count("missing")
        details = ", ".join(d for d in (f"{changed} changed" if changed else "",
                                        f"{missing} missing" if missing else "") if d)
        self.log(f"Reloaded the meshes and {len(outcomes)} texture(s)" + (f" ({details})" if details else ""),
                 RED if missing else None)
        self._refresh_preview()
        self._update_enabled()

    def _reload_row(self, row, table_row):
        """Re-read a row's textures' sizes and formats. Returns "same", "changed" or "missing".

        A finished transfer is marked out of date if one of its textures changed since.
        """
        changed = False
        for tile, texture in row.textures.items():
            try:
                mtime = os.path.getmtime(texture.path)
                texture.width, texture.height, texture.high_precision = image_info(texture.path)
            except OSError as e:
                row.result_path = ""
                self._set_status(row, Status.MISSING, 1.0, tooltip=f"{texture.path}\n{e}")
                return "missing"
            changed = changed or mtime != texture.mtime
            texture.mtime = mtime
            if tile != PLAIN:
                self.table.setItem(table_row, self.table.tile_column(tile), self._tile_item(row, tile))
        self.table.item(table_row, self.table.size_column).setText(
            _size_text(row.width, row.height, row.high_precision))
        if changed and row.result_path:
            row.result_path = ""
            self._set_status(row, Status.CHANGED)
        elif row.status in (Status.MISSING, Status.FAILED):
            self._set_status(row, Status.READY)
        return "changed" if changed else "same"

    def _set_status(self, row, status, fraction=0.0, text=None, tooltip=""):
        row.status = status
        row.status_cell.show_status(status, fraction, text, tooltip)

    # -- preview ------------------------------------------------------------------

    def _after_path(self, row):
        """(path to show as "after", note): this session's result, or an existing output file."""
        if row.result_path and os.path.isfile(row.result_path):
            return row.result_path, ""
        out = self.out_field.path()
        if out:
            candidate = os.path.join(out, row.output_name(self.sixteen_check.isChecked()))
            if os.path.isfile(candidate) and all(_norm(candidate) != _norm(t.path) for t in row.textures.values()):
                return candidate, " (existing file)"
        return "", ""

    def _refresh_preview(self):
        self.preview_request += 1
        request = self.preview_request
        selected = self._row_ids(selected_only=True)
        current = self.table.currentRow()
        if 0 <= current < self.table.rowCount() and self.table.item(current, 0).data(ROW_ID) in selected:
            selected.insert(0, self.table.item(current, 0).data(ROW_ID))
        row = self.rows.get(selected[0]) if selected else None
        if row is None:
            self.preview_name.setText("Select a texture to preview it")
            self.before_view.set_image(None)
            self.after_view.set_image(None)
            return
        after, note = self._after_path(row)
        if row.tiled:                             # the selected tile's texture, with only that tile's UVs
            tile = self._current_tile()
            tile = tile if tile in row.textures else min(row.textures)
            before = row.textures[tile].path
            self.before_view.title = f"Before  (tile {tile}, source UVs)"
            self.before_view.set_wireframe(self.tile_wireframes.get(tile))
            self.preview_name.setText(f"{row.name}: tile {tile}, {os.path.basename(before)}{note}")
        else:
            before = row.path
            self.before_view.title = "Before  (source UVs)"
            self.before_view.set_wireframe(self.src_wireframe)
            self.preview_name.setText(os.path.basename(row.path) + note)
        self.before_view.set_image(None, message="Loading…")
        self.after_view.set_image(None, message="Loading…" if after else "Not transferred yet")

        def work():
            self.bridge.previewReady.emit(request, (before, self.previews.get(before)),
                                          (after, self.previews.get(after)))

        threading.Thread(target=work, daemon=True, name="preview").start()

    def _on_preview_ready(self, request, before, after):
        if request != self.preview_request:
            return                                # selection changed since
        self.before_view.set_image(before[1], before[0], "Couldn't load the image")
        self.after_view.set_image(after[1], after[0],
                                  "Couldn't load the image" if after[0] else "Not transferred yet")

    # -- enabling -----------------------------------------------------------------

    def _update_enabled(self):
        idle = not self.running
        meshes = self.transfer is not None or self.loading_token is not None
        self.run_btn.setEnabled(idle and meshes and bool(self.rows) and bool(self.out_field.path()))
        self.cancel_btn.setEnabled(self.running)
        self.remove_btn.setEnabled(idle and bool(self.table.selectedIndexes()))
        self.clear_btn.setEnabled(idle and bool(self.rows))
        self.add_btn.setEnabled(idle)
        self.reload_btn.setEnabled(idle)
        for w in (self.src_field, self.dst_field, self.out_field, self.size_combo, self.quality_combo,
                  self.fill_check, self.conv_combo, self.uv_spin, self.sixteen_check,
                  self.open_action, self.reload_action):
            w.setEnabled(idle)
        self.pad_spin.setEnabled(idle and not self.fill_check.isChecked())
        for row in self.rows.values():
            if row.type_combo is not None:
                row.type_combo.setEnabled(idle)
            if row.fill_button is not None:
                row.fill_button.setEnabled(idle and row.kind != "normal")
        self.remove_btn.setToolTip("Remove the selected outputs, or only the selected tile textures"
                                   if self.tiled else "Remove the selected textures")
        if not self.rows:
            self.run_btn.setToolTip("Add some textures first")
        elif self.transfer is None:
            self.run_btn.setToolTip(self.mesh_error or "Choose the source and target FBX files")
        else:
            self.run_btn.setToolTip("")

    # -- running --------------------------------------------------------------------

    def start(self):
        """Transfer every texture in the list, after loading meshes and confirming if needed."""
        if self.running or not self._meshes_ready():
            return
        out_dir = self.out_field.path()
        if not out_dir:
            self.warn("Choose an output folder.")
            return
        opts = self.options()
        rows = [self.rows[rid] for rid in self._row_ids()]
        if not rows or not self._outputs_ok(rows, out_dir, opts) \
                or not self._confirm_overwrite(rows, out_dir, opts) \
                or not self._confirm_memory(rows, opts):
            return
        self._launch(rows, out_dir, opts)

    def _meshes_ready(self):
        """True if the loaded meshes match the files on disk; otherwise (re)loads them first."""
        token = self._mesh_token()
        if token is None:
            self.warn("Choose the source and target FBX files first.")
            return False
        if token == self.mesh_token:
            if self.transfer is None:
                self.warn(self.mesh_error or "The meshes couldn't be loaded.")
            return self.transfer is not None
        self.run_after_load = True                # start() runs again once loaded
        self._start_mesh_load()
        return False

    def _outputs_ok(self, rows, out_dir, opts):
        """Refuse duplicate output names and overwriting the input textures."""
        names = [r.output_name(opts.sixteen_bit) for r in rows]
        lowered = [n.lower() for n in names]
        dupes = sorted({n for n in lowered if lowered.count(n) > 1})
        if dupes:
            self.warn("These textures would be saved under the same name:\n\n" + "\n".join(dupes)
                      + "\n\nRemove one of each, or rename the files or outputs.")
            return False
        clash = [n for r, n in zip(rows, names, strict=True)
                 if any(_norm(os.path.join(out_dir, n)) == _norm(t.path) for t in r.textures.values())]
        if clash:
            self.warn("The output folder is the folder the textures are in, so the originals would "
                      "be overwritten:\n\n" + "\n".join(clash) + "\n\nChoose a different output folder.")
            return False
        return True

    def _confirm_overwrite(self, rows, out_dir, opts):
        existing = [r.output_name(opts.sixteen_bit) for r in rows
                    if os.path.exists(os.path.join(out_dir, r.output_name(opts.sixteen_bit)))]
        if not existing:
            return True
        shown = "\n".join(existing[:12]) + ("\n…" if len(existing) > 12 else "")
        return self.ask(f"{len(existing)} file(s) already exist in\n{out_dir}\n\n{shown}\n\n"
                        "Overwrite them?")

    def _confirm_memory(self, rows, opts):
        free = available_memory()
        if not free:
            return True
        need = max(estimate_memory(r.width, r.height, 4, r.kind, opts) for r in rows)
        if need <= MEMORY_HEADROOM * free:
            return True
        return self.ask(f"The largest texture needs about {need / 2**30:.1f} GB of memory, but only "
                        f"{free / 2**30:.1f} GB is free. It may fail or slow the computer down.\n\n"
                        "Continue anyway?")

    def _launch(self, rows, out_dir, opts):
        for row in rows:
            row.result_path = ""
            row.stage = ""
            for texture in row.textures.values():
                try:
                    texture.mtime = os.path.getmtime(texture.path)   # the version this transfer reads
                except OSError:
                    pass                                  # the worker reports the missing file
            self._set_status(row, Status.WAITING)
        jobs = [Job(r.id, tiled=r.to_map()) if r.tiled else Job(r.id, r.path, r.kind) for r in rows]
        self.worker = TransferWorker(self.bridge, self.transfer, jobs, out_dir, opts)
        self.cancel_btn.setText("Cancel")
        self._update_enabled()
        self.log(f"Transferring {len(jobs)} {'output' if self.tiled else 'texture'}(s) to {out_dir}")
        self.save_settings()
        self.worker.start()

    def cancel(self):
        """Stop the running batch after the current step."""
        if self.running:
            self.worker.cancel()
            self.cancel_btn.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, row_id, fraction, message):
        row = self.rows.get(row_id)
        if row:
            if message:
                row.stage = message
            self._set_status(row, Status.RUNNING, fraction, f"{row.stage or 'Working'}  %p%")

    def _on_row_done(self, row_id, result, seconds):
        row = self.rows.get(row_id)
        if not row:
            return
        row.result_path = result.path
        tooltip = "\n".join([result.path, f"UV coverage {100 * result.coverage:.1f}%"] + result.notes)
        self._set_status(row, Status.DONE, 1.0, tooltip=tooltip)
        name = row.label
        self.log(f"{name} → {result.path}  ({'; '.join(result.notes + [f'{seconds:.1f}s'])})", GREEN)
        for note in result.notes:
            if "precision" in note or "couldn't" in note:
                self.log(f"{name}: {note}", AMBER)
        selected = self._row_ids(selected_only=True)
        if selected and selected[0] == row_id:
            self._refresh_preview()

    def _on_row_failed(self, row_id, message, cancelled):
        row = self.rows.get(row_id)
        if not row:
            return
        name = row.label
        if cancelled:
            self._set_status(row, Status.CANCELLED)
            self.log(f"{name}: cancelled", AMBER)
        else:
            self._set_status(row, Status.FAILED, 1.0, tooltip=message)
            self.log(f"{name}: {message}", RED)

    def _on_batch_done(self, cancelled):
        self.worker = None
        for row in self.rows.values():
            if row.status == Status.WAITING:
                self._set_status(row, Status.SKIPPED if cancelled else Status.READY)
        self.cancel_btn.setText("Cancel")
        self._update_enabled()
        done = sum(r.status == Status.DONE for r in self.rows.values())
        failed = sum(r.status == Status.FAILED for r in self.rows.values())
        message = f"Finished: {done} done" + (f", {failed} failed" if failed else "") \
            + (" (cancelled)" if cancelled else "")
        self.log(message, RED if failed else AMBER if cancelled else GREEN)
        self.statusBar().showMessage(message, 8000)

    # -- window events --------------------------------------------------------------

    def dragEnterEvent(self, e):
        if any(is_fbx(p) or is_image(p) or is_setup(p) for p in local_paths(e.mimeData())):
            e.acceptProposedAction()

    def dropEvent(self, e):
        """A setup file opens; FBX files fill the empty mesh slot (or replace the target); images are added."""
        if self.running:
            return
        paths = local_paths(e.mimeData())
        setups = [p for p in paths if is_setup(p)]
        if setups:
            self.open_setup(setups[0])
            e.acceptProposedAction()
            return
        for path in (p for p in paths if is_fbx(p)):
            field = self.src_field if not self.src_field.path() else self.dst_field
            field.set_path(path, user=True)
        self.add_textures([p for p in paths if is_image(p)])
        e.acceptProposedAction()

    def closeEvent(self, e):
        if self.running:
            if not self.ask("A transfer is running. Stop it and quit?"):
                e.ignore()
                return
            self._stop_worker()
        self.save_settings()
        e.accept()

    def _stop_worker(self):
        """Cancel the batch and wait for the current texture to stop, so no file is left half-written."""
        self.worker.cancel()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.worker.join(CLOSE_WAIT_SECONDS)
        finally:
            QApplication.restoreOverrideCursor()
