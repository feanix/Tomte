"""The Tomte main window: pick two FBX files, add textures, transfer, preview."""

import html
import os
import shlex
import threading
import time

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QSplitter, QTableWidgetItem, QVBoxLayout, QWidget, QComboBox,
                               QFileDialog)

from ..errors import TransferError
from ..images import IMAGE_EXTENSIONS, image_info
from ..setupfile import EXTENSION as SETUP_EXTENSION, Setup, TextureEntry
from ..system import available_memory
from ..transfer import Options, classify, default_output_dir, estimate_memory
from .preview import PreviewCache
from .rows import Status, TextureRow
from .widgets import (GREY, FileField, ImageView, NoWheelComboBox, StatusCell, TextureTable,
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
        hint = QLabel("Types are guessed from file names; change them if needed.")
        hint.setStyleSheet(f"color: {GREY}")
        buttons = QHBoxLayout()
        for b in (self.add_btn, self.remove_btn, self.clear_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(hint)
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
        self.table.filesDropped.connect(self.add_textures)
        self.table.itemSelectionChanged.connect(self._refresh_preview)
        self.table.itemSelectionChanged.connect(self._update_enabled)
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
        self._set_wireframes(load.src_wireframe, load.dst_wireframe)
        self._update_enabled()
        if self.run_after_load and load.transfer is not None:
            self.run_after_load = False
            self.start()

    def _set_wireframes(self, src, dst):
        self.before_view.set_wireframe(src)
        self.after_view.set_wireframe(dst)

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

    def _browse_textures(self):
        start = self.settings.value("last_tex_dir", "") or os.path.dirname(self.src_field.path())
        paths, _ = QFileDialog.getOpenFileNames(self, "Add textures", start, IMAGE_FILTER)
        if paths:
            self.settings.setValue("last_tex_dir", os.path.dirname(paths[0]))
            self.add_textures(paths)

    def add_textures(self, paths):
        """Add image files to the list (duplicates and unreadable files are skipped)."""
        if self.running:
            return
        have = {_norm(r.path) for r in self.rows.values()}
        first_new = None
        for path in paths:
            if not is_image(path) or _norm(path) in have:
                continue
            try:
                mtime = os.path.getmtime(path)
                width, height, high_precision = image_info(path)
            except OSError as e:
                self.log(f"Couldn't read {os.path.basename(path)}: {e}", RED)
                continue
            have.add(_norm(path))
            table_row = self._add_row(path, width, height, high_precision, mtime)
            if first_new is None:
                first_new = table_row
        if first_new is not None and not self.table.selectedItems():
            self.table.selectRow(first_new)
        self._update_enabled()

    def _add_row(self, path, width, height, high_precision, mtime):
        """Create the table row and its TextureRow. Returns the table row number."""
        row_id = self.next_row_id
        self.next_row_id += 1
        r = self.table.rowCount()
        self.table.insertRow(r)

        name = QTableWidgetItem(os.path.basename(path))
        name.setData(ROW_ID, row_id)
        name.setToolTip(path)
        self.table.setItem(r, 0, name)

        type_combo = NoWheelComboBox()
        for i, (kind, label, tip) in enumerate(KIND_CHOICES):
            type_combo.addItem(label, kind)
            type_combo.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
        type_combo.setCurrentIndex([k for k, _, _ in KIND_CHOICES].index(classify(path)))
        type_combo.currentIndexChanged.connect(lambda _=0, rid=row_id: self._kind_changed(rid))
        self.table.setCellWidget(r, 1, type_combo)

        size = QTableWidgetItem(_size_text(width, height, high_precision))
        size.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(r, 2, size)

        status_cell = StatusCell()
        self.table.setCellWidget(r, 3, status_cell)
        self.rows[row_id] = TextureRow(row_id, path, width, height, high_precision,
                                       type_combo, status_cell, mtime=mtime)
        return r

    def _row_ids(self, selected_only=False):
        """Row ids in table order (optionally only selected rows)."""
        if selected_only:
            numbers = sorted({i.row() for i in self.table.selectedIndexes()})
        else:
            numbers = range(self.table.rowCount())
        return [self.table.item(r, 0).data(ROW_ID) for r in numbers]

    def _remove_selected(self):
        if self.running:
            return
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
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
        if row and row.result_path:
            row.result_path = ""
            self._set_status(row, Status.STALE)
            self._refresh_preview()

    # -- setup files ----------------------------------------------------------------

    def current_setup(self):
        """The meshes, textures and settings in the window, as a `Setup`."""
        return Setup(
            source=self.src_field.path(), target=self.dst_field.path(),
            textures=[TextureEntry(self.rows[rid].path, self.rows[rid].kind) for rid in self._row_ids()],
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
        self.add_textures([t.path for t in setup.textures])
        kinds = {_norm(t.path): t.kind for t in setup.textures}
        choices = [k for k, _, _ in KIND_CHOICES]
        for row in self.rows.values():
            row.type_combo.setCurrentIndex(choices.index(kinds[_norm(row.path)]))
        return [t.path for t in setup.textures if not is_image(t.path)]

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
        """Re-read one texture's size and format. Returns "same", "changed" or "missing".

        A finished transfer is marked out of date if its texture changed since.
        """
        try:
            mtime = os.path.getmtime(row.path)
            row.width, row.height, row.high_precision = image_info(row.path)
        except OSError as e:
            row.result_path = ""
            self._set_status(row, Status.MISSING, 1.0, tooltip=f"{row.path}\n{e}")
            return "missing"
        self.table.item(table_row, 2).setText(_size_text(row.width, row.height, row.high_precision))
        changed = mtime != row.mtime
        row.mtime = mtime
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
            if os.path.isfile(candidate) and _norm(candidate) != _norm(row.path):
                return candidate, " (existing file)"
        return "", ""

    def _refresh_preview(self):
        self.preview_request += 1
        request = self.preview_request
        selected = self._row_ids(selected_only=True)
        row = self.rows.get(selected[0]) if selected else None
        if row is None:
            self.preview_name.setText("Select a texture to preview it")
            self.before_view.set_image(None)
            self.after_view.set_image(None)
            return
        after, note = self._after_path(row)
        self.preview_name.setText(os.path.basename(row.path) + note)
        self.before_view.set_image(None, message="Loading…")
        self.after_view.set_image(None, message="Loading…" if after else "Not transferred yet")

        def work():
            self.bridge.previewReady.emit(request, (row.path, self.previews.get(row.path)),
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
            row.type_combo.setEnabled(idle)
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
                      + "\n\nRemove one of each, or rename the files.")
            return False
        clash = [n for r, n in zip(rows, names, strict=True) if _norm(os.path.join(out_dir, n)) == _norm(r.path)]
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
            try:
                row.mtime = os.path.getmtime(row.path)    # the version this transfer reads
            except OSError:
                pass                                      # the worker reports the missing file
            self._set_status(row, Status.WAITING)
        jobs = [Job(r.id, r.path, r.kind) for r in rows]
        self.worker = TransferWorker(self.bridge, self.transfer, jobs, out_dir, opts)
        self.cancel_btn.setText("Cancel")
        self._update_enabled()
        self.log(f"Transferring {len(jobs)} texture(s) to {out_dir}")
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
        name = os.path.basename(row.path)
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
        name = os.path.basename(row.path)
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
