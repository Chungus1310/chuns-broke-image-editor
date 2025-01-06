import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton,
                           QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, QMenuBar,
                           QMenu, QStatusBar, QAction, QFrame, QSplitter, QMessageBox, QApplication, QInputDialog, QSlider, QToolTip, QColorDialog)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QCursor, QIcon
from PyQt5.QtCore import Qt, QSize, QRect, QPoint, pyqtSignal, QObject, QThread
from gui.adjust_dialog import AdjustDialog
from gui.toolbar import ImageEditorToolBar
from processing.image_functions import ImageProcessor
from processing.background_removal import BackgroundRemovalThread
from gui.resize_dialog import ResizeDialog
from gui.segment_dialog import SegmentDialog          
import logging
from PIL import Image, ImageDraw
from config import ICON_PATHS

class ColorPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.color = (0, 0, 0, 255)
        self.setMinimumSize(30, 30)
        self.setMaximumSize(30, 30)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(*self.color))
        painter.setPen(QPen(Qt.gray))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def setColor(self, color):
        self.color = color
        self.update()

class UpscaleWorker(QObject):
    finished = pyqtSignal(Image.Image)
    progress = pyqtSignal(float)
    
    def __init__(self, image, image_processor):
        super().__init__()
        self.image = image
        self.image_processor = image_processor
    
    def run(self):
        try:
            upscaled_image = self.image_processor.upscale_image(self.image)
            self.finished.emit(upscaled_image)
        except Exception as e:
            logging.error(f"Upscaling failed: {e}")
            self.finished.emit(self.image)

class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        logging.info("Initializing ImageEditor...")
        
        # Set application icon
        if os.path.exists(ICON_PATHS['app_icon']):
            self.setWindowIcon(QIcon(ICON_PATHS['app_icon']))
            
        # Add logo to title bar
        if os.path.exists(ICON_PATHS['app_logo']):
            app = QApplication.instance()
            app.setWindowIcon(QIcon(ICON_PATHS['app_logo']))
        
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        self.setWindowState(Qt.WindowMaximized)
        self.setMinimumSize(800, 600)
        self.setMaximumSize(screen_size)
        self.setWindowTitle("Chun's Broke Bitch Image Editor")
        
        # Define tools dictionary before setting up the panel
        self.tools = {
            "Select": ("🔲", self.select_tool),
            "Move": ("↖", self.move_tool),
            "Brush": ("🖌", self.brush_tool),
            "Eraser": ("⌫", self.eraser_tool),
            "Text": ("T", self.text_tool),
            "Crop": ("✂", self.crop_tool),
            "Fill": ("🪣", self.fill_tool),
            "Eyedrop": ("👁", self.eyedrop_tool),
            "Segment": ("✂️", self.segment_object),
            "Sharpen": ("🔧", self.sharpen_tool)
        }
        
        # Rest of initialization
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, '..', 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: rgba(34, 40, 49, 0.92);  /* #222831 with transparency */
                color: #EEEEEE;
            }
            QMenuBar {
                background-color: rgba(57, 62, 70, 0.85);  /* #393E46 with transparency */
                color: #EEEEEE;
                border: none;
                padding: 4px;
            }
            QMenuBar::item:selected {
                background-color: #00ADB5;
                border-radius: 4px;
            }
            QMenu {
                background-color: rgba(57, 62, 70, 0.92);
                color: #EEEEEE;
                border: 1px solid #00ADB5;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #00ADB5;
                border-radius: 4px;
            }
            QToolBar {
                background-color: rgba(34, 40, 49, 0.85);
                border: 1px solid #393E46;
                border-radius: 8px;
                spacing: 4px;
                padding: 4px;
            }
            QPushButton {
                background-color: rgba(57, 62, 70, 0.92);
                color: #EEEEEE;
                border: none;
                padding: 8px;
                border-radius: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #00ADB5;
                border: 1px solid #EEEEEE;
            }
            QPushButton:pressed {
                background-color: #008B92;
            }
            QFrame#toolPanel, QFrame#propertyPanel {
                background-color: rgba(57, 62, 70, 0.75);
                border: 1px solid #00ADB5;
                border-radius: 8px;
                padding: 4px;
            }
            QLabel#workspace {
                background-color: rgba(34, 40, 49, 0.92);
                border: 2px solid #393E46;
                border-radius: 8px;
            }
            QStatusBar {
                background-color: rgba(57, 62, 70, 0.85);
                color: #EEEEEE;
                border-top: 1px solid #00ADB5;
            }
            QSplitter::handle {
                background-color: #393E46;
                margin: 2px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00ADB5;
                background: rgba(34, 40, 49, 0.92);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ADB5;
                border: 1px solid #EEEEEE;
                width: 18px;
                margin: -4px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #008B92;
            }
            QToolTip {
                background-color: rgba(34, 40, 49, 0.95);
                color: #EEEEEE;
                border: 1px solid #00ADB5;
                border-radius: 4px;
                padding: 4px;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #222831;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #393E46;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #00ADB5;
            }
            QLineEdit {
                background-color: rgba(34, 40, 49, 0.92);
                border: 1px solid #393E46;
                border-radius: 4px;
                padding: 4px;
                color: #EEEEEE;
            }
            QLineEdit:focus {
                border: 1px solid #00ADB5;
            }
        """)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        splitter = QSplitter(Qt.Horizontal)
        self.tools_panel = QFrame()
        self.tools_panel.setObjectName("toolPanel")
        self.tools_panel.setMinimumWidth(80)  # Increase minimum width slightly
        self.tools_panel.setMaximumWidth(100)  # Increase maximum width slightly
        tools_layout = QVBoxLayout(self.tools_panel)
        self.color_preview = ColorPreviewWidget()
        self.fill_tolerance = QSlider(Qt.Horizontal)
        self.fill_tolerance.setRange(0, 100)
        self.fill_tolerance.setValue(32)
        self.fill_tolerance.setToolTip("Fill Tolerance")
        self.setup_tools_panel(tools_layout)
        splitter.addWidget(self.tools_panel)
        workspace_container = QWidget()
        workspace_layout = QVBoxLayout(workspace_container)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar = ImageEditorToolBar(self)
        workspace_layout.addWidget(self.toolbar)
        self.image_label = QLabel()
        self.image_label.setObjectName("workspace")
        self.image_label.setAlignment(Qt.AlignCenter)
        workspace_layout.addWidget(self.image_label)
        splitter.addWidget(workspace_container)
        self.properties_panel = QFrame()
        self.properties_panel.setObjectName("propertyPanel")
        self.properties_panel.setMinimumWidth(200)
        self.properties_panel.setMaximumWidth(300)
        properties_layout = QVBoxLayout(self.properties_panel)
        self.setup_properties_panel(properties_layout)
        splitter.addWidget(self.properties_panel)
        main_layout.addWidget(splitter)
        self.setup_menu()
        self.setup_statusbar()
        self.current_image = None
        self.modified_image = None
        self.image_processor = ImageProcessor()
        self.zoom_level = 100
        self.undo_stack = []
        self.redo_stack = []
        self.selection_mode = False
        self.last_click_pos = None
        self.text_selected = False
        self.current_tool = None
        self.drawing = False
        self.last_point = None
        self.brush_color = (0, 0, 0, 255)
        self.brush_size = 5
        self.crop_rect = None
        self.move_start = None

    def setup_tools_panel(self, layout):
        # Update layout settings
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        color_layout = QHBoxLayout()
        color_layout.setAlignment(Qt.AlignLeft)
        color_layout.addWidget(QLabel("Color:"))
        color_layout.addWidget(self.color_preview)
        layout.addLayout(color_layout)

        tolerance_layout = QHBoxLayout()
        tolerance_layout.setAlignment(Qt.AlignLeft)
        tolerance_layout.addWidget(QLabel("Fill Tolerance:"))
        tolerance_layout.addWidget(self.fill_tolerance)
        layout.addLayout(tolerance_layout)

        color_picker_btn = QPushButton("Pick Color")
        color_picker_btn.setFixedWidth(70)
        color_picker_btn.clicked.connect(self.pick_color)
        layout.addWidget(color_picker_btn)
        
        # Update tool buttons setup
        for name, (icon, func) in self.tools.items():
            btn = QPushButton(icon)
            btn.setFixedSize(32, 32)  # Slightly smaller buttons
            btn.setToolTip(name)
            btn.clicked.connect(func)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: center;
                    padding: 2px;
                    margin: 2px;
                }
            """)
            layout.addWidget(btn, 0, Qt.AlignLeft)  # Align each button to the left
        
        layout.addStretch()

    def pick_color(self):
        dialog = QColorDialog(QColor(*self.brush_color), self)
        chosen = dialog.getColor()
        if chosen.isValid():
            r, g, b, a = chosen.getRgb()
            self.brush_color = (r, g, b, a)
            self.color_preview.setColor(self.brush_color)

    def setup_properties_panel(self, layout):
        header = QLabel("Adjustments")
        header.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        adjustments = [
            ("Adjust Image", self.open_adjust_dialog),
            ("Resize", self.resize_image),
            ("Upscale", self.upscale_image),
            ("Rotate", self.rotate_image),
            ("Flip H", self.flip_horizontal),
            ("Flip V", self.flip_vertical)
        ]
        
        for name, func in adjustments:
            btn = QPushButton(name)
            btn.clicked.connect(func)
            layout.addWidget(btn)
            
        layout.addWidget(QLabel("Filters"))
        filters = [
            ("Grayscale", self.apply_grayscale),
            ("Blur", self.apply_blur),
            ("Sharpen", self.apply_sharpen),
            ("Invert Colors", self.apply_invert_colors),
            ("Remove Background", self.remove_background)
        ]
        
        for name, func in filters:
            btn = QPushButton(name)
            btn.clicked.connect(func)
            layout.addWidget(btn)

        layout.addStretch()

    def setup_menu(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        file_menu = QMenu("File", self)
        file_menu.addAction("Export").triggered.connect(self.export_image)
        file_menu.addSeparator()
        file_menu.addAction("Exit").triggered.connect(self.close)
        help_menu = QMenu("Help", self)
        help_menu.addAction("Documentation")
        help_menu.addAction("About")
        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(help_menu)

    def setup_statusbar(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("Ready")

    def open_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")

        if file_path:
            try:
                self.current_image = self.image_processor.open_image(file_path)
                self.modified_image = self.current_image.copy()
                self.display_image()
                logging.info(f"Opened image: {file_path}")
            except Exception as e:
                logging.error(f"Error opening image: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Error opening image: {e}")

    def save_image(self):
        if self.modified_image:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
            if file_path:
                try:
                    self.image_processor.save_image(self.modified_image, file_path)
                    logging.info(f"Saved image: {file_path}")
                except Exception as e:
                    logging.error(f"Error saving image: {e}", exc_info=True)
                    QMessageBox.critical(self, "Error", f"Error saving image: {e}")

    def display_image(self):
        if self.modified_image:
            try:
                workspace_size = self.image_label.size()
                image = self.modified_image.convert("RGBA")
                scale_w = workspace_size.width() / image.width
                scale_h = workspace_size.height() / image.height
                scale = min(scale_w, scale_h) * 0.9
                new_width = int(image.width * scale * (self.zoom_level / 100))
                new_height = int(image.height * scale * (self.zoom_level / 100))
                data = image.tobytes("raw", "RGBA")
                qim = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qim)
                scaled_pixmap = pixmap.scaled(
                    new_width, new_height,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
                logging.debug(f"Displayed image with zoom level: {self.zoom_level}%")
            except Exception as e:
                logging.error(f"Error displaying image: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Error displaying image: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.modified_image:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(50, self.display_image)

    def open_adjust_dialog(self):
        if self.modified_image:
            adjust_dialog = AdjustDialog(self, self.modified_image)
            if adjust_dialog.exec_() == adjust_dialog.Accepted:
                self.modified_image = adjust_dialog.modified_image
                self.display_image()

    def apply_grayscale(self):
        if self.modified_image:
            self.save_current_state()
            self.modified_image = self.image_processor.apply_grayscale(self.modified_image)
            self.display_image()

    def apply_blur(self):
        if self.modified_image:
            self.save_current_state()
            self.modified_image = self.image_processor.apply_blur(self.modified_image)
            self.display_image()
            self.modified_image = self.image_processor.apply_blur(self.modified_image)
            self.display_image()

    def apply_sharpen(self):
        if self.modified_image:
            self.save_current_state()
            self.modified_image = self.image_processor.apply_sharpen(self.modified_image)
            self.display_image()
            logging.info("Applied sharpen filter.")

    def apply_invert_colors(self):
        if self.modified_image:
            self.save_current_state()
            self.modified_image = self.image_processor.apply_invert_colors(self.modified_image)
            self.display_image()

    def remove_background(self):
        if self.modified_image:
            try:
                self.save_current_state()
                self.statusBar().showMessage("Removing background... Please wait.")
                QApplication.processEvents()
                self.bg_thread = BackgroundRemovalThread(self.modified_image, self)
                self.bg_thread.finished_signal.connect(self.update_image_after_bg_removal)
                self.bg_thread.start()
            except Exception as e:
                logging.error(f"Error initiating background removal: {e}")
                QMessageBox.critical(self, "Error", "Failed to start background removal")

    def update_image_after_bg_removal(self, result_image):
        if result_image:
            self.modified_image = result_image
            self.display_image()
            self.statusBar().showMessage("Background removed successfully.")
        else:
            QMessageBox.critical(self, "Error", "Background removal failed")
        if hasattr(self, 'bg_thread'):
            self.bg_thread.deleteLater()
            del self.bg_thread

    def new_image(self):
        if self.modified_image:
            self.save_current_state()
        self.current_image = Image.new('RGBA', (800, 600), (255, 255, 255, 0))
        self.modified_image = self.current_image.copy()
        self.display_image()

    def save_current_state(self):
        if self.modified_image:
            self.undo_stack.append(self.modified_image.copy())
            self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.modified_image.copy())
            self.modified_image = self.undo_stack.pop()
            self.display_image()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.modified_image.copy())
            self.modified_image = self.redo_stack.pop()
            self.display_image()

    def zoom_in(self):
        self.zoom_level = min(self.zoom_level + 10, 200)
        self.display_image()

    def zoom_out(self):
        self.zoom_level = max(self.zoom_level - 10, 20)
        self.display_image()

    def fit_to_screen(self):
        if not self.modified_image:
            return
        workspace = self.image_label.size()
        image_size = self.modified_image.size
        scale_w = workspace.width() / image_size[0]
        scale_h = workspace.height() / image_size[1]
        scale = min(scale_w, scale_h) * 0.9
        self.zoom_level = scale * 100
        self.display_image()
        logging.info(f"Image fit to screen with zoom level: {self.zoom_level}%")

    def actual_size(self):
        self.zoom_level = 100
        self.display_image()

    def save_image_as(self):
        if self.modified_image:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Save Image As", "", 
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*.*)")
            if file_path:
                try:
                    self.image_processor.save_image(self.modified_image, file_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def export_image(self):
        if self.modified_image:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Export Image", "", 
                "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp);;TIFF Files (*.tiff)")
            if file_path:
                try:
                    self.image_processor.save_image(self.modified_image, file_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to export image: {str(e)}")

    def rotate_image(self):
        if self.modified_image:
            self.save_current_state()
            self.modified_image = self.modified_image.rotate(90, expand=True)
            self.display_image()

    def flip_horizontal(self):
        if self.modified_image:
            self.save_current_state()
            self.modified_image = self.modified_image.transpose(Image.FLIP_LEFT_RIGHT)
            self.display_image()

    def flip_vertical(self):
        if self.modified_image:
            self.save_current_state()
            self.modified_image = self.modified_image.transpose(Image.FLIP_TOP_BOTTOM)
            self.display_image()

    def resize_image(self):
        if self.modified_image:
            resize_dialog = ResizeDialog(self, self.modified_image)
            if resize_dialog.exec_() == resize_dialog.Accepted:
                new_size = resize_dialog.get_new_size()
                try:
                    self.save_current_state()
                    self.modified_image = self.image_processor.resize_image(self.modified_image, new_size)
                    self.display_image()
                    self.statusBar().showMessage(f"Image resized to {new_size[0]}x{new_size[1]}")
                    logging.info(f"Resized image to: {new_size}")
                except Exception as e:
                    logging.error(f"Failed to resize image: {e}", exc_info=True)
                    QMessageBox.critical(self, "Error", f"Failed to resize image: {str(e)}")

    def upscale_image(self):
        if self.modified_image:
            self.statusBar().showMessage("Upscaling image...")
            self.upscale_thread = QThread()
            self.upscale_worker = UpscaleWorker(self.modified_image, self.image_processor)
            self.upscale_worker.moveToThread(self.upscale_thread)
            self.upscale_thread.started.connect(self.upscale_worker.run)
            self.upscale_worker.finished.connect(self.on_upscale_finished)
            self.upscale_worker.finished.connect(self.upscale_thread.quit)
            self.upscale_worker.finished.connect(self.upscale_worker.deleteLater)
            self.upscale_thread.finished.connect(self.upscale_thread.deleteLater)
            self.upscale_thread.start()

    def on_upscale_finished(self, upscaled_image):
        self.modified_image = upscaled_image
        self.display_image()
        self.statusBar().showMessage("Image upscaled successfully.")
        logging.info("Upscaling completed.")
                
    def segment_object(self):
        if self.modified_image:
            dialog = SegmentDialog(self, self.modified_image)
            dialog.exec_()
        else:
            QMessageBox.warning(self, "No Image", "Please open an image first.")

    def mousePressEvent(self, event):
        if not self.modified_image:
            return
        if self.current_tool == "Text":
            pos = self.get_image_coordinates(event.pos())
            if pos and event.button() == Qt.LeftButton:
                text, ok = QInputDialog.getText(self, "Add Text", "Enter text:")
                if ok and text.strip():
                    self.save_current_state()
                    self.modified_image = self.image_processor.apply_text(
                        self.modified_image, text, pos, (0,0,0), 24
                    )
                    self.display_image()
                    self.text_selected = True
            return
        if self.selection_mode:
            pos = self.get_image_coordinates(event.pos())
            if pos:
                x, y = pos
                self.statusBar().showMessage("Segmenting object... Please wait.")
                QApplication.processEvents()
                segmented = self.image_processor.segment_object_at_point(
                    self.modified_image, (x, y))
                if segmented:
                    self.save_current_state()
                    self.modified_image = segmented
                    self.display_image()
                    self.statusBar().showMessage("Object segmented successfully.")
                else:
                    self.statusBar().showMessage("No object found at this position. Click elsewhere to try again.")
            return
        pos = self.get_image_coordinates(event.pos())
        if pos:
            x, y = pos
            if self.current_tool == "Move":
                self.move_start = (x, y)
                self.setCursor(Qt.ClosedHandCursor)
            elif self.selection_mode:
                self.statusBar().showMessage("Segmenting object... Please wait.")
                QApplication.processEvents()
                segmented = self.image_processor.segment_object_at_point(
                    self.modified_image, (x, y))
                if segmented:
                    self.save_current_state()
                    self.modified_image = segmented
                    self.display_image()
                    self.statusBar().showMessage("Object segmented successfully.")
                else:
                    self.statusBar().showMessage("No object found at this position.")
                self.selection_mode = False
                self.setCursor(Qt.ArrowCursor)
            if self.current_tool == "Move" and self.text_selected:
                self.move_start = self.get_image_coordinates(event.pos())
                self.setCursor(Qt.ClosedHandCursor)
            if self.current_tool == "Eyedrop":
                self.apply_eyedropper(pos)
            if self.current_tool == "Fill":
                self.save_current_state()
                filled = self.image_processor.apply_fill(
                    self.modified_image, 
                    pos, 
                    self.brush_color,
                    self.fill_tolerance.value()
                )
                if filled is not None:
                    self.modified_image = filled
                    self.display_image()
            if self.current_tool == "Brush":
                self.drawing = True
                self.last_point = pos
            elif self.current_tool == "Eraser":
                self.drawing = True
                self.last_point = pos
            if self.current_tool == "Crop" and pos:
                self.crop_rect = [pos[0], pos[1], pos[0], pos[1]]

    def get_image_coordinates(self, pos):
        label_pos = self.image_label.mapFrom(self, pos)
        img_rect = self.get_image_rect()
        
        if img_rect.contains(label_pos):
            x = int((label_pos.x() - img_rect.x()) * (self.modified_image.width / img_rect.width()))
            y = int((label_pos.y() - img_rect.y()) * (self.modified_image.height / img_rect.height()))
            return (x, y)
        return None

    def mouseMoveEvent(self, event):
        if not self.current_tool or not self.modified_image:
            return
            
        pos = self.image_label.mapFromParent(event.pos())
        x = int(pos.x() * self.modified_image.width / self.image_label.width())
        y = int(pos.y() * self.modified_image.height / self.image_label.height())
        
        if self.drawing and self.current_tool == "Brush":
            if self.last_point:
                self.modified_image = self.image_processor.apply_brush(
                    self.modified_image, [self.last_point, (x, y)],
                    self.brush_color, self.brush_size)
                self.display_image()
            self.last_point = (x, y)
        elif self.drawing and self.current_tool == "Eraser":
            if self.last_point:
                self.modified_image = self.image_processor.apply_eraser(
                    self.modified_image, [self.last_point, (x, y)],
                    self.brush_size)
                self.display_image()
            self.last_point = (x, y)
        elif self.move_start and self.current_tool == "Move":
            pos = self.get_image_coordinates(event.pos())
            if pos:
                dx = pos[0] - self.move_start[0]
                dy = pos[1] - self.move_start[1]
                self.move_image(dx, dy)
                self.move_start = pos
        elif self.crop_rect and self.current_tool == "Crop":
            self.crop_rect[2], self.crop_rect[3] = x, y
            self.update_crop_overlay()

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.last_point = None
            self.save_current_state()
        elif self.move_start:
            self.move_start = None
            self.save_current_state()
        elif self.crop_rect and self.current_tool == "Crop":
            self.apply_crop()
            self.crop_rect = None
            self.save_current_state()
        
        if self.current_tool != "Text":
            self.text_selected = False

    def select_tool(self): self.activate_tool("Select")
    def move_tool(self): self.activate_tool("Move")
    def brush_tool(self): 
        self.activate_tool("Brush")
        self.brush_color = self.get_current_color()
        self.brush_size = self.get_brush_size()
    def eraser_tool(self): 
        self.activate_tool("Eraser")
        self.brush_size = self.get_eraser_size()
    def text_tool(self): 
        self.activate_tool("Text")
        self.text_selected = False
    def crop_tool(self): self.activate_tool("Crop")
    def fill_tool(self): self.activate_tool("Fill")
    def eyedrop_tool(self): self.activate_tool("Eyedrop")
    def zoom_tool(self): self.activate_tool("Zoom")
    def sharpen_tool(self):
        self.activate_tool("Sharpen")
        self.apply_sharpen()

    def activate_tool(self, tool_name):
        self.current_tool = tool_name
        if tool_name == "Brush":
            self.setCursor(Qt.CrossCursor)
        elif tool_name == "Eraser":
            self.setCursor(Qt.CrossCursor)
        elif tool_name == "Move":
            self.setCursor(Qt.OpenHandCursor)
        elif tool_name == "Crop":
            self.setCursor(Qt.CrossCursor)
        elif tool_name == "Eyedrop":
            self.setCursor(Qt.ArrowCursor)
        elif tool_name == "Fill":
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.statusBar().showMessage(f"Active tool: {tool_name}")

    def move_image(self, dx, dy):
        if self.modified_image:
            self.save_current_state()
            self.modified_image = self.image_processor.move_image_by(
                self.modified_image, dx, dy, sensitivity=0.25
            )
            self.display_image()
            logging.info(f"Moved image by ({dx, dy}).")

    def apply_crop(self):
        if self.crop_rect and self.modified_image:
            x1, y1, x2, y2 = self.crop_rect
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            try:
                self.modified_image = self.image_processor.apply_crop(
                    self.modified_image, (x1, y1, x2, y2))
                self.display_image()
            except Exception as e:
                logging.error(f"Error applying crop: {e}")

    def update_crop_overlay(self):
        pass

    def get_image_rect(self):
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return QRect()
        scaled_size = pixmap.size()
        label_size = self.image_label.size()
        x = (label_size.width() - scaled_size.width()) // 2
        y = (label_size.height() - scaled_size.height()) // 2
        return QRect(x, y, scaled_size.width(), scaled_size.height())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.selection_mode:
                self.selection_mode = False
                self.setCursor(Qt.ArrowCursor)
                self.statusBar().showMessage("Segmentation cancelled.")
        event.accept()

    def get_current_color(self):
        return self.brush_color

    def get_brush_size(self):
        return 5

    def get_eraser_size(self):
        return 10

    def apply_brush(self, points, color, size):
        self.modified_image = self.image_processor.apply_brush(self.modified_image, points, color, size)
        qimage = self.pil2qimage(self.modified_image)
        self.display_image()
        return qimage

    def apply_eraser(self, points, size):
        self.modified_image = self.image_processor.apply_eraser(self.modified_image, points, size)
        self.display_image()

    def set_modified_image(self, image):
        self.modified_image = image
        self.display_image()
        self.save_current_state()
        logging.info("Modified image updated.")

    def pil2qimage(self, im):
        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
            data = im.tobytes("raw", "RGB")
        elif im.mode == "RGBA":
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (b, g, r, a))
            data = im.tobytes("raw", "RGBA")
        else:
            im = im.convert("RGBA")
            data = im.tobytes("raw", "RGBA")
        qimage = QImage(data, im.size[0], im.size[1], QImage.Format_RGBA8888)
        return qimage

    def apply_eyedropper(self, pos):
        if self.modified_image:
            color = self.image_processor.apply_eyedropper(self.modified_image, pos)
            self.brush_color = color
            self.color_preview.setColor(color)
            QToolTip.showText(
                QCursor.pos(),
                f"RGB: {color[:3]}\nAlpha: {color[3]}",
                self,
                QRect(QPoint(), QSize(1, 1)),
                2000
            )