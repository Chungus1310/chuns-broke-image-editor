import logging
from PyQt5.QtWidgets import (QDialog, QLabel, QSlider, QPushButton,
                           QGridLayout, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt
from processing.image_functions import ImageProcessor

class AdjustDialog(QDialog):
    def __init__(self, parent, image):
        super().__init__(parent)
        self.setWindowTitle("Adjust Image")
        self.setGeometry(200, 200, 400, 300)
        self.original_image = image
        self.modified_image = image.copy()
        self.image_processor = ImageProcessor()
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(34, 40, 49, 0.92);
                border: 2px solid #00ADB5;
                border-radius: 10px;
            }
            QLabel {
                color: #EEEEEE;
                font-size: 12px;
                padding: 8px;
                background-color: rgba(57, 62, 70, 0.75);
                border-radius: 6px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00ADB5;
                background: rgba(57, 62, 70, 0.75);
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
            QPushButton {
                background-color: #393E46;
                color: #EEEEEE;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                min-width: 80px;
                margin: 4px;
            }
            QPushButton:hover {
                background-color: #00ADB5;
                border: 1px solid #EEEEEE;
            }
            QPushButton:pressed {
                background-color: #008B92;
            }
        """)
        layout = QGridLayout(self)
        layout.addWidget(QLabel("Brightness"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.preview_adjustment)
        layout.addWidget(self.brightness_slider, 0, 1)
        layout.addWidget(QLabel("Contrast"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.preview_adjustment)
        layout.addWidget(self.contrast_slider, 1, 1)
        layout.addWidget(QLabel("Saturation"), 2, 0)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setMinimum(-100)
        self.saturation_slider.setMaximum(100)
        self.saturation_slider.setValue(0)
        self.saturation_slider.valueChanged.connect(self.preview_adjustment)
        layout.addWidget(self.saturation_slider, 2, 1)
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_adjustments)
        button_layout.addWidget(apply_button)
        layout.addLayout(button_layout, 3, 0, 1, 2)

    def apply_adjustments(self):
        try:
            temp_image = self.original_image.copy()
            brightness = self.brightness_slider.value()
            contrast = self.contrast_slider.value()
            saturation = self.saturation_slider.value()
            if brightness != 0:
                temp_image = self.image_processor.adjust_brightness(temp_image, brightness)
            if contrast != 0:
                temp_image = self.image_processor.adjust_contrast(temp_image, contrast)
            if saturation != 0:
                temp_image = self.image_processor.adjust_saturation(temp_image, saturation)
            self.modified_image = temp_image
            self.parent().modified_image = self.modified_image
            self.parent().display_image()
            self.accept()
        except Exception as e:
            logging.error(f"Error applying adjustments: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error applying adjustments: {e}")

    def preview_adjustment(self):
        try:
            temp_image = self.original_image.copy()
            brightness = self.brightness_slider.value()
            contrast = self.contrast_slider.value()
            saturation = self.saturation_slider.value()
            if brightness != 0:
                temp_image = self.image_processor.adjust_brightness(temp_image, brightness)
            if contrast != 0:
                temp_image = self.image_processor.adjust_contrast(temp_image, contrast)
            if saturation != 0:
                temp_image = self.image_processor.adjust_saturation(temp_image, saturation)
            self.parent().modified_image = temp_image
            self.parent().display_image()
            logging.info("Previewed adjustments.")
        except Exception as e:
            logging.error(f"Error in preview: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error in preview: {e}")