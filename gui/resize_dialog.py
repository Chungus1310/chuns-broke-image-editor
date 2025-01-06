import logging
from PyQt5.QtWidgets import (QDialog, QLabel, QLineEdit, QPushButton,
                             QGridLayout, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import Qt

class ResizeDialog(QDialog):
    def __init__(self, parent, image):
        super().__init__(parent)
        self.setWindowTitle("Resize Image")
        self.image = image
        self.new_width = 0
        self.new_height = 0
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
                padding: 4px;
            }
            QLineEdit {
                background-color: rgba(57, 62, 70, 0.75);
                border: 1px solid #393E46;
                border-radius: 4px;
                padding: 6px;
                color: #EEEEEE;
            }
            QLineEdit:focus {
                border: 1px solid #00ADB5;
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
        layout.addWidget(QLabel("New Width:"), 0, 0)
        self.width_input = QLineEdit()
        self.width_input.setText(str(self.image.width))
        layout.addWidget(self.width_input, 0, 1)
        layout.addWidget(QLabel("New Height:"), 1, 0)
        self.height_input = QLineEdit()
        self.height_input.setText(str(self.image.height))
        layout.addWidget(self.height_input, 1, 1)
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout, 2, 0, 1, 2)

    def accept(self):
        try:
            width = int(self.width_input.text())
            height = int(self.height_input.text())
            if width <= 0 or height <= 0:
                raise ValueError("Dimensions must be positive integers.")
            self.new_width = width
            self.new_height = height
            logging.info(f"Resizing image to {self.new_width}x{self.new_height}.")
            super().accept()
        except ValueError as ve:
            logging.error(f"Invalid resize input: {ve}", exc_info=True)
            QMessageBox.critical(self, "Invalid Input", str(ve))

    def get_new_size(self):
        return (self.new_width, self.new_height)