# image_editor/processing/background_removal.py
import threading
from PIL import Image
import logging
import io
from PyQt5.QtCore import QThread, pyqtSignal
from rembg import remove

class BackgroundRemovalThread(QThread):
    finished_signal = pyqtSignal(object)

    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image

    def run(self):
        try:
            img_byte_arr = io.BytesIO()
            self.image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            output = remove(img_byte_arr, only_mask=True)  # Ensure no model downloading
            result_image = Image.open(io.BytesIO(output))
            self.finished_signal.emit(result_image)
            logging.info("Background removal completed successfully.")
        except Exception as e:
            logging.error(f"Background removal failed: {e}", exc_info=True)
            self.finished_signal.emit(None)