import logging
from PyQt5.QtWidgets import (QDialog, QLabel, QPushButton,
                             QVBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from PIL import Image

class SegmentDialog(QDialog):
    def __init__(self, parent, image):
        super().__init__(parent)
        self.setWindowTitle("Segment Object")
        self.image = image
        self.window_name = "Interactive Segmentation"
        self.click_position = None
        self.current_mask_index = None
        self.overlay_alpha = 0.3
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
            QPushButton {
                background-color: #393E46;
                color: #EEEEEE;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                min-width: 100px;
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
        layout = QVBoxLayout(self)
        self.info_label = QLabel("A new window will open.\nClick on the object you want to segment.")
        layout.addWidget(self.info_label)
        
        self.start_button = QPushButton("Start Segmentation")
        self.start_button.clicked.connect(self.start_interactive_segmentation)
        layout.addWidget(self.start_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)

    def start_interactive_segmentation(self):
        try:
            cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            self.results = self.parent().image_processor.model.predict(cv_image, verbose=False)
            while True:
                display_img = cv_image.copy()
                display_img = self.draw_overlay(display_img)
                cv2.imshow(self.window_name, display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(f"Error in interactive segmentation: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", str(e))

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_position = (x, y)
            self.process_click()
        elif event == cv2.EVENT_MOUSEMOVE:
            self.current_mask_index = self.find_closest_mask((x, y))

    def find_closest_mask(self, click_pos):
        if not self.results or not self.results[0].masks:
            return None
        result = self.results[0]
        closest_mask = None
        min_distance = float('inf')
        for i, mask in enumerate(result.masks.xy):
            mask_points = np.array(mask, dtype=np.int32)
            distance = cv2.pointPolygonTest(mask_points, click_pos, True)
            if distance >= 0:
                return i
            elif abs(distance) < min_distance:
                min_distance = abs(distance)
                closest_mask = i
        return closest_mask if min_distance < 50 else None

    def process_click(self):
        if self.click_position is None:
            return
        mask_index = self.find_closest_mask(self.click_position)
        if mask_index is not None:
            isolated_obj = self.isolate_object(mask_index)
            if isolated_obj is not None:
                pil_image = Image.fromarray(cv2.cvtColor(isolated_obj, cv2.COLOR_BGRA2RGBA))
                self.parent().set_modified_image(pil_image)
                QMessageBox.information(self, "Success", "Object segmented successfully!")
                cv2.destroyAllWindows()
                self.accept()

    def draw_overlay(self, image):
        overlay = image.copy()
        if self.current_mask_index is not None:
            mask = np.zeros(image.shape[:2], np.uint8)
            contour = self.results[0].masks.xy[self.current_mask_index].astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            overlay[mask > 0] = [0, 255, 0]
        cv2.addWeighted(overlay, self.overlay_alpha, image, 1 - self.overlay_alpha, 0, image)
        cv2.putText(image, "Click on an object to segment it", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Press 'q' to cancel", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image

    def isolate_object(self, mask_index):
        try:
            cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            result = self.results[0]
            b_mask = np.zeros(cv_image.shape[:2], np.uint8)
            contour = result.masks.xy[mask_index].astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            bgra = cv2.cvtColor(cv_image, cv2.COLOR_BGR2BGRA)
            bgra[..., 3] = b_mask
            return bgra
        except Exception as e:
            logging.error(f"Error isolating object: {e}", exc_info=True)
            return None