import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import sys
import os
import urllib.request
import logging

class InteractiveSegmentation:
    def __init__(self, model_path="yolov8n-seg.pt"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, '..', 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, model_path)
        if not os.path.exists(self.model_path):
            try:
                url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
                messagebox.showinfo("Download", "Downloading YOLOv8-seg model. Please wait...")
                urllib.request.urlretrieve(url, self.model_path)
                messagebox.showinfo("Success", "Model downloaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to download model: {str(e)}")
                sys.exit(1)
        try:
            self.model = YOLO(self.model_path, task='segment')
            logging.info(f"Loaded YOLOv8-seg model from {self.model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {str(e)}")
            sys.exit(1)
        self.image = None
        self.results = None
        self.click_position = None
        self.window_name = "Interactive Segmentation"
        self.current_mask_index = None
        self.overlay_alpha = 0.3
        self.original_height = None
        self.original_width = None

    def load_image(self):
        root = tk.Tk()
        root.withdraw()
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("All files", "*.*")
                ]
            )
            if not file_path:
                return False
            self.image = cv2.imread(file_path)
            if self.image is None:
                raise ValueError("Failed to load image")
            self.original_height, self.original_width = self.image.shape[:2]
            max_dimension = 1200
            if max(self.original_height, self.original_width) > max_dimension:
                scale = max_dimension / max(self.original_height, self.original_width)
                new_width = int(self.original_width * scale)
                new_height = int(self.original_height * scale)
                self.image = cv2.resize(self.image, (new_width, new_height))
            self.results = self.model.predict(
                self.image,
                conf=0.25,
                iou=0.45,
                verbose=False
            )
            if not self.results[0].masks:
                messagebox.showwarning("Warning", "No objects detected in the image")
                return False
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
            return False

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

    def isolate_object(self, mask_index):
        if mask_index is None:
            return None
        try:
            result = self.results[0]
            b_mask = np.zeros(self.image.shape[:2], np.uint8)
            contour = result.masks.xy[mask_index].astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            isolated = np.dstack([self.image, b_mask])
            boxes = result.boxes.xyxy.cpu().numpy()
            x1, y1, x2, y2 = boxes[mask_index].astype(np.int32)
            pad_x = int(0.05 * (x2 - x1))
            pad_y = int(0.05 * (y2 - y1))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(self.image.shape[1], x2 + pad_x)
            y2 = min(self.image.shape[0], y2 + pad_y)
            isolated_cropped = isolated[y1:y2, x1:x2]
            return isolated_cropped
        except Exception as e:
            messagebox.showerror("Error", f"Error isolating object: {str(e)}")
            return None

    def process_click(self):
        if self.click_position is None:
            return
        mask_index = self.find_closest_mask(self.click_position)
        if mask_index is not None:
            isolated_obj = self.isolate_object(mask_index)
            if isolated_obj is not None:
                cv2.namedWindow("Isolated Object", cv2.WINDOW_NORMAL)
                cv2.putText(isolated_obj, "Press 's' to save, any other key to close", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Isolated Object", isolated_obj)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    try:
                        save_path = filedialog.asksaveasfilename(
                            defaultextension=".png",
                            filetypes=[("PNG files", "*.png")],
                            title="Save Isolated Object"
                        )
                        if save_path:
                            cv2.imwrite(save_path, isolated_obj)
                            messagebox.showinfo("Success", "Object saved successfully!")
                    except Exception as e:
                        messagebox.showerror("Error", f"Error saving file: {str(e)}")
                cv2.destroyWindow("Isolated Object")

    def draw_overlay(self, image):
        overlay = image.copy()
        if self.current_mask_index is not None:
            mask = np.zeros(image.shape[:2], np.uint8)
            contour = self.results[0].masks.xy[self.current_mask_index].astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            overlay[mask > 0] = [0, 255, 0]
        cv2.addWeighted(overlay, self.overlay_alpha, image, 1 - self.overlay_alpha, 0, image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Click on an object to isolate it", 
                   (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(image, "Press 'q' to quit", 
                   (10, 70), font, 1, (255, 255, 255), 2)
        return image

    def run(self):
        if not self.load_image():
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        while True:
            display_img = self.image.copy()
            display_img = self.draw_overlay(display_img)
            cv2.imshow(self.window_name, display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = InteractiveSegmentation()
    app.run()