import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
from skimage import exposure, filters, color, morphology
import cv2
import logging
from rembg import remove, new_session
import torch
import urllib.request
import warnings
from ultralytics import YOLO  
from transformers import AutoImageProcessor, AutoModelForImageToImage
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from config import get_models_dir, MODEL_PATHS, MODEL_URLS

warnings.filterwarnings('ignore', category=FutureWarning)

def convert_rgb_to_ycbcr(img):
    img = np.array(img)
    y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
    cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    return np.stack([y, cb, cr], axis=-1)

def convert_ycbcr_to_rgb(img):
    r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
    g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
    b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                          output_padding=scale_factor-1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

class ImageProcessor:
    def __init__(self):
        self.models_dir = get_models_dir()
        self.rembg_session = new_session("u2net")
        self.yolo_weights = MODEL_PATHS['yolo']
        self.model = None
        self.model_loaded = False
        self.initialize_yolo_model()
        self.load_yolo_model()
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.sr_model = FSRCNN(scale_factor=2, num_channels=1, d=56, s=12, m=4).to(self.device)
            if os.path.exists(MODEL_PATHS['fsrcnn']):
                state_dict = torch.load(MODEL_PATHS['fsrcnn'], map_location=self.device)
                self.sr_model.load_state_dict(state_dict)
                self.sr_model.eval()
                logging.info("FSRCNN model loaded successfully")
            else:
                logging.error(f"FSRCNN weights not found at {MODEL_PATHS['fsrcnn']}")
                self.sr_model = None
        except Exception as e:
            logging.error(f"Failed to load FSRCNN model: {e}")
            self.sr_model = None

    def initialize_yolo_model(self):
        try:
            if not os.path.exists(self.yolo_weights):
                logging.error("YOLOv8 weights not found.")
            else:
                logging.info("YOLOv8 model already exists.")
        except Exception as e:
            logging.error(f"Error initializing YOLO model: {e}", exc_info=True)
    
    def load_yolo_model(self):
        if not self.model_loaded:
            try:
                self.model = YOLO(self.yolo_weights)
                self.model_loaded = True
                logging.info("YOLOv8 model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading YOLO model: {e}", exc_info=True)
        return self.model_loaded

    def open_image(self, file_path):
        return Image.open(file_path)

    def save_image(self, image, file_path):
        image.save(file_path)

    def apply_grayscale(self, image):
        gray_image = image.convert("L").convert("RGBA")
        return gray_image

    def apply_blur(self, image):
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))
        return blurred_image

    def apply_sharpen(self, image):
        sharpened_image = image.filter(ImageFilter.SHARPEN)
        return sharpened_image

    def apply_invert_colors(self, image):
        inverted_image = Image.fromarray(255 - np.array(image))
        return inverted_image

    def adjust_brightness(self, image, value):
        enhancer = ImageEnhance.Brightness(image)
        factor = 1 + (value / 100)
        bright_image = enhancer.enhance(factor)
        return bright_image

    def adjust_contrast(self, image, value):
        enhancer = ImageEnhance.Contrast(image)
        factor = 1 + (value / 100)
        contrast_image = enhancer.enhance(factor)
        return contrast_image

    def adjust_saturation(self, image, value):
        enhancer = ImageEnhance.Color(image)
        factor = 1 + (value / 100)
        saturated_image = enhancer.enhance(factor)
        return saturated_image

    def resize_image(self, image, new_size):
        resized_image = image.resize(new_size, Image.ANTIALIAS)
        return resized_image

    def _preprocess_for_esrgan(self, image):
        np_image = np.array(image)
        if len(np_image.shape) == 2:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
        elif np_image.shape[2] == 4:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
        h, w = np_image.shape[:2]
        h = (h // 4) * 4
        w = (w // 4) * 4
        np_image = np_image[:h, :w]
        tensor = tf.convert_to_tensor(np_image)
        tensor = tf.cast(tensor, tf.float32)
        tensor = tf.expand_dims(tensor, 0)
        return tensor

    def upscale_image(self, image):
        try:
            if self.sr_model is None:
                raise ValueError("FSRCNN model not initialized")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image).astype(np.float32)
            ycbcr = convert_rgb_to_ycbcr(img_array)
            y = ycbcr[..., 0]
            y /= 255.
            y_tensor = torch.from_numpy(y).to(self.device)
            y_tensor = y_tensor.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                output = self.sr_model(y_tensor).clamp(0.0, 1.0)
            output = output.cpu().numpy().squeeze() * 255.0
            output = output.clip(0, 255).astype(np.uint8)
            cb = ycbcr[..., 1]
            cr = ycbcr[..., 2]
            cb_up = cv2.resize(cb, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_CUBIC)
            cr_up = cv2.resize(cr, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_CUBIC)
            output_ycbcr = np.stack([output, cb_up, cr_up], axis=-1)
            output_rgb = convert_ycbcr_to_rgb(output_ycbcr)
            upscaled_image = Image.fromarray(output_rgb)
            logging.info("Image upscaled successfully using FSRCNN")
            return upscaled_image
        except Exception as e:
            logging.error(f"Error in FSRCNN upscaling: {e}", exc_info=True)
            return image

    def segment_object(self, image):
        try:
            if isinstance(image, Image.Image):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image
            results = self.model.predict(cv_image)
            if results and results[0].masks:
                mask = results[0].masks.data[0]
                mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                mask = np.squeeze(mask).astype(np.float32) * 255
                mask = mask.astype(np.uint8)
                h, w = cv_image.shape[:2]
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                segmented = cv2.bitwise_and(cv_image, cv_image, mask=mask)
                segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
                return Image.fromarray(segmented_rgb)
            else:
                logging.warning("No objects detected for segmentation.")
                return None
        except Exception as e:
            logging.error(f"Error segmenting object: {e}", exc_info=True)
            return None

    def segment_object_at_point(self, image, point):
        try:
            cv_image = np.array(image)
            results = self.model.predict(cv_image)
            for result in results:
                for bbox, mask in zip(result.boxes, result.masks.xy):
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                        mask_img = np.zeros(image.shape[:2], dtype=np.uint8)
                        polygon = np.array(mask, dtype=np.int32)
                        cv2.fillPoly(mask_img, [polygon], 255)
                        segmented = cv2.bitwise_and(image, image, mask=mask_img)
                        return segmented
            logging.warning("No object found at the given point.")
            return None
        except Exception as e:
            logging.error(f"Error segmenting object at point: {e}", exc_info=True)
            return None

    def apply_brush(self, image, points, color, size):
        draw = ImageDraw.Draw(image)
        for i in range(1, len(points)):
            draw.line([points[i-1], points[i]], fill=color, width=size)
        return image

    def apply_eraser(self, image, points, size):
        draw = ImageDraw.Draw(image)
        for i in range(1, len(points)):
            draw.line([points[i-1], points[i]], fill=(255, 255, 255, 0), width=size)
        return image

    def apply_text(self, image, text, position, color=(0,0,0), font_size=20):
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        draw.text(position, text, fill=color, font=font)
        return image

    def apply_eyedropper(self, image, point):
        try:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            color = image.getpixel(point)
            return color if len(color) == 4 else color + (255,)
        except Exception as e:
            logging.error(f"Error using eyedropper: {e}", exc_info=True)
            return (0, 0, 0, 255)

    def apply_fill(self, image, point, color, tolerance=32):
        try:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            result = image.copy()
            target_color = image.getpixel(point)
            img_array = np.array(image)
            color_diff = np.abs(img_array - np.array(target_color))
            color_distance = np.sqrt(np.sum(color_diff ** 2, axis=-1))
            mask = color_distance <= tolerance
            if img_array.shape[2] == 4 and len(color) == 3:
                img_array[mask, :3] = color
                img_array[mask, 3] = 255
            elif img_array.shape[2] == 3 and len(color) == 4:
                img_array[mask] = color[:3]
            else:
                img_array[mask] = color
            return Image.fromarray(img_array)
        except Exception as e:
            logging.error(f"Error applying fill: {e}", exc_info=True)
            return image

    def apply_crop(self, image, rect):
        try:
            cropped_image = image.crop(rect)
            return cropped_image
        except Exception as e:
            logging.error(f"Error applying crop: {e}", exc_info=True)
            return image

    def move_image_by(self, image, dx, dy, sensitivity=0.25):
        try:
            new_image = Image.new("RGBA", image.size, (255, 255, 255, 0))
            new_image.paste(image, (int(dx * sensitivity), int(dy * sensitivity)))
            return new_image
        except Exception as e:
            logging.error(f"Error moving image: {e}", exc_info=True)
            return image