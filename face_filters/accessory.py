import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


class FaceAccessory:
    
    def __init__(self, image_path, left_idx, right_idx, width_scale=1.5,
                 vertical_offset_ratio=0.0, horizontal_offset_ratio=0.0):
        self.image = self._load_and_clean(image_path)
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.width_scale = width_scale
        self.vertical_offset_ratio = vertical_offset_ratio
        self.horizontal_offset_ratio = horizontal_offset_ratio

    @staticmethod
    def _load_and_clean(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        b, g, r, a = cv2.split(img)
        mask = a < 10
        b[mask] = 0
        g[mask] = 0
        r[mask] = 0
        return cv2.merge([b, g, r, a])

    @staticmethod
    def _rotate(image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(rot[0, 0])
        sin = np.abs(rot[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rot[0, 2] += (new_w / 2) - center[0]
        rot[1, 2] += (new_h / 2) - center[1]

        return cv2.warpAffine(
            image, rot, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    def apply(self, frame, landmarks, frame_w, frame_h):
        lm = landmarks
        p1 = np.array([lm[self.left_idx].x * frame_w, lm[self.left_idx].y * frame_h])
        p2 = np.array([lm[self.right_idx].x * frame_w, lm[self.right_idx].y * frame_h])
        dist = np.linalg.norm(p2 - p1)

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        angle = np.degrees(np.arctan2(dy, dx))

        target_w = int(dist * self.width_scale)
        target_h = int(target_w / self.image.shape[1] * self.image.shape[0])
        if target_w <= 0 or target_h <= 0:
            return frame

        resized = cv2.resize(self.image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rotated = self._rotate(resized, -angle)
        rh, rw = rotated.shape[:2]

        center = (p1 + p2) / 2
        x = int(center[0] - rw / 2 + target_w * self.horizontal_offset_ratio)
        y = int(center[1] - rh / 2 + target_h * self.vertical_offset_ratio)

        return self._overlay(frame, rotated, x, y)

    @staticmethod
    def _overlay(background, overlay_img, x, y):
        h, w = overlay_img.shape[:2]
        bg_h, bg_w = background.shape[:2]

        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, bg_w), min(y + h, bg_h)
        if x1 >= x2 or y1 >= y2:
            return background

        cropped = overlay_img[y1 - y: y2 - y, x1 - x: x2 - x]
        alpha = cropped[:, :, 3] / 255.0
        alpha = alpha[..., np.newaxis]

        roi = background[y1:y2, x1:x2]
        blended = (alpha * cropped[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
        background[y1:y2, x1:x2] = blended
        return background
