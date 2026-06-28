"""
Visualiza cómo se divide la imagen estéreo ancha en dos mitades (izquierda/derecha).
Uso: python split_preview.py [ruta_imagen]
"""
import cv2
import numpy as np
import sys
import os

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(__file__), "../data/new_new_cam_calibration_imgs/photo_1.jpeg")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"No se pudo cargar: {IMAGE_PATH}")

h, w = img.shape[:2]
mid = w // 2

left  = img[:, :mid]
right = img[:, mid:]

print(f"Imagen original : {w}x{h} px")
print(f"Mitad izquierda : {mid}x{h} px  (cámara izquierda)")
print(f"Mitad derecha   : {w - mid}x{h} px  (cámara derecha)")

# Dibuja línea divisoria en la imagen original
preview = img.copy()
cv2.line(preview, (mid, 0), (mid, h), (0, 0, 255), 3)
cv2.putText(preview, "IZQUIERDA", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
cv2.putText(preview, "DERECHA", (mid + 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

cv2.imshow("Division estéreo (línea roja = corte)", preview)
cv2.imshow("Cámara IZQUIERDA", left)
cv2.imshow("Cámara DERECHA",   right)

print("\nPulsa cualquier tecla para cerrar...")
cv2.waitKey(0)
cv2.destroyAllWindows()
