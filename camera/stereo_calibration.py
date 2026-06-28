"""
Calibración estéreo de cámara dual con OpenCV.
Las imágenes son capturas anchas (2560x800) con ambas cámaras juntas.
Se divide cada imagen por la mitad para obtener izquierda y derecha.

Uso: python stereo_calibration.py
Salida: stereo_calibration_result.npz  (matrices de calibración para calcular profundidad)
"""
import cv2
import numpy as np
import glob
import os

# ──────────────────────────────────────────────
# PARÁMETROS — ajustar si el tablero es distinto
# ──────────────────────────────────────────────
IMGS_DIR       = os.path.join(os.path.dirname(__file__), "../data/new_new_cam_calibration_imgs")
OUTPUT_FILE    = os.path.join(os.path.dirname(__file__), "stereo_calibration_result.npz")

# Número de ESQUINAS INTERIORES del tablero (columnas, filas)
# Para un tablero 8x8 de cuadros → 7x7 esquinas interiores
CHESSBOARD_COLS = 9      # esquinas interiores = columnas_cuadros - 1
CHESSBOARD_ROWS = 6      # esquinas interiores = filas_cuadros   - 1
SQUARE_SIZE_MM  = 40.5   # tamaño real de cada cuadro en milímetros

# ──────────────────────────────────────────────

def split_stereo(img):
    """Divide imagen ancha en mitad izquierda y derecha."""
    mid = img.shape[1] // 2
    return img[:, :mid], img[:, mid:]


def find_corners(img_gray, board_size, criteria):
    """Detecta esquinas del tablero en escala de grises."""
    found, corners = cv2.findChessboardCorners(img_gray, board_size, None)
    if found:
        corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def main():
    board_size = (CHESSBOARD_COLS, CHESSBOARD_ROWS)

    # Puntos 3D del tablero en el mundo real (z=0, plano)
    objp = np.zeros((CHESSBOARD_COLS * CHESSBOARD_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    criteria_refine = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    image_paths = sorted(glob.glob(os.path.join(IMGS_DIR, "*.jpeg")) +
                         glob.glob(os.path.join(IMGS_DIR, "*.jpg"))  +
                         glob.glob(os.path.join(IMGS_DIR, "*.png")))

    print(f"{'='*60}")
    print(f"  CALIBRACIÓN ESTÉREO")
    print(f"{'='*60}")
    print(f"  Directorio  : {IMGS_DIR}")
    print(f"  Imágenes    : {len(image_paths)} encontradas")
    print(f"  Tablero     : {CHESSBOARD_COLS}x{CHESSBOARD_ROWS} esquinas interiores")
    print(f"  Cuadro real : {SQUARE_SIZE_MM} mm")
    print(f"{'='*60}\n")

    obj_points  = []   # puntos 3D
    pts_left    = []   # esquinas cámara izquierda
    pts_right   = []   # esquinas cámara derecha
    img_size    = None
    ok_count    = 0
    fail_imgs   = []

    for i, path in enumerate(image_paths):
        fname = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"  [{i+1:>3}] {fname}  ← ERROR al cargar")
            continue

        left_c, right_c = split_stereo(img)
        gray_l = cv2.cvtColor(left_c,  cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_c, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray_l.shape[::-1]  # (width, height)

        found_l, corners_l = find_corners(gray_l, board_size, criteria_refine)
        found_r, corners_r = find_corners(gray_r, board_size, criteria_refine)

        status = "OK" if (found_l and found_r) else \
                 ("FALLO_IZQ" if not found_l else "FALLO_DER")
        marker = "✓" if status == "OK" else "✗"
        print(f"  [{i+1:>3}] {fname:<30}  izq={str(found_l):<5}  der={str(found_r):<5}  {marker}")

        if found_l and found_r:
            obj_points.append(objp)
            pts_left.append(corners_l)
            pts_right.append(corners_r)
            ok_count += 1
        else:
            fail_imgs.append(fname)

    print(f"\n  Imágenes válidas: {ok_count} / {len(image_paths)}")
    if fail_imgs:
        print(f"  Fallidas        : {', '.join(fail_imgs)}")

    if ok_count < 6:
        print("\n  ERROR: Se necesitan al menos 6 imágenes válidas para calibrar.")
        return

    print(f"\n{'='*60}")
    print("  CALIBRACIÓN INDIVIDUAL — CÁMARA IZQUIERDA")
    print(f"{'='*60}")
    rms_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        obj_points, pts_left, img_size, None, None)
    print(f"  RMS reprojection error : {rms_l:.6f} px")
    print(f"  Matriz intrínseca K_L  :\n{K_l}")
    print(f"  Distorsión D_L         : {D_l.ravel()}")

    print(f"\n{'='*60}")
    print("  CALIBRACIÓN INDIVIDUAL — CÁMARA DERECHA")
    print(f"{'='*60}")
    rms_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        obj_points, pts_right, img_size, None, None)
    print(f"  RMS reprojection error : {rms_r:.6f} px")
    print(f"  Matriz intrínseca K_R  :\n{K_r}")
    print(f"  Distorsión D_R         : {D_r.ravel()}")

    print(f"\n{'='*60}")
    print("  CALIBRACIÓN ESTÉREO")
    print(f"{'='*60}")
    stereo_flags = (cv2.CALIB_FIX_INTRINSIC)   # fija las K y D ya calculadas
    rms_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, pts_left, pts_right,
        K_l, D_l, K_r, D_r,
        img_size,
        criteria=criteria_stereo,
        flags=stereo_flags
    )
    print(f"  RMS estéreo            : {rms_stereo:.6f} px")
    print(f"  Rotación R (cám der respecto izq):\n{R}")
    print(f"  Traslación T (mm)      : {T.ravel()}")
    baseline_mm = np.linalg.norm(T)
    print(f"  Baseline (distancia entre cámaras): {baseline_mm:.2f} mm  ({baseline_mm/10:.2f} cm)")
    print(f"  Matriz Esencial E      :\n{E}")
    print(f"  Matriz Fundamental F   :\n{F}")

    print(f"\n{'='*60}")
    print("  RECTIFICACIÓN ESTÉREO")
    print(f"{'='*60}")
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, D_l, K_r, D_r,
        img_size, R, T,
        alpha=0   # recorta zonas negras; usar alpha=1 para imagen completa
    )
    print(f"  R1 (rectificación izq) :\n{R1}")
    print(f"  R2 (rectificación der) :\n{R2}")
    print(f"  P1 (proyección izq)    :\n{P1}")
    print(f"  P2 (proyección der)    :\n{P2}")
    print(f"  Q  (matriz disparidad→3D):\n{Q}")
    print(f"  ROI izquierda válida   : {roi_l}")
    print(f"  ROI derecha válida     : {roi_r}")

    # Focal length y campo de visión aproximado
    fx_l = K_l[0, 0]
    fy_l = K_l[1, 1]
    cx_l = K_l[0, 2]
    cy_l = K_l[1, 2]
    fov_h = 2 * np.degrees(np.arctan(img_size[0] / (2 * fx_l)))
    fov_v = 2 * np.degrees(np.arctan(img_size[1] / (2 * fy_l)))
    print(f"\n  --- Parámetros derivados ---")
    print(f"  Focal length izq  : fx={fx_l:.2f}  fy={fy_l:.2f}  px")
    print(f"  Centro óptico izq : cx={cx_l:.2f}  cy={cy_l:.2f}  px")
    print(f"  FOV horizontal    : {fov_h:.1f}°")
    print(f"  FOV vertical      : {fov_v:.1f}°")
    print(f"  Resolución imagen : {img_size[0]}x{img_size[1]} px")

    # ── Guardar resultados ──────────────────────────────────
    np.savez(OUTPUT_FILE,
             K_l=K_l, D_l=D_l,
             K_r=K_r, D_r=D_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2,
             P1=P1, P2=P2,
             Q=Q,
             roi_l=roi_l, roi_r=roi_r,
             img_size=img_size,
             rms_stereo=rms_stereo)

    print(f"\n{'='*60}")
    print(f"  Resultados guardados en: {OUTPUT_FILE}")
    print(f"{'='*60}")
    print("\n  Para calcular profundidad usa la matriz Q con cv2.reprojectImageTo3D()")
    print("  Distancia estimada = (fx * baseline) / disparidad")


if __name__ == "__main__":
    main()
