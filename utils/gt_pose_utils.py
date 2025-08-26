from pathlib import  Path
import cv2, yaml, numpy as np


def _load_dist_from_calib(calibration_path: str) -> np.ndarray:
    try:
        y = yaml.safe_load(open(calibration_path, "r"))
        dc = y['distortion_coefficients']
        return np.array([[dc.get("k1", 0), dc.get("k2", 0), dc.get("p1", 0),
                      dc.get("p2", 0), dc.get("k3", 0)]], dtype=float)
    except Exception:
        return np.zeros((1, 5), dtype=float)

def _load_K_from_calib(calib_path: str) -> np.ndarray:
    y = yaml.safe_load(open(calib_path, "r"))
    if "camera_matrix" in y and isinstance(y["camera_matrix"], dict):
        fx, fy = y["camera_matrix"]["fx"], y["camera_matrix"]["fy"]
        cx, cy = y["camera_matrix"]["cx"], y["camera_matrix"]["cy"]
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)
    elif "K" in y:
        return np.array(y["K"], float)
    raise SystemExit("calib yaml must contain camera_matrix{fx,fy,cx,cy} or K.")

def rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion.
    Args:
        R: A 3x3 rotation matrix.
    Returns:
        A numpy array representing the quaternion (qw, qx, qy, qz).
    """
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.0)*2
        w = 0.25*s
        x = (R[2,1]-R[1,2])/s
        y = (R[0,2]-R[2,0])/s
        z = (R[1,0]-R[0,1])/s
    else:
        i = np.argmax([R[0,0],R[1,1],R[2,2]])
        if i == 0:
            s = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
            w = (R[2,1]-R[1,2])/s
            x = 0.25*s
            y = (R[0,1]+R[1,0])/s
            z = (R[0,2]+R[2,0])/s
        elif i == 1:
            s = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
            w = (R[0,2]-R[2,0])/s
            x = (R[0,1]+R[1,0])/s
            y = 0.25*s
            z = (R[1,2]+R[2,1])/s
        else:
            s = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
            w = (R[1,0]-R[0,1])/s
            x = (R[0,2]+R[2,0])/s
            y = (R[1,2]+R[2,1])/s
            z = 0.25*s
    return (float(w), float(x), float(y), float(z))

def save_intrinsics(output_dir: Path, K: np.ndarray, dist: np.ndarray) -> None:
    (output_dir / "intrinsics.yaml").write_text(
        yaml.safe_dump({
            "K": K.tolist(),
            "distortion_coefficients": {
                "k1": float(dist[0, 0]), "k2": float(dist[0, 1]),
                "p1": float(dist[0, 2]), "p2": float(dist[0, 3]),
                "k3": float(dist[0, 4]),
            }
        }, sort_keys=False)
    )