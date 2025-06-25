import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial import distance
from VideoSearch.models import Keyframe

max_palette_dist = np.sqrt(15 * 255**2)

class ColorFeatureExtractor:

    def __init__(self, hist_bins=(32, 8, 8), use_palette=True, use_colorfulness=True, command=None):
        self.command = command
        self.hist_bins = hist_bins
        self.use_palette = use_palette
        self.use_colorfulness = use_colorfulness

    def extract_hsv_histogram(self, image: Image.Image) -> np.ndarray:
        img = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, self.hist_bins, [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def extract_dominant_colors(self, image: Image.Image, k=5) -> np.ndarray:
        img = np.array(image.convert("RGB"))
        pixels = img.reshape(-1, 3)

        unique_pixels = np.unique(pixels, axis=0)

        if unique_pixels.shape[0] < k:
            mean_color = np.mean(unique_pixels, axis=0)
            return np.tile(mean_color, (k, 1)).flatten()

        kmeans = KMeans(n_clusters=k, n_init="auto").fit(pixels)
        centers = kmeans.cluster_centers_
        sorted_centers = centers[np.argsort(np.mean(centers, axis=1))]
        return sorted_centers.flatten()

    def calculate_colorfulness(self, image: Image.Image) -> float:
        img = np.array(image.convert("RGB")).astype("float")
        rg = np.abs(img[:, :, 0] - img[:, :, 1])
        yb = np.abs(0.5 * (img[:, :, 0] + img[:, :, 1]) - img[:, :, 2])
        return np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

    def extract_all(self, image: Image.Image) -> dict:
        result = {
            "histogram": self.extract_hsv_histogram(image),
        }
        if self.use_palette:
            result["palette"] = self.extract_dominant_colors(image)
        if self.use_colorfulness:
            result["colorfulness"] = self.calculate_colorfulness(image)
        return result

    def distance_to_existing_keyframes(self, clip, query_feat: dict, weights=None):
        keyframes = Keyframe.objects.filter(clip=clip)
        if not keyframes.exists():
            return 1.0, 1.0

        feature_list = []
        for kf in keyframes:
            features = {
                "histogram": kf.load_histogram_hsv(),
                "palette": kf.load_dominant_colors(),
                "colorfulness": kf.colorfulness
            }
            feature_list.append(features)

        return self.distance_to_feature_set(query_feat, feature_list, weights)
    
    def _log(self, msg, level="info"):
        if self.command:
            self.command.stdout.write(getattr(self.command, f"style_{level}")(msg))
        else:
            print(msg)

def distance_to_feature_set(self, query_feat: dict, feature_list: list[dict], weights=None):
    distances = [compute_distance(query_feat, f, weights) for f in feature_list]
    return min(distances), max(distances)

def compute_distance(a: dict, b: dict, weights=None) -> float:
    DEFAULT_WEIGHTS = {
        "histogram": 1.0,
        "palette": 0.5,
        "colorfulness": 0.2
    }
    COLORFULNESS_MAX = 100.0

    weights = weights or DEFAULT_WEIGHTS
    dist = 0.0
    norm = 0.0

    if "histogram" in a and "histogram" in b:
        w = weights.get("histogram", 1.0)
        dist += w * cv2.compareHist(a["histogram"].astype("float32"),
                                    b["histogram"].astype("float32"),
                                    cv2.HISTCMP_BHATTACHARYYA)
        norm += w

    if "palette" in a and "palette" in b:
        if a["palette"] is not None and b["palette"] is not None:
            w = weights.get("palette", 0.5)

            vec_a = np.ravel(a["palette"])
            vec_b = np.ravel(b["palette"])

            euclid = distance.euclidean(vec_a, vec_b)
            normalized = euclid / max_palette_dist

            dist += w * normalized
            norm += w

    if "colorfulness" in a and "colorfulness" in b:
        w = weights.get("colorfulness", 0.2)
        diff = min(COLORFULNESS_MAX, abs(a["colorfulness"] - b["colorfulness"])) / COLORFULNESS_MAX
        dist += w * diff
        norm += w

    return dist / norm if norm > 0 else 1.0