import numpy as np
import cv2
from scipy import fftpack
from scipy.stats import skew, kurtosis
from skimage import feature
import warnings
import os

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Precision loss occurred in moment calculation')


class PureStyleExtractor:
    """
    Extracts 25 technical features
    """
    
    def __init__(self, device="cpu"):
        self.device = device
    
    def extract_frequency_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        fft = fftpack.fft2(gray)
        fft_shift = fftpack.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        power_spectrum = magnitude_spectrum ** 2
        
        h, w = power_spectrum.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        radial_profile = np.bincount(r.ravel(), power_spectrum.ravel()) / np.bincount(r.ravel())
        
        features = {
            'high_freq_energy': np.mean(radial_profile[-len(radial_profile)//4:]) if len(radial_profile) > 4 else 0,
            'mid_freq_energy': np.mean(radial_profile[len(radial_profile)//4:len(radial_profile)//2]) if len(radial_profile) > 4 else 0,
            'freq_falloff': np.polyfit(np.log(np.arange(1, min(50, len(radial_profile)))), 
                                       np.log(radial_profile[1:min(50, len(radial_profile))] + 1e-10), 
                                       1)[0] if len(radial_profile) > 2 else 0,
            'spectral_entropy': -np.sum((radial_profile / (radial_profile.sum() + 1e-10)) * 
                                       np.log(radial_profile / (radial_profile.sum() + 1e-10) + 1e-10)),
        }
        return features
    
    def extract_noise_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        else:
            gray = image.astype(float)
        
        noise = cv2.Laplacian(gray, cv2.CV_64F)
        
        features = {
            'noise_variance': np.var(noise),
            'noise_skewness': float(skew(noise.ravel())),
            'noise_kurtosis': float(kurtosis(noise.ravel())),
            'noise_local_var': np.std([np.var(noise[i:i+32, j:j+32]) 
                                       for i in range(0, gray.shape[0]-32, 32) 
                                       for j in range(0, gray.shape[1]-32, 32)]) if gray.shape[0] > 32 and gray.shape[1] > 32 else 0,
        }
        return features
    
    def extract_color_features(self, image):
        if len(image.shape) != 3:
            return {'color_saturation_var': 0, 'color_correlation_rg': 0, 
                   'color_correlation_rb': 0, 'color_correlation_gb': 0,
                   'lab_a_skewness': 0, 'lab_b_skewness': 0}
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        
        features = {
            'color_saturation_var': np.var(hsv[:,:,1]),
            'color_correlation_rg': float(np.corrcoef(r.ravel(), g.ravel())[0,1]),
            'color_correlation_rb': float(np.corrcoef(r.ravel(), b.ravel())[0,1]),
            'color_correlation_gb': float(np.corrcoef(g.ravel(), b.ravel())[0,1]),
            'lab_a_skewness': float(skew(lab[:,:,1].ravel())),
            'lab_b_skewness': float(skew(lab[:,:,2].ravel())),
        }
        return features
    
    def extract_texture_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if gray.shape[0] > 512:
            gray = cv2.resize(gray, (512, 512))
        
        gray_normalized = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-10) * 255).astype(np.uint8)
        
        glcm_features = []
        for distance in [1, 3, 5]:
            try:
                glcm = feature.graycomatrix(gray_normalized, [distance], [0], 256, symmetric=True, normed=True)
                contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
                energy = feature.graycoprops(glcm, 'energy')[0, 0]
                glcm_features.extend([contrast, homogeneity, energy])
            except:
                glcm_features.extend([0, 0, 0])
        
        try:
            lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-10)
        except:
            lbp_hist = np.zeros(10)
        
        features = {
            'glcm_contrast_1': glcm_features[0],
            'glcm_homogeneity_1': glcm_features[1],
            'glcm_energy_1': glcm_features[2],
            'glcm_contrast_3': glcm_features[3],
            'glcm_contrast_5': glcm_features[6],
            'lbp_entropy': -np.sum(lbp_hist * np.log(lbp_hist + 1e-10)),
        }
        return features
    
    def extract_edge_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features = {
            'edge_density': np.mean(edges > 0),
            'gradient_mean': np.mean(gradient_magnitude),
            'gradient_std': np.std(gradient_magnitude),
            'gradient_skewness': float(skew(gradient_magnitude.ravel())),
            'edge_coherence': np.mean(cv2.dilate(edges, np.ones((3,3))) == edges) if np.any(edges) else 0,
        }
        return features
        
    def __call__(self, image, normalize=True):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        freq_features = self.extract_frequency_features(image)
        noise_features = self.extract_noise_features(image)
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)
        edge_features = self.extract_edge_features(image)

        all_features = {**freq_features, **noise_features, **color_features,
                    **texture_features, **edge_features}

        feature_vector = np.array([all_features[k] for k in sorted(all_features.keys())])
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e10, neginf=-1e10)

        if normalize:
            feature_vector = self._normalize_features(feature_vector)

        return feature_vector.astype(np.float32)
    def _normalize_features(self, features):
        # Use robust hand-tuned constants that generalize across splits
        feature_means = np.array([
            0.9, 0.9, 0.9, 8000, 0.5, 0.2, 7e7, 100, 0.5, 400, 0.5, 0.5,
            80, 1.5, 50, 15e6, 0.0, 0.0, 1.5, 10e6, 8, 1500, 0.5, 2000, -5
        ])
        
        feature_stds = np.array([
            0.1, 0.1, 0.1, 3000, 0.2, 0.1, 5e7, 200, 300, 300, 0.2, 0.2,
            50, 1.0, 30, 12e6, 10, 10, 0.5, 8e6, 10, 1000, 2.0, 1500, 5
        ])
        
        normalized = (features - feature_means) / (feature_stds + 1e-8)
        normalized = np.clip(normalized, -10, 10)
        
        return normalized
        
    def get_feature_names(self):
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        freq = self.extract_frequency_features(dummy)
        noise = self.extract_noise_features(dummy)
        color = self.extract_color_features(dummy)
        texture = self.extract_texture_features(dummy)
        edge = self.extract_edge_features(dummy)
        
        all_features = {**freq, **noise, **color, **texture, **edge}
        return sorted(all_features.keys())

