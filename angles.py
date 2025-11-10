import numpy as np
from config import MIC_POSITIONS, SOUND_SPEED


class AngleEstimator:
    """Estimates azimuth and elevation angles from TDOA"""
    
    def __init__(self, mic_positions=MIC_POSITIONS):
        self.mic_positions = mic_positions
        self.num_mics = len(mic_positions)
    
    def estimate_azimuth_elevation(self, tdoa_matrix):
        """Estimate azimuth and elevation from TDOA matrix using TDOA-to-angle conversion"""
        x_components = []
        y_components = []
        z_components = []
        
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                tdoa = tdoa_matrix[i, j]
                if abs(tdoa) < 1e-7:
                    continue
                mic_vec = self.mic_positions[j] - self.mic_positions[i]
                mic_distance = np.linalg.norm(mic_vec)
                if mic_distance < 1e-6:
                    continue
                mic_vec_norm = mic_vec / mic_distance
                cos_theta = np.clip(tdoa * SOUND_SPEED / mic_distance, -1, 1)
                direction_component = cos_theta * mic_vec_norm
                x_components.append(direction_component[0])
                y_components.append(direction_component[1])
                z_components.append(direction_component[2])
        
        if x_components and y_components:
            avg_x = np.mean(x_components)
            avg_y = np.mean(y_components)
            avg_z = np.mean(z_components) if z_components else 0.0
            direction_mag = np.sqrt(avg_x**2 + avg_y**2 + avg_z**2)
            if direction_mag > 1e-6:
                direction_vec = np.array([avg_x, avg_y, avg_z]) / direction_mag
                azimuth_rad = np.arctan2(direction_vec[1], direction_vec[0])
                elevation_rad = np.arcsin(np.clip(direction_vec[2], -1, 1))
            else:
                azimuth_rad = 0.0
                elevation_rad = 0.0
        else:
            azimuth_rad = 0.0
            elevation_rad = 0.0
        
        azimuth_deg = np.degrees(azimuth_rad)
        elevation_deg = np.degrees(elevation_rad)
        return azimuth_deg, elevation_deg


