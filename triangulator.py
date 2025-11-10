import numpy as np
from scipy.optimize import least_squares
from config import MIC_POSITIONS, SOUND_SPEED as DEFAULT_SOUND_SPEED


class TriangulationLocalizer:
    """Simple 3D position estimation from angles and TDOA"""
    
    def __init__(self, mic_positions=MIC_POSITIONS, sound_speed: float = DEFAULT_SOUND_SPEED):
        self.mic_positions = mic_positions
        self.num_mics = len(mic_positions)
        self.sound_speed = float(sound_speed)
        # Calculate baseline distance between microphones
        pair_dists = []
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                pair_dists.append(np.linalg.norm(self.mic_positions[j] - self.mic_positions[i]))
        self.baseline = float(np.median(pair_dists)) if pair_dists else 0.05
    
    def triangulate_3d(self, tdoa_matrix):
        """Triangulate 3D position from TDOA matrix using least squares optimization"""
        # Check if all TDOA are near zero (source directly above)
        max_tdoa_abs = np.max(np.abs(tdoa_matrix))
        if max_tdoa_abs < 5e-5:  # Less than 50 microseconds
            # Source is directly above - return None to use special case
            return None
        
        # Collect all TDOA measurements
        tdoa_pairs = []
        mic_pairs = []
        
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                tdoa = tdoa_matrix[i, j]
                # Use a lower threshold to include more measurements
                if abs(tdoa) > 1e-8:  # Include very small but non-zero TDOA
                    tdoa_pairs.append(tdoa)
                    mic_pairs.append((i, j))
        
        if len(tdoa_pairs) < 3:  # Need at least 3 pairs for 3D
            return None
        
        # Estimate initial direction from TDOA
        direction_estimate = np.zeros(3)
        for (i, j), tdoa in zip(mic_pairs, tdoa_pairs):
            mic_vec = self.mic_positions[j] - self.mic_positions[i]
            mic_dist = np.linalg.norm(mic_vec)
            if mic_dist > 1e-6:
                cos_theta = np.clip(tdoa * self.sound_speed / mic_dist, -1, 1)
                direction_estimate += cos_theta * (mic_vec / mic_dist)
        
        direction_estimate = direction_estimate / (len(mic_pairs) + 1e-10)
        dir_mag = np.linalg.norm(direction_estimate)
        if dir_mag > 1e-6:
            direction_estimate = direction_estimate / dir_mag
        else:
            direction_estimate = np.array([0.0, 0.0, 1.0])  # Default: straight up
        
        # Estimate distance from max TDOA
        max_tdoa = max([abs(t) for t in tdoa_pairs]) if tdoa_pairs else 0
        if max_tdoa > 1e-6:
            estimated_distance = min(max(0.1, self.baseline / (max_tdoa * self.sound_speed + 1e-6)), 3.0)
        else:
            estimated_distance = 0.5  # Default 50cm
        
        # Initial position: use estimated distance in estimated direction
        initial_pos = np.mean(self.mic_positions, axis=0) + estimated_distance * direction_estimate
        # Ensure z is positive and reasonable (5cm to 2m)
        initial_pos[2] = max(0.05, min(2.0, initial_pos[2]))
        
        def residual_function(pos):
            """Residual function for least squares - use all pairs"""
            residuals = []
            for (i, j), tdoa in zip(mic_pairs, tdoa_pairs):
                dist_i = np.linalg.norm(pos - self.mic_positions[i])
                dist_j = np.linalg.norm(pos - self.mic_positions[j])
                expected_tdoa = (dist_j - dist_i) / self.sound_speed
                residuals.append(expected_tdoa - tdoa)
            return np.array(residuals)
        
        # Try multiple initial guesses with different distances and z-heights
        best_result = None
        best_cost = float('inf')
        
        # Test different distances (prioritize closer ones first)
        test_distances = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
        # Test a few key z-heights: low, medium, high
        test_z_heights = [0.1, 0.3, 0.5, 0.75, 1.0]
        
        # First try with estimated distance and a few z-heights
        for z_height in test_z_heights[:3]:  # Try top 3 z-heights first
            try:
                test_init = np.mean(self.mic_positions, axis=0) + estimated_distance * direction_estimate
                test_init[2] = z_height
                
                bounds = ([-3, -3, 0.05], [3, 3, 2.0])
                result = least_squares(residual_function, test_init, method='trf', 
                                     bounds=bounds, max_nfev=200, ftol=1e-5, verbose=0)
                
                if result.success and result.cost < best_cost:
                    min_dist = min([np.linalg.norm(result.x - mic) for mic in self.mic_positions])
                    max_dist = max([np.linalg.norm(result.x - mic) for mic in self.mic_positions])
                    if 0.05 < min_dist < 3.0 and max_dist < 3.0:
                        best_result = result
                        best_cost = result.cost
            except Exception:
                continue
        
        # If that didn't work well, try more combinations
        if best_result is None or best_cost > 0.5:
            for init_dist in test_distances[:4]:  # Try first 4 distances
                for z_height in test_z_heights:
                    try:
                        test_init = np.mean(self.mic_positions, axis=0) + init_dist * direction_estimate
                        test_init[2] = z_height
                        
                        bounds = ([-3, -3, 0.05], [3, 3, 2.0])
                        result = least_squares(residual_function, test_init, method='trf', 
                                             bounds=bounds, max_nfev=200, ftol=1e-5, verbose=0)
                        
                        if result.success and result.cost < best_cost:
                            min_dist = min([np.linalg.norm(result.x - mic) for mic in self.mic_positions])
                            max_dist = max([np.linalg.norm(result.x - mic) for mic in self.mic_positions])
                            if 0.05 < min_dist < 3.0 and max_dist < 3.0:
                                best_result = result
                                best_cost = result.cost
                    except Exception:
                        continue
        
        # More lenient cost threshold - accept if residual is reasonable
        if best_result is not None and best_cost < 1.0:  # More lenient threshold
            return best_result.x
        
        return None
    
    def estimate_from_angles(self, azimuth_deg, elevation_deg, distance=0.15):
        """Convert angles and distance to 3D position - SIMPLE METHOD"""
        azimuth = np.radians(azimuth_deg)
        elevation = np.radians(elevation_deg)
        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)
        # Ensure z is above array (allow up to 2m height)
        z = max(0.05, min(2.0, z))
        return np.array([x, y, z])
    
    def estimate_distance_from_tdoa(self, tdoa_matrix):
        """Estimate distance from maximum TDOA value"""
        max_tdoa = np.max(np.abs(tdoa_matrix))
        if max_tdoa > 1e-6:
            # Simple distance estimate: closer source = larger TDOA
            # For near-field: distance ≈ baseline / (max_tdoa * sound_speed)
            distance = self.baseline / (max_tdoa * self.sound_speed + 1e-6)
            # Allow distances from 10cm to 3m (to support sources above 50cm)
            return min(max(0.1, distance), 3.0)
        return 0.15  # Default 15cm


