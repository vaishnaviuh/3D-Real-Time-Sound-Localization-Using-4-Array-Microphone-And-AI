"""
KML parser for extracting sensor pole locations.
Parses the Sensor-Locations-BOP-Dharma.kml file to get precise coordinates.
"""
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from src.utils import get_logger

logger = get_logger("sound_localization.kml_parser")

@dataclass
class SensorPole:
    """Represents a sensor pole with its location and metadata."""
    name: str
    longitude: float
    latitude: float
    altitude: float
    timestamp: str
    accuracy: float
    
    def to_xyz(self, reference_pole: 'SensorPole' = None) -> Tuple[float, float, float]:
        """
        Convert lat/lon/alt to local XYZ coordinates in meters.
        If reference_pole is provided, coordinates are relative to it.
        Otherwise, uses the first pole as reference (0,0,0).
        """
        if reference_pole is None:
            return (0.0, 0.0, self.altitude)
        
        # Convert lat/lon differences to meters using approximate conversion
        # At latitude ~31.35°N (from KML data):
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(31.35°) ≈ 95,000 meters
        
        lat_diff = self.latitude - reference_pole.latitude
        lon_diff = self.longitude - reference_pole.longitude
        alt_diff = self.altitude - reference_pole.altitude
        
        # Convert to meters
        x = lon_diff * 95000.0  # East-West (longitude)
        y = lat_diff * 111320.0  # North-South (latitude)
        z = alt_diff  # Altitude difference
        
        return (x, y, z)


def parse_kml_file(kml_path: str) -> List[SensorPole]:
    """
    Parse KML file and extract sensor pole locations.
    
    Args:
        kml_path: Path to the KML file
        
    Returns:
        List of SensorPole objects sorted by name
    """
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        
        # Define KML namespace
        ns = {'kml': 'http://www.opengis.net/kml/2.2',
              'gpstestkml': 'http://www.chartcross.co.uk/XML/GPXDATA/1/0'}
        
        poles = []
        
        # Find all Placemark elements
        for placemark in root.findall('.//kml:Placemark', ns):
            try:
                # Extract name
                name_elem = placemark.find('kml:name', ns)
                if name_elem is None:
                    continue
                name = name_elem.text.strip('[]')
                
                # Extract timestamp
                timestamp_elem = placemark.find('.//kml:when', ns)
                timestamp = timestamp_elem.text if timestamp_elem is not None else ""
                
                # Extract accuracy
                accuracy_elem = placemark.find('.//gpstestkml:accuracy', ns)
                accuracy = float(accuracy_elem.text) if accuracy_elem is not None else 0.0
                
                # Extract coordinates
                coord_elem = placemark.find('.//kml:coordinates', ns)
                if coord_elem is None:
                    continue
                
                coords = coord_elem.text.strip().split(',')
                if len(coords) < 3:
                    continue
                
                longitude = float(coords[0])
                latitude = float(coords[1])
                altitude = float(coords[2])
                
                pole = SensorPole(
                    name=name,
                    longitude=longitude,
                    latitude=latitude,
                    altitude=altitude,
                    timestamp=timestamp,
                    accuracy=accuracy
                )
                poles.append(pole)
                
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse placemark: {e}")
                continue
        
        # Sort by name (extract number from name like "SSSD BOP DHARMA 001")
        def extract_number(name: str) -> int:
            try:
                # Extract the last number from the name
                parts = name.split()
                return int(parts[-1])
            except (ValueError, IndexError):
                return 999  # Put unparseable names at the end
        
        poles.sort(key=lambda p: extract_number(p.name))
        
        logger.info(f"Parsed {len(poles)} sensor poles from KML file")
        for pole in poles:
            logger.debug(f"  {pole.name}: ({pole.longitude:.6f}, {pole.latitude:.6f}, {pole.altitude:.1f}m)")
        
        return poles
        
    except Exception as e:
        logger.error(f"Failed to parse KML file {kml_path}: {e}")
        raise


def get_sensor_positions_xyz(kml_path: str, add_opposite_sensors: bool = True) -> Tuple[List[str], np.ndarray]:
    """
    Get sensor pole positions in local XYZ coordinates.
    
    Args:
        kml_path: Path to the KML file
        add_opposite_sensors: If True, add 5 sensors opposite to the cluster for channels 16-20
        
    Returns:
        Tuple of (pole_names, positions_array)
        positions_array shape: (N, 3) where N is number of poles
    """
    poles = parse_kml_file(kml_path)
    
    if not poles:
        raise ValueError("No sensor poles found in KML file")
    
    # Use first pole as reference (origin)
    reference_pole = poles[0]
    
    names = []
    positions = []
    
    for pole in poles:
        names.append(pole.name)
        xyz = pole.to_xyz(reference_pole)
        positions.append(xyz)
    
    # Add 1 single cluster pointer between sensors 5 and 7 (represents channels 16-20 combined)
    if add_opposite_sensors and len(poles) == 15:
        logger.info("Adding single cluster pointer between sensors 5 and 7 for channels 16-20")
        
        # Get positions of sensors 5 and 7 (indices 4 and 6)
        sensor_5_pos = np.array(positions[4])  # SSSD BOP DHARMA 005
        sensor_7_pos = np.array(positions[6])  # SSSD BOP DHARMA 007
        
        # Calculate cluster center between sensors 5 and 7
        cluster_center = (sensor_5_pos + sensor_7_pos) / 2.0
        cluster_center[2] += 2.0  # Slightly elevated
        
        logger.info(f"Sensor 5 position: ({sensor_5_pos[0]:.1f}, {sensor_5_pos[1]:.1f}, {sensor_5_pos[2]:.1f})")
        logger.info(f"Sensor 7 position: ({sensor_7_pos[0]:.1f}, {sensor_7_pos[1]:.1f}, {sensor_7_pos[2]:.1f})")
        logger.info(f"Single cluster pointer: ({cluster_center[0]:.1f}, {cluster_center[1]:.1f}, {cluster_center[2]:.1f})")
        
        # Add single cluster pointer
        cluster_name = "SSSD BOP DHARMA CLUSTER"
        names.append(cluster_name)
        positions.append(cluster_center.tolist())
        
        logger.info(f"Added single cluster pointer: {cluster_name} at ({cluster_center[0]:.1f}, {cluster_center[1]:.1f}, {cluster_center[2]:.1f})")
    
    positions_array = np.array(positions, dtype=np.float64)
    
    logger.info(f"Total sensors: {len(names)} (15 original + 1 cluster pointer)")
    logger.info(f"Reference pole: {reference_pole.name} at origin (0, 0, 0)")
    logger.info(f"Position range: X=[{positions_array[:, 0].min():.1f}, {positions_array[:, 0].max():.1f}]m, "
                f"Y=[{positions_array[:, 1].min():.1f}, {positions_array[:, 1].max():.1f}]m, "
                f"Z=[{positions_array[:, 2].min():.1f}, {positions_array[:, 2].max():.1f}]m")
    
    return names, positions_array


def calculate_distances_between_poles(positions: np.ndarray) -> np.ndarray:
    """
    Calculate distances between all pairs of sensor poles.
    
    Args:
        positions: Array of shape (N, 3) with pole positions
        
    Returns:
        Distance matrix of shape (N, N)
    """
    N = positions.shape[0]
    distances = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


if __name__ == "__main__":
    # Test the parser
    kml_path = "Sensor-Locations-BOP-Dharma.kml"
    try:
        names, positions = get_sensor_positions_xyz(kml_path)
        distances = calculate_distances_between_poles(positions)
        
        print(f"\nFound {len(names)} sensor poles:")
        for i, (name, pos) in enumerate(zip(names, positions)):
            print(f"  {i+1:2d}. {name}: ({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}) m")
        
        print(f"\nDistance statistics:")
        non_zero_distances = distances[distances > 0]
        print(f"  Min distance: {non_zero_distances.min():.1f} m")
        print(f"  Max distance: {non_zero_distances.max():.1f} m")
        print(f"  Mean distance: {non_zero_distances.mean():.1f} m")
        
        # Check if distances are approximately 100m as expected
        expected_distance = 100.0
        close_to_100m = np.abs(non_zero_distances - expected_distance) < 20.0
        print(f"  Distances close to 100m: {close_to_100m.sum()}/{len(non_zero_distances)} ({100*close_to_100m.mean():.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
