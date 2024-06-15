import json
import os

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, NamedTuple


class CameraModel:
    def __init__(self, camera_name: str, match_date: str = "01-01-2023", sport: str = "afl", camera_coords_json_path: str = "../data/homography_data/afl_camera_coordinates.json"):

        if sport == "afl":
            assert "marvel" in camera_name.lower(), "Camera name must start with 'marvel' for AFL."

        self.name: str = camera_name.lower()
        self.id: int = int(camera_name[-1])
        self.date = match_date
        self.pitch_pixel_coords_path = camera_coords_json_path
        assert os.path.exists(self.pitch_pixel_coords_path), f"File not found: {self.pitch_pixel_coords_path}"

        self.pitch_pixel_coords, self.real_world_camera_coords, self.image_field_coordinates = self.load_camera_data()

    def load_camera_data(self) -> Tuple[Dict[str, List[Optional[int]]], Dict[str, float], Dict[str, List[int]]]:
        """
    Loads camera data from a JSON file based on specific camera name and date. Returns a tuple with three dictionaries:
    pitch pixel coordinates (with optional values), real-world camera coordinates, and image field coordinates.
        """
        with open(self.pitch_pixel_coords_path, 'r') as f:
            all_camera_data = json.load(f)

        camera_data = all_camera_data.get("cameras", {}).get(self.name, {})
        date_data = camera_data.get("dates", {}).get(self.date, {})

        return date_data.get("pitch_pixel_coords", {}), \
            date_data.get("camera_coords", {}), \
            date_data.get("image_field_coordinates", {})


class PitchModel:
    def __init__(self, sport: str):
        if sport.lower() == "afl":
            print("Using AFL Pitch model...")
            self.pitch_dimensions_path = "../data/homography_data/afl_real_world_pitch_coordinates_normalized.json"
        else:  # Football (soccer)
            print("Using Football (soccer) Pitch model...")
            self.pitch_dimensions_path = "../data/homography_data/real_world_pitch_dimensions.json"

        self.pitch_dimensions = self.load_pitch_dimensions()

    def load_pitch_dimensions(self):
        with open(self.pitch_dimensions_path, 'r') as f:
            pitch_dimensions = json.load(f)
        pitch_dimensions = pitch_dimensions.get("real_world_pitch_coords", {})
        return pitch_dimensions


@dataclass
class Detections:
    camera_id: int
    probability: float = 0.9
    timestamp: float = 0
    x: float = 0
    y: float = 0
    z: float = 1  # Set to 1 for homography normalisation for now


@dataclass
class TriangulationResult:
    x: float
    y: float
    z: float
    shortest_perpendicular_distance: float  # Shortest distance between the two skew lines from the triangulation algo


class Camera(NamedTuple):
    id: int
    homography: list
    real_world_camera_coords: Tuple
    image_field_coordinates: List[Tuple[int, int]]  # Coordinates are in the format (x, y), and start from the top left corner and go clockwise.



################ Unused so far ################


@dataclass
class ThreeDPoints:
    # Object for storing the resulting 3D points in the MultiCameraTracker object for error handling
    x: float
    y: float
    z: float
    timestamp: float

    @classmethod
    def three_d_point_from_det(cls, det: Detections):
        """
        Creates a ThreeDPoints instance from a Detections object, preserving coordinates and timestamp.

        Args:
            det (Detections): The Detections object to convert.

        Returns:
            ThreeDPoints: A ThreeDPoints instance with the same attributes as the input Detections object.
        """
        # TODO: I sometimes set z to 0 (not really sure why tbh), so will need to account for that
        return cls(det.x, det.y, det.z, det.timestamp)


@dataclass
class DetectionError:
    """
    Base dataclass representing a detection error, containing the 3D coordinates
    of the erroneous detection (x, y, z) and the time of the detection (timestamp).
    """

    x: float
    y: float
    z: float
    timestamp: float

    @classmethod
    def from_three_d_points(cls, threedpoints: ThreeDPoints):
        """
        Creates a DetectionError instance from a ThreeDPoints object, preserving coordinates and timestamp.

        Args:
            threedpoints (ThreeDPoints): The ThreeDPoints object to convert.

        Returns:
            DetectionError: A DetectionError instance with the same attributes as the input ThreeDPoints object.
        """
        return cls(threedpoints.x, threedpoints.y, threedpoints.z, threedpoints.timestamp)


@dataclass
class OutOfBounds(DetectionError):
    """
    Dataclass representing an out-of-bounds detection.
    Inherits from DetectionError and retains the same structure and methods.
    """
    pass


@dataclass
class FailedCommonSense(DetectionError):
    """
    Dataclass representing a detection that failed the 'common_sense' test.
    Inherits from DetectionError and retains the same structure and methods.
    """
    pass

##############################################


def main():
    # camera_model1 = CameraModel("Jetson1", match_date="01-01-2023")
    # print(camera_model1.name)
    # print(camera_model1.pitch_pixel_coords)
    # print(camera_model1.real_world_camera_coords)
    #
    camera_model2 = CameraModel("Jetson2", match_date="01-01-2023")
    print(camera_model2.name)
    print(camera_model2.pitch_pixel_coords)
    print(camera_model2.real_world_camera_coords)
    print(camera_model2.image_field_coordinates)

    # pitch_model = PitchModel()
    # print(pitch_model.pitch_dimensions)


if __name__ == '__main__':
    main()
