import cv2
import numpy as np

from typing import Tuple
from data_models import CameraModel, PitchModel

np.set_printoptions(suppress=True)


class HomographyEstimator:
    def __init__(self, camera_model: CameraModel, pitch_model: PitchModel):
        self.camera_model = camera_model
        self.pitch_model = pitch_model
        self.pixel_coords, self.world_coords = self._extract_matching_coordinates()
        assert len(self.pixel_coords) is not 0 and len(self.world_coords) is not 0, "No matching coordinates found."

    def _extract_matching_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts and matches coordinates from camera and pitch models.
        Only coordinates that are present in both models are kept.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of numpy arrays containing
            the matched pixel coordinates and world coordinates.
        """
        matched_pixel_coords = []
        matched_world_coords = []

        for coord_key in self.camera_model.pitch_pixel_coords:
            if coord_key in self.pitch_model.pitch_dimensions and self.pitch_model.pitch_dimensions[coord_key] and len(self.camera_model.pitch_pixel_coords[coord_key]) != 0:
                matched_pixel_coords.append(self.camera_model.pitch_pixel_coords[coord_key])
                matched_world_coords.append(self.pitch_model.pitch_dimensions[coord_key])

        pixel_array = np.array(matched_pixel_coords).reshape(-1, 2)
        world_array = np.array(matched_world_coords).reshape(-1, 2)

        assert len(pixel_array) == len(world_array), "Pixel and world coordinate arrays must be of equal length."
        assert pixel_array.shape[1] == 2, "Pixel coordinate array must have shape (n, 2)."

        return pixel_array, world_array

    def estimate_homography(self) -> np.ndarray:
        """
        Estimates the homography matrix between the camera and pitch models.

        Returns:
            np.ndarray: Homography matrix. Shape (3, 3).
        """
        return cv2.findHomography(self.pixel_coords, self.world_coords)[0]


def main():
    camera_model = CameraModel("Jetson1", match_date="01-01-2023")
    pitch_model = PitchModel(sport="soccer")
    homography_estimator = HomographyEstimator(camera_model, pitch_model)

    homography_matrix = homography_estimator.estimate_homography()
    print(homography_matrix)


if __name__ == '__main__':
    main()
