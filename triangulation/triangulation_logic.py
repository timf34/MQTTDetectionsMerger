import os.path

import numpy as np
from numpy.linalg import lstsq

from copy import deepcopy
from matplotlib.path import Path
from statistics import mean
from typing import List, Dict, Tuple, Optional

from data_models import CameraModel, PitchModel, Detections, TriangulationResult
from triangulation.homography_estimator import HomographyEstimator

OPTICAL_FLOW_MULTIPLIER = 5


class KalmanFilter:
    def __init__(self, initial_state, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = np.eye(3)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def reset(self, initial_state) -> None:
        self.state = initial_state
        self.covariance = np.eye(3)

    def predict(self) -> None:
        # Assuming constant velocity model
        self.state = self.state
        self.covariance = self.covariance + self.process_noise

    def update(self, measurement) -> None:
        innovation = measurement - self.state
        innovation_covariance = self.covariance + self.measurement_noise
        kalman_gain = self.covariance @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        self.covariance = (np.eye(3) - kalman_gain) @ self.covariance


# TODO: note this is only a temporary function for now to make things simpler
def filter_most_confident_dets(_detections: List[Detections]) -> List[Detections]:
    """
    Temporary function. Filters each camera ID for the most confident detection.
    """
    filtered_dets = []
    for camera_id in set([_det.camera_id for _det in _detections]):
        camera_dets = [_det for _det in _detections if _det.camera_id == camera_id]
        camera_dets.sort(key=lambda x: x.probability, reverse=True)
        filtered_dets.append(camera_dets[0])

    return filtered_dets


class MultiCameraTracker:
    def __init__(self, sport: str, camera_coords_json_path: str = "../data/homography_data/afl_camera_coordinates.json"):
        print(f"Using {sport} MultiCameraTracker")

        self.cameras: Dict[str, CameraModel] = {}

        valid_sports = ["afl", "soccer"]
        if sport.lower() not in valid_sports:
            raise ValueError(f"Invalid sport: {sport}. Valid options are: {', '.join(valid_sports)}")
        self.pitch_model = PitchModel(sport=sport)
        self.pitch_boundary_path: Dict[str, Path] = {}
        self.homography_matrices: Dict[str, np.ndarray] = {}
        self.plane = None  # TODO: will fill this out later. Will need testing/ refactoring as I don't think the current implementation is correct

        self.kalman_filter = KalmanFilter(np.zeros(3), np.eye(3) * 0.01, np.eye(3) * 0.1)
        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        self.max_consecutive_misses = 5

        self.longest_perpendicular_distance_when_z_is_above_one: int = 8
        self.longest_perpendicular_distance_when_z_is_below_one: int = 6
        self.min_z_value_for_triangulation_result: int = -2
        self.max_z_value_for_triangulation_result: float = 9.5  # In case two close cameras on the same side have two detections that oppose each other (they'll have a very close perpendicular line, but high z)

        self.latest_camera_id: Optional[str] = None  # Last camera to detect a ball
        self.last_triangulated_position: Optional[Detections] = None
        self.optical_flow_used: bool = False  # Bool to help with resetting the Kalman filter

        self.camera_coords_json_path = camera_coords_json_path
        assert os.path.exists(self.camera_coords_json_path), f"File not found: {self.camera_coords_json_path}"

        # Two+ camera detections attributes
        self.ball_has_been_detected_by_two_or_more_cameras_within_time_frame: bool = False
        self.triangulated_position_where_ball_was_last_detected_by_two_or_more_cameras: Optional[Detections] = None
        self.buffer_radius: int = 40
        self.acceptable_time_frame_for_single_cam_det_after_two_or_more_cam_dets: int = 50  # accept single cam dets in the same area for 50 frames
        self.multi_camera_det_counter: int = 0  # Counter for how many frames since the ball was last detected by two or more cameras

        self._instantiate_cameras()

    def _instantiate_cameras(self) -> None:
        self.add_camera(camera_name="marvel1", match_date="20-08-2023")
        self.add_camera(camera_name="marvel2", match_date="20-08-2023")
        self.add_camera(camera_name="marvel3", match_date="20-08-2023")
        self.add_camera(camera_name="marvel5", match_date="27-08-2023")
        self.add_camera(camera_name="marvel6", match_date="19-08-2023")
        self.add_camera(camera_name="marvel7", match_date="27-08-2023")
        self.add_camera(camera_name="marvel8", match_date="27-08-2023")
        self.add_camera(camera_name="marvel9", match_date="27-08-2023")
        self.add_camera(camera_name="marvel10", match_date="27-08-2023")

    @staticmethod
    def calculate_midpoint(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        return mean([x1, x2]), mean([y1, y2])

    def camera_count(self) -> int:
        """Returns the number of cameras in the MultiCameraTracker object."""
        return len(self.cameras)

    def add_camera(self, camera_name: str = "Jetson1", match_date: str = "01-01-2023") -> None:
        """
        Adds a camera to the MultiCameraTracker object.
        """
        if not isinstance(camera_name, str):
            raise ValueError("camera_name must be a string, such as 'Jetson1' or 'Marvel1'")
        if not isinstance(match_date, str):
            raise ValueError("match_date must be a string, such as '01-01-2023'")

        camera_name = camera_name.lower()

        cam = CameraModel(camera_name, match_date, camera_coords_json_path=self.camera_coords_json_path)
        self.cameras[str(cam.id)] = cam
        self.pitch_boundary_path[str(cam.id)] = self._create_pitch_path(str(cam.id))
        self._set_homography(str(cam.id))

    def _set_homography(self, camera_id: str) -> None:
        if camera_id not in self.cameras:
            raise ValueError(f"Camera ID {camera_id} not found.")
        camera_model = self.cameras[str(camera_id)]
        estimator = HomographyEstimator(camera_model, self.pitch_model)
        self.homography_matrices[camera_id] = estimator.estimate_homography()

    def _create_pitch_path(self, camera_id: str) -> Path:
        """
        Create a path representing the boundary of the pitch for a given camera.
        :param camera_id (str): The ID of the camera.
        :return: A Path object representing the pitch boundary.
        """
        pitch_boundary: List[Tuple[int, int]] = self.cameras[camera_id].image_field_coordinates
        return Path(pitch_boundary)

    # Note: the below func is no longer needed as we mask the pitch on the camera itself now
    def _remove_oob_detections(self, _dets: List[Detections]) -> List[Detections]:
        """
        Removes detections that are out of bounds of the pitch within the image frame.
        Returns a new list of filtered detections.
        :param _dets (List[Detections]): The list of detections to check.
        :return: The new list of detections with out of bounds detections removed.
        """
        filtered_dets = []
        for _det in _dets:
            camera_id = str(_det.camera_id)
            if camera_id in self.cameras:
                if self.pitch_boundary_path[str(_det.camera_id)].contains_point((_det.x, _det.y)):
                    filtered_dets.append(_det)
                else:
                    print(f"Warning: Detection {_det} is out of bounds.")
            else:
                print(f"Warning: Camera ID {camera_id} not found in the tracker.")

        return filtered_dets

    def _filter_triangulation_result(self, result: TriangulationResult) -> Optional[Tuple[float, float, float]]:
        if result.z < self.min_z_value_for_triangulation_result:
            return None
        if result.z > self.max_z_value_for_triangulation_result:
            return None

        if result.z >= 1:
            if result.shortest_perpendicular_distance <= self.longest_perpendicular_distance_when_z_is_above_one:
                return result.x, result.y, result.z
        else:  # result.z < 1
            if result.shortest_perpendicular_distance <= self.longest_perpendicular_distance_when_z_is_below_one:
                return result.x, result.y, result.z

        return None

    def _apply_homography_transformation(self, camera_id: str, x: float, y: float) -> Tuple[float, float]:
        """
        Applies a homography transformation to a set of coordinates.
        :param x (float): The x coordinate.
        :param y (float): The y coordinate.
        :return: The transformed coordinates.
        """
        homography_matrix = self.homography_matrices[str(camera_id)]

        # Apply homography transformation
        transformed_points = homography_matrix @ np.array([[x], [y], [1.0]], dtype=object)

        # Normalize the transformed coordinates back to 2D from homogenous space
        transformed_x = transformed_points[0, 0] / transformed_points[2, 0]
        transformed_y = transformed_points[1, 0] / transformed_points[2, 0]

        return transformed_x, transformed_y

    def _perform_homography(self, _detections: List[Detections]) -> List[Detections]:
        """
        Performs a homography transformation on a list of detections.
        :param _detections (List[Detections]): The list of detections to transform.
        :return: The transformed list of detections.
        """
        transformed_dets = []
        for _det in _detections:
            camera_id = str(_det.camera_id)
            if camera_id in self.cameras:
                _det.x, _det.y = self._apply_homography_transformation(camera_id, _det.x, _det.y)
                transformed_dets.append(_det)
            else:
                print(f"Warning: Camera ID {camera_id} not found in the tracker.")

        return transformed_dets

    def _camera_coords_to_np_array(self, camera_id: int) -> np.ndarray:
        coords = self.cameras[str(camera_id)].real_world_camera_coords
        return np.array([[coords["real_world_x"]], [coords["real_world_y"]], [coords["real_world_z"]]])

    # TODO: note, for now return just one ball, but later on we'll want to incorporate better error handling
    def one_camera_triangulation(self, _detections: List[Detections]) -> Detections:
        # TODO: come back to look at implementing the plane stuff if I can get it work.
        # if self.plane is None:
        #     # TODO: We would try to fill in the plane here, but pass for now
        #     pass
        #
        # if self.plane is not None:
        #     # Now compute where the ball instersects the plane and return that
        #     if np.all(self.plane == 0):  # Check if the ball was still in previous frames (i.e. plane is all 0's
        #         return _detections
        #     else:
        #         pass
        # else:
        #     # Just return the detection as is.
        #     return _detections#

        # TODO: note that this should only be receiving a single det (its one camera!)!!!
        #  This needs refactoring
        for det in _detections:
            # Check if its within the buffer radius of the last triangulated position
            if self.triangulated_position_where_ball_was_last_detected_by_two_or_more_cameras:
                # Get the euclidean distance in x and y between the two points
                euclidean_distance = np.linalg.norm(np.array([det.x, det.y]) - np.array([self.triangulated_position_where_ball_was_last_detected_by_two_or_more_cameras.x, self.triangulated_position_where_ball_was_last_detected_by_two_or_more_cameras.y]))
                if euclidean_distance <= self.buffer_radius:
                    return det
                else:
                    return None
        return _detections[0]

    def n_camera_triangulation(self, _detections: List[Detections]) -> Optional[Detections]:
        # Ensure there are more than two detections
        if len(_detections) < 2:
            raise ValueError("At least three camera detections are required for n-camera triangulation")

        triangulation_points = []
        # Iterate over all unique pairs of cameras
        for i in range(len(_detections)):
            for j in range(i + 1, len(_detections)):
                camera_i_coords = self._camera_coords_to_np_array(_detections[i].camera_id)
                camera_j_coords = self._camera_coords_to_np_array(_detections[j].camera_id)
                print(_detections[i], camera_i_coords, _detections[j], camera_j_coords, "\n")
                triangulation_result = self.triangulate(_detections[i], camera_i_coords, _detections[j], camera_j_coords)
                # filtered_triangulation_result = self._filter_triangulation_result(triangulation_result)
                filtered_triangulation_result = triangulation_result
                # if filtered_triangulation_result:
                #     print(f"filtered_triangulation_result: {filtered_triangulation_result}")
                #     self.ball_has_been_detected_by_two_or_more_cameras_within_time_frame = True
                #     self.multi_camera_det_counter = 0
                triangulation_points.append(np.array([filtered_triangulation_result[0], filtered_triangulation_result[1], filtered_triangulation_result[2]]))

        # Calculate the centroid of all triangulation points
        if not triangulation_points:
            return None

        final_position = np.mean(triangulation_points, axis=0)

        averaged_det = Detections(camera_id=0, x=final_position[0], y=final_position[1], z=final_position[2])

        if self.triangulated_position_where_ball_was_last_detected_by_two_or_more_cameras:
            self.triangulated_position_where_ball_was_last_detected_by_two_or_more_cameras = deepcopy(averaged_det)
        return averaged_det

    def two_camera_triangulation(self, _detections: List[Detections]) -> Optional[Detections]:
        self.plane = None  # destroy the plane

        camera1_coords = self._camera_coords_to_np_array(_detections[0].camera_id)
        camera2_coords = self._camera_coords_to_np_array(_detections[1].camera_id)

        triangulation_result = self.triangulate(_detections[0], camera1_coords, _detections[1], camera2_coords)
        # filtered_triangulation_result = self._filter_triangulation_result(triangulation_result)
        filtered_triangulation_result = triangulation_result  # Temp commit to remove two camera constraint
        # if filtered_triangulation_result:
        #     self.ball_has_been_detected_by_two_or_more_cameras_within_time_frame = True
        #     self.multi_camera_det_counter = 0
        three_d_det = Detections(camera_id=0, x=filtered_triangulation_result[0], y=filtered_triangulation_result[1], z=filtered_triangulation_result[2])
        # self.triangulated_position_where_ball_was_last_detected_by_two_or_more_cameras = deepcopy(three_d_det)
        return three_d_det

    def _set_latest_camera_id(self, _detections: List[Detections]) -> None:
        if len(_detections) > 0:
            self.latest_camera_id = _detections[0].camera_id

    def _apply_optical_flow_to_ball(self, average_flow: Dict[str, np.ndarray]) -> Detections:

        print("applying optical flow to ball")

        last_triangulated_pos = self.last_triangulated_position

        # TODO: What to do?
        # Note: when we do this, optical flow is not being used on purpose
        # The average flow will be None when it has been used for more than 10 frames.
        last_camera_optical_flow = average_flow[f"marvel{self.latest_camera_id}"]

        optical_flow_starting_position = last_triangulated_pos.x, last_triangulated_pos.y

        optical_flow_end_point =(int(optical_flow_starting_position[0] + last_camera_optical_flow[0] * OPTICAL_FLOW_MULTIPLIER), int(optical_flow_starting_position[1] + last_camera_optical_flow[1] * OPTICAL_FLOW_MULTIPLIER))

        # Perform homography on both optical_flow_starting_position and optical_flow_end_point
        homographied_starting_positioon = self._apply_homography_transformation(self.latest_camera_id, optical_flow_starting_position[0], optical_flow_starting_position[1])
        homographied_end_position = self._apply_homography_transformation(self.latest_camera_id, optical_flow_end_point[0], optical_flow_end_point[1])

        # Normalize these positions to a direction vector from the origin
        normalized_flow_vector = (homographied_end_position[0] - homographied_starting_positioon[0], homographied_end_position[1] - homographied_starting_positioon[1])

        # Add this to our last_triangulated_pos
        new_triangulated_pos =  (last_triangulated_pos.x + normalized_flow_vector[0], last_triangulated_pos.y + normalized_flow_vector[1])

        # Makeit a Detections object
        new_triangulated_pos = Detections(camera_id=last_triangulated_pos.camera_id, x=new_triangulated_pos[0], y=new_triangulated_pos[1], z=last_triangulated_pos.z)

        return new_triangulated_pos

    def multi_camera_analysis(self, _detections: List[Detections], average_flow: Dict[str, Optional[np.ndarray]]) -> Optional[Detections]:
        """
        Triangulates a list of detections.
        :param _detections (List[Detections]): The list of detections to triangulate.
        :return: The triangulated list of detections.
        """
        # TODO: removing temporarily as I don't have the image field coords set up for AFL yet.
        self._set_latest_camera_id(_detections)

        _detections = self._perform_homography(_detections)

        # Temp to make things easier
        _detections = filter_most_confident_dets(_detections)

        # Create a list with the different camera ids
        cam_list = set(det.camera_id for det in _detections)

        if len(cam_list) == 0:
            triangulated_det = None
            # Check if all values in average_flow are not None
            if not all(value is None for value in average_flow.values()):  # TODO: Note I don't think they all need to be None in practice...
                if average_flow.get(f"{self.latest_camera_id}") is not None:
                    print("!")
                    print("Apply optical flow value to the triangulated position")
                    print("!")
                    triangulated_det = self._apply_optical_flow_to_ball(average_flow)  # Only use when no detections are found
                    self.optical_flow_used = True
                else:

                    triangulated_det = None
                    self.optical_flow_used = False
        else:
            if self.optical_flow_used:
                self.optical_flow_used = False
            if len(cam_list) == 1:
                triangulated_det = self.one_camera_triangulation(_detections)
            elif len(cam_list) == 2:
                triangulated_det = self.two_camera_triangulation(_detections)
            else:
                triangulated_det = self.n_camera_triangulation(_detections)

        if triangulated_det is not None:

            self.consecutive_detections += 1

            # Reset the bool which tracks if the ball has been detected by 2+ cameras within the last while.
            if self.multi_camera_det_counter >= self.acceptable_time_frame_for_single_cam_det_after_two_or_more_cam_dets:
                self.ball_has_been_detected_by_two_or_more_cameras_within_time_frame = False

            # Increment the counter for how many frames since the ball was last detected by two or more cameras
            if self.ball_has_been_detected_by_two_or_more_cameras_within_time_frame:
                self.multi_camera_det_counter += 1

            if self.consecutive_detections >= 2:
                self.kalman_filter.predict()
                self.kalman_filter.update(np.array([triangulated_det.x, triangulated_det.y, triangulated_det.z]))
                triangulated_det.x, triangulated_det.y, triangulated_det.z = self.kalman_filter.state
            else:
                # TODO: I'm not sure if this is right... immediately resetting the Kalman filter on a first new detection...
                self.kalman_filter.reset(np.array([triangulated_det.x, triangulated_det.y, triangulated_det.z]))

            self.consecutive_non_detections = 0
            self.last_triangulated_position = triangulated_det
            return triangulated_det
        else:
            self.consecutive_detections = 0
            self.consecutive_non_detections += 1
            if self.consecutive_non_detections >= self.max_consecutive_misses:  # Reset the kalman filter after 3 misses
                # TODO: I think this is misleading
                #  How does optical flow effect when this gets called?
                self.kalman_filter.reset(np.zeros(3))  # Reset the Kalman filter
            return None

    @staticmethod
    def triangulate(ball_p: Detections, cam_p: np.ndarray, ball_q: Detections, cam_q: np.ndarray) -> TriangulationResult:
        """
        Performs mid-point triangulation based on the positions of a ball from two different camera perspectives.

        This function calculates the 3D coordinates of a point (ball) as observed from two different camera positions
        using the mid-point method of triangulation. The coordinates are assumed to be in a real-world coordinate
        system.
        https://en.wikipedia.org/wiki/Triangulation_(computer_vision)#Mid-point_method

        Parameters:
        ball_p (Detections): The ball's coordinates from the perspective of camera P.
        cam_p (np.ndarray): The real-world position of camera P.
        ball_q (Detections): The ball's coordinates from the perspective of camera Q.
        cam_q (np.ndarray): The real-world position of camera Q.

        Returns:
        List[float]: The estimated real-world coordinates of the ball as a list [x, y, z].
        """
        # Convert Detections to numpy arrays
        XA0 = np.array([ball_p.x, ball_p.y, ball_p.z], dtype=float)
        XA1 = np.squeeze(cam_p)
        XB0 = np.array([ball_q.x, ball_q.y, ball_q.z], dtype=float)
        XB1 = np.squeeze(cam_q)

        # Calculate direction vectors for each line
        VA = XA1 - XA0  # V1
        VB = XB1 - XB0  # V2

        # Calculate V3 as the cross product of VB and VA
        V3 = np.cross(VB, VA)  # V3 = V2 x V1

        # Coefficients matrix for the system of equations derived from
        # P1 + t1V1 + t3V3 = P2 + t2V2
        coefficients = np.array([VA, -VB, V3]).T  # The matrix is transposed to match the equation

        # Constants vector is the displacement from P1 to P2
        constants = XB0 - XA0

        # Use least squares to solve for t1, t2, and t3
        params, _, _, _ = lstsq(coefficients, constants, rcond=None)
        t1, t2, t3 = params  # Extract the parameters

        # Find the closest points on each line using the parameters
        closest_point_on_A = XA0 + t1 * VA  # Q1 = P1 + t1V1
        closest_point_on_B = XB0 + t2 * VB  # Q2 = P2 + t2V2

        distance_between_A_and_B = np.linalg.norm(closest_point_on_A - closest_point_on_B)

        # print(f"closest_point_on_A: {closest_point_on_A}")
        # print(f"closest_point_on_B: {closest_point_on_B}")
        print(f"distance_between_A_and_B: {distance_between_A_and_B}")

        # Calculate the midpoint between the closest points
        midpoint = (closest_point_on_A + closest_point_on_B) / 2

        return TriangulationResult(
            x=midpoint[0].item(),
            y=midpoint[1].item(),
            z=midpoint[2].item(),
            shortest_perpendicular_distance=distance_between_A_and_B
        )


def main():
    tracker = MultiCameraTracker(sport="afl", camera_coords_json_path="../data/homography_data/afl_camera_coordinates.json")
    tracker.add_camera(camera_name="marvel1", match_date="20-08-2023")

    x = Detections(camera_id=1, x=10, y=20, z=0)
    y = Detections(camera_id=1, x=1070, y=88)

    tracker.multi_camera_analysis([x, y], {})


if __name__ == '__main__':
    main()