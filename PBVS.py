import numpy as np
from typing import Tuple, Optional
from spatialmath import SE3, SO3, UnitQuaternion

def fix_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Fix a rotation matrix to ensure it is a valid element of SO(3)
    by enforcing orthogonality and det=+1.

    Args:
        R (np.ndarray): 3x3 rotation matrix (may be non-orthogonal due to numerical errors)

    Returns:
        np.ndarray: Fixed orthogonal rotation matrix with det=+1
    """
    # Use SVD to find the closest orthogonal matrix
    U, _, Vh = np.linalg.svd(R)
    R_fixed = U @ Vh

    # Ensure determinant is +1 (correct reflection if needed)
    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vh

    return R_fixed


class PBVSController:
    """
    Position-Based Visual Servoing controller.
    """

    # Status constants
    RESULT_WORKING = 0
    RESULT_FINISHED = 1
    RESULT_ERROR_INVALID_INPUT = -1

    def __init__(
        self,
        kp: float = 0.1,  # Proportional gain
        max_translation_step: float = 0.05,  # Maximum translation per step (m)
        max_rotation_step: float = 0.1,      # Maximum rotation per step (rad)
        workspace_min: np.ndarray = np.array([0.01, -1.0, -1.0]),  # Workspace lower bound [x, y, z]
        workspace_max: np.ndarray = np.array([1.0, 1.0, 1.0]),     # Workspace upper bound [x, y, z]
        position_error_threshold: float = 0.001,  # Position convergence threshold (m)
        rotation_error_threshold: float = 0.01,   # Rotation convergence threshold (rad)
    ):
        self.kp = float(kp)
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step = float(max_rotation_step)
        self.workspace_min = np.array(workspace_min)
        self.workspace_max = np.array(workspace_max)
        self.position_error_threshold = float(position_error_threshold)
        self.rotation_error_threshold = float(rotation_error_threshold)
        self.counter = 0

    def _interp(self, current: SE3, target: SE3, s_position: float, s_orientation: float) -> SE3:
        """
        Interpolate between current pose and target pose.
        Translation uses linear interpolation; rotation uses SO3 interpolation.

        Args:
            current (SE3): Current pose
            target (SE3): Target pose
            s_position (float): Interpolation factor for position (0 to 1)
            s_orientation (float): Interpolation factor for orientation (0 to 1)
        Returns:
            SE3: Interpolated pose
        """
        # 1. Translation interpolation
        out_translation = (1 - s_position) * current.t + s_position * target.t

        # 2. Rotation interpolation
        R_current = SO3(fix_rotation_matrix(current.R))
        R_target = SO3(fix_rotation_matrix(target.R))
        out_R = R_current.interp(R_target, s_orientation)

        # 3. Compose output pose
        return SE3.Rt(out_R, out_translation, False)

    def _interp1(self, end: SE3, s_position: float, s_orientation: float) -> SE3:
        """
        Interpolate from identity pose to end pose.

        Args:
            end (SE3): Target pose
            s_position (float): Interpolation factor for position (0 to 1)
            s_orientation (float): Interpolation factor for orientation (0 to 1)
        Returns:
            SE3: Interpolated pose
        """
        start = SE3()  # Identity by default
        return self._interp(start, end, s_position, s_orientation)

    def _pid_controller(self, current_pose: SE3, desired_pose: SE3) -> SE3:
        """
        Compute pose increment using a PID-like controller based on the error between current and desired pose.

        Args:
            current_pose (SE3): Current robot end-effector pose
            desired_pose (SE3): Desired target pose
        Returns:
            SE3: Pose increment to apply
        """
        # Compute error pose in the current frame: T_err = T_curr^-1 * T_des
        error_pose = current_pose.inv() * desired_pose
        error_pose.R = fix_rotation_matrix(error_pose.R)

        # Translation error magnitude
        position_error_magnitude = np.linalg.norm(error_pose.t)
        # Rotation error magnitude: safely compute angular difference
        rotation_error_magnitude = SO3(error_pose.R).angvec()[0]

        # 4. Gain and step limiting
        kp_position = self.kp
        kp_orientation = self.kp

        # Small epsilon to avoid division by zero
        EPS = 1e-9
        if position_error_magnitude > EPS:
            if position_error_magnitude * self.kp > self.max_translation_step:
                kp_position = self.max_translation_step / position_error_magnitude
        else:
            kp_position = 0.0

        if rotation_error_magnitude > EPS:
            if rotation_error_magnitude * self.kp > self.max_rotation_step:
                kp_orientation = self.max_rotation_step / rotation_error_magnitude
        else:
            kp_orientation = 0.0

        # 5. Compute incremental pose
        pose_incr = self._interp1(error_pose, kp_position, kp_orientation)

        return pose_incr

    def step(self, current_pose: SE3, target_pose: SE3) -> Tuple[int, SE3]:
        """
        Run one visual servoing control step.

        Args:
            current_pose: Current robot end-effector pose (SE3)
            target_pose: Desired target pose (SE3)

        Returns:
            Tuple[int, SE3]: (status code, next pose)
            Status code: 0 (WORKING), 1 (FINISHED), -1 (ERROR)
        """
        self.counter += 1

        # 1. Check invalid input (NaN or Inf)
        if not np.all(np.isfinite(current_pose.A)) or not np.all(np.isfinite(target_pose.A)):
            print("[PIDController] Invalid input: NaN or Inf detected. Skipping this step.")
            return self.RESULT_ERROR_INVALID_INPUT, current_pose

        # 2. Compute increment using PID-like controller
        pose_incr = self._pid_controller(current_pose, target_pose)

        # 3. Compute output pose
        new_pose = current_pose * pose_incr

        # 4. Workspace constraints
        clamped_translation = np.clip(new_pose.t, self.workspace_min, self.workspace_max)
        new_pose = SE3.Rt(new_pose.R, clamped_translation, False)

        # 5. Check convergence based on error
        error_pose_final = current_pose.inv() * target_pose
        position_error_magnitude = np.linalg.norm(error_pose_final.t)
        rotation_error_magnitude = SO3(fix_rotation_matrix(error_pose_final.R)).angvec()[0]

        if (
            position_error_magnitude < self.position_error_threshold
            and rotation_error_magnitude < self.rotation_error_threshold
        ):
            print(f"PositionBasedVisualServoController finished. counter: {self.counter}")
            return self.RESULT_FINISHED, current_pose

        return self.RESULT_WORKING, new_pose


if __name__ == "__main__":
    # Initialize controller parameters
    controller = PBVSController(
        kp=0.05,
        max_translation_step=0.02,  # 2 cm
        max_rotation_step=np.deg2rad(5),  # 5 degrees
        position_error_threshold=0.001,  # 1 mm
        rotation_error_threshold=np.deg2rad(0.5),  # 0.5 degrees
    )

    # Define initial pose (at origin, no rotation)
    current_pose = SE3()

    # Define target pose (translation [0.1, 0.2, 0.3], rotate around Z by 10 degrees)
    target_pose = SE3(0.1, 0.2, 0.3) * SE3.Rz(10, unit="deg")

    print(f"Initial Pose:\n{current_pose}")
    print(f"Target Pose:\n{target_pose}")
    print("-" * 50)

    # Simulate control loop
    for i in range(200):
        status, new_pose = controller.step(current_pose, target_pose)

        if status == controller.RESULT_FINISHED:
            print("\nConverged!")
            break
        elif status == controller.RESULT_ERROR_INVALID_INPUT:
            print("\nError occurred!")
            break

        # Update current pose
        current_pose = new_pose

        # Compute current error
        error_pose = current_pose.inv() * target_pose
        pos_error = np.linalg.norm(error_pose.t)
        rot_error = SO3(error_pose.R).angvec()[0]

        print(
            f"Step {i+1:3d}: Position Error = {pos_error:.6f} m, "
            f"Rotation Error = {rot_error:.6f} rad ({np.rad2deg(rot_error):.2f}°)"
        )

    print("-" * 50)
    print(f"Final Pose:\n{current_pose}")

    # Compute final error
    final_error_pose = current_pose.inv() * target_pose
    final_pos_error = np.linalg.norm(final_error_pose.t)
    final_rot_error = SO3(final_error_pose.R).angvec()[0]

    print("\n" + "=" * 50)
    print("Final Error Summary:")
    print(f"Position error: {final_pos_error:.6f} m (threshold: {controller.position_error_threshold} m)")
    print(
        f"Rotation error: {final_rot_error:.6f} rad ({np.rad2deg(final_rot_error):.2f}°) "
        f"(threshold: {controller.rotation_error_threshold:.6f} rad)"
    )
    print("=" * 50)
