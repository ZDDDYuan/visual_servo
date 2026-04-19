import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, List, Optional
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive



class BaseGripper(ABC):
    @abstractmethod
    def open(self):
        """Open the gripper"""
        pass

    @abstractmethod
    def close(self):
        """Close the gripper"""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect and release resources"""
        pass


class UR5Robot:
    def __init__(
        self,
        tcp_host: str = "192.168.1.20",
        gripper: Optional[BaseGripper] = None,
        calib_path: Optional[str] = None,
    ):
        # Initialization
        self.tcp_host = tcp_host
        self.tcp_socket = None
        self.gripper = gripper

        self.rtde_c = RTDEControl(self.tcp_host)
        self.rtde_r = RTDEReceive(self.tcp_host)

        # Define the acceleration and velocity of joint and tool
        self.joint_acc, self.joint_vel = 0.5, 0.2
        self.tool_acc, self.tool_vel = 0.5, 0.2

        self.cam2gripper_mat = self._load_calib_matrix(calib_path)
        self.cam_depth_scale = 0.0010000000474974513

    def _load_calib_matrix(self, calib_path: Optional[str]) -> np.ndarray:
        """
        Load the camera-to-gripper calibration matrix from a specified path.

        Args:
            calib_path: The file path to the calibration matrix. If None, a default path will be used.
        Returns:
            np.ndarray: The loaded calibration matrix.
        """
        try:
            if calib_path and Path(calib_path).is_file():
                print(f"[INFO] Loading calibration matrix from {calib_path}")
                return np.loadtxt(calib_path)

            base_dir = Path("./calib_handeye_data")
            if base_dir.is_dir():
                calib_files = list(base_dir.rglob("*/camera2gripper.txt"))
                if calib_files:
                    latest_calib_file = max(calib_files, key=lambda f: f.stat().st_mtime)
                    print(f"[INFO] Loading calibration matrix from {latest_calib_file}")
                    return np.loadtxt(latest_calib_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load calibration matrix: {e}")

    def shutdown(self) -> None:
        """
        Shutdown the UR5e robot.
        """
        # RTDE connection usually does not need to be closed explicitly,
        # so close the gripper connection if it exists.
        if self.gripper:
            self.gripper.disconnect()

    def get_current_tcp(self) -> List[float]:
        """
        Get current tcp pose.
        """
        return self.rtde_r.getActualTCPPose()

    def get_current_joint_config(self) -> Tuple[float, ...]:
        """
        Get current joint configuration.
        """
        return self.rtde_r.getActualQ()

    def move_j_ik(
        self,
        target_tcp_pose: Sequence[float],
        k_acc: float = 1.0,
        k_vel: float = 1.0,
    ) -> None:
        """
        Calculate the inverse kinematics and move the UR5 robot to a specified pose.

        Args:
            target_tcp_pose: A sequence of 6 values representing the target TCP pose (x, y, z in meters and rx, ry, rz in radians).
            k_acc: A scaling factor for the joint acceleration (default: 1.0).
            k_vel: A scaling factor for the joint velocity (default: 1.0).
        Returns:
            None
        """
        self.rtde_c.moveJ_IK(target_tcp_pose, acceleration=k_acc * self.joint_acc, speed=k_vel * self.joint_vel)

    def move_l(
        self,
        target_tcp_pose: Sequence[float],
        k_acc: float = 1.0,
        k_vel: float = 1.0,
    ) -> None:
        """
        Linear move the UR5 robot to a specified pose.

        Args:
            target_tcp_pose: A sequence of 6 values representing the target TCP pose (x, y, z in meters and rx, ry, rz in radians).
            k_acc: A scaling factor for the tool acceleration (default: 1.0).
            k_vel: A scaling factor for the tool velocity (default: 1.0).
        Returns:
            None
        """
        self.rtde_c.moveL(target_tcp_pose, acceleration=k_acc * self.tool_acc, speed=k_vel * self.tool_vel)

    def servo_l(
        self,
        target_tcp_pose: Sequence[float],
        time: float = 0.033,
        lookahead_time: float = 0.1,
        gain: float = 300
    ) -> None:
        """
        Non-blocking real-time servo control, suitable for visual tracking.

        Args:
            target_tcp_pose: A sequence of 6 values representing the target TCP pose (x, y, z in meters and rx, ry, rz in radians).
            time: The time step for each servo command, typically matching your camera frame rate (e.g., 0.033s for 30fps).
            lookahead_time: The lookahead time for the servo controller, smaller values make it more responsive, 
                            larger values make it smoother (0.1~0.2 is a reasonable range).
            gain: The proportional gain for the servo controller, typically between 100 and 500.
        Returns:
            None
        """
        self.rtde_c.servoL(target_tcp_pose, self.tool_vel, self.tool_acc, time, lookahead_time, gain)


if __name__ == "__main__":
    ur5_robot = UR5Robot(gripper=None)
    curr_tcp = ur5_robot.get_current_tcp()
    print(curr_tcp)
    curr_joint_config = ur5_robot.get_current_joint_config()
    print(np.degrees(curr_joint_config))

    target_tcp_pose = [0.15, -0.28039290004612755, 0.1833946785635515,
                       0.0, 0.0, 0.0]
    ur5_robot.move_j_ik(target_tcp_pose)
    ur5_robot.shutdown()
