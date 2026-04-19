import pyrealsense2 as rs
import numpy as np
import cv2
from spatialmath import SE3, SO3

from ur5e_robot import UR5Robot
from aruco_detect import ArucoDetector
from PBVS import PBVSController, fix_rotation_matrix
from servo_utils import plot_error_and_increment

# ================= Helper conversion functions =================
def ur_pose_to_se3(ur_pose: list) -> SE3:
    """UR [x,y,z,rx,ry,rz] -> SE3"""
    t = ur_pose[:3]
    rvec = np.array(ur_pose[3:], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    return SE3.Rt(R, t, check=False)

def se3_to_ur_pose(T: SE3) -> list:
    """SE3 -> UR [x,y,z,rx,ry,rz]"""
    t = T.t
    rvec, _ = cv2.Rodrigues(T.R)
    return [t[0], t[1], t[2], rvec[0,0], rvec[1,0], rvec[2,0]]

def get_realsense_intrinsics(profile):
    """Extract intrinsic matrix from RealSense profile"""
    intr = profile.as_video_stream_profile().get_intrinsics()
    # camera_matrix = np.array([
    #     [intr.fx, 0, intr.ppx],
    #     [0, intr.fy, intr.ppy],
    #     [0, 0, 1]
    # ], dtype=np.float32)
    camera_matrix = np.array([
        [600.14094118, 0., 325.03318006],
        [0., 600.23725494, 230.2565916],
        [0., 0., 1.],
    ])
    dist_coeffs = np.array(intr.coeffs, dtype=np.float32)
    return camera_matrix, dist_coeffs

class EMAPoseFilter:
    def __init__(self, alpha=0.3):
        """
        alpha: Filter coefficient (0.0 to 1.0).
        Smaller values are smoother but add more delay;
        larger values react faster but jitter more. 0.3 is a good starting point.
        """
        self.alpha = alpha
        self.filtered_pose = None

    def update(self, new_pose: SE3) -> SE3:
        if self.filtered_pose is None:
            self.filtered_pose = new_pose
            return new_pose

        # EMA filter for translation
        t_new = self.alpha * new_pose.t + (1 - self.alpha) * self.filtered_pose.t
        
        # Slerp-based filtering for rotation
        R_curr = SO3(fix_rotation_matrix(self.filtered_pose.R))
        R_new = SO3(fix_rotation_matrix(new_pose.R))
        R_filtered = R_curr.interp(R_new, self.alpha)

        self.filtered_pose = SE3.Rt(R_filtered.R, t_new, check=False)
        return self.filtered_pose

# ================================================

def main():
    # 1. Initialize robot
    robot = UR5Robot(tcp_host="192.168.1.20", gripper=None)
    
    # 2. Configure RealSense pipeline (640x480)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("Starting RealSense D435i...")
    cfg = pipeline.start(config)
    profile = cfg.get_stream(rs.stream.color)
    
    # 3. Automatically get camera intrinsics
    camera_matrix, dist_coeffs = get_realsense_intrinsics(profile)
    print("Retrieved RealSense intrinsics:\n", camera_matrix)

    # 4. Initialize detector and controller
    # Note: marker_size must match the actual printed ArUco size (unit: meters)
    aruco_detector = ArucoDetector(image_width=640, image_height=480, marker_size=0.05)
    aruco_detector.set_camera_params(camera_matrix, dist_coeffs)
    
    pbvs_controller = PBVSController(
        kp=0.3,                       # Lower initial gain
        max_translation_step=0.01,    # Max 1 cm per step
        max_rotation_step=np.deg2rad(2),
        position_error_threshold=0.002, 
        rotation_error_threshold=np.deg2rad(1.0)
    )

    # 5. Load hand-eye calibration matrix (from camera2gripper.txt)
    T_ee_cam = SE3(robot.cam2gripper_mat)

    # Initialize filter
    pose_filter = EMAPoseFilter(alpha=0.3)

    # Desired pose: camera is 30 cm above marker (0.30 m)
    T_desired_cam_aruco = SE3(0.0, 0.0, 0.30) * SE3.Rz(0.0)

    errors = []
    increments = []

    try:
        while True:
            # A. Wait for image frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: 
                continue
            bgr = np.asanyarray(color_frame.get_data())

            # B. Estimate pose
            T_cam_aruco = aruco_detector.estimate_pose(bgr, debug=False)
            aruco_detector.draw(bgr, None, T_cam_aruco) 
            
            if T_cam_aruco is None:
                cv2.imshow("RealSense PBVS", bgr)
                if cv2.waitKey(1) == 27: break
                continue

            # Apply filtering to smooth ArUco jitter
            T_cam_aruco = pose_filter.update(T_cam_aruco)
            
            # C. Get current robot state
            current_tcp_ur = robot.get_current_tcp()
            T_base_ee = ur_pose_to_se3(current_tcp_ur)

            # D. PBVS core computation
            T_base_target = T_base_ee * T_ee_cam * T_cam_aruco * T_desired_cam_aruco.inv() * T_ee_cam.inv()

            # E. Controller computes step
            status, T_base_ee_des = pbvs_controller.step(T_base_ee, T_base_target)

            # F. Record error
            error_pose = T_base_ee.inv() * T_base_target
            pos_error = np.linalg.norm(error_pose.t)
            errors.append(pos_error)
            increments.append((T_base_ee_des.t - T_base_ee.t) * 1000.0)

            print(f"Error: {pos_error*1000:.2f} mm")

            if status == pbvs_controller.RESULT_FINISHED:
                print("Target pose reached!")
                break

            # G. Execute motion
            target_tcp_ur = se3_to_ur_pose(T_base_ee_des)
            robot.servo_l(target_tcp_ur, time=0.033, lookahead_time=0.15)

            if cv2.waitKey(1) == 27: break # ESC to exit

    finally:
        pipeline.stop()
        robot.shutdown()
        cv2.destroyAllWindows()
        if errors:
            plot_error_and_increment(errors, increments)

if __name__ == "__main__":
    main()