import cv2
import numpy as np
import spatialmath as sm
import time

class ArucoDetector:
    def __init__(self, image_width=720, image_height=720, marker_size=0.1, desired_size=150.0):

        # Dictionary (4x4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params
        )
        self.marker_size = marker_size

        # Compute desired corner points (blue crosses)
        cx, cy = image_width / 2.0, image_height / 2.0
        half = desired_size / 2.0
        self.desired_corners = np.array([
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ], dtype=np.float32)

        # Set camera intrinsics (computed from fovy=45)
        if image_width == 640 and image_height == 480:
            # 640x480 TODO
            self.camera_matrix = np.array([
                [772.55,   0.0, 320.0],
                [0.0,    579.41, 240.0],
                [0.0,      0.0,   1.0]
            ], dtype=np.float32)
        elif image_width == 1280 and image_height == 720:
            # 1280x720 TODO
            self.camera_matrix = np.array([
                [1545.1,   0.0, 360.0],
                [0.0,    869.12, 360.0],
                [0.0,      0.0,   1.0]
            ], dtype=np.float32)
        elif image_width == 720 and image_height == 720:
            # 720x720
            self.camera_matrix = np.array([
                [869.12,   0.0, 360.0],
                [0.0,    869.12, 360.0],
                [0.0,      0.0,   1.0]
            ], dtype=np.float32)
        else:
            self.camera_matrix = None

        self.dist_coeffs = np.zeros(5, dtype=np.float32)

    def set_camera_params(self, camera_matrix, dist_coeffs=None):
        self.camera_matrix = camera_matrix
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs

    def detect(self, image_bgr, debug=False):
        """
        Input: BGR image (H,W,3)
        Output: four corner points (4,2) or None
        """
        if image_bgr is None:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(corners) == 0:
            if debug:
                print("No ArUco detected")
            return None

        if debug:
            print("Detected corners: ", corners)

        return corners[0][0].astype(np.float32)  # shape (4, 2)

    @staticmethod
    def rvec_tvec_to_SE3(rvec, tvec):
        """
        Convert OpenCV solvePnP output (rvec, tvec) to spatialmath.SE3
        """
        R, _ = cv2.Rodrigues(rvec)          # (3,3)
        t = tvec.reshape(3)                 # (3,)
        T = sm.SE3.Rt(R, t, False)
        return T

    def estimate_pose(self, image_bgr, debug=False):
        """
        Estimate ArUco pose relative to the camera using PnP,
        and return SE3 (ArUco pose in camera frame).

        :param image_bgr: input BGR image
        :return:
            success: SE3 object
            failed:  None
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("camera_matrix / dist_coeffs not set, call set_camera_params() first")

        img_points = self.detect(image_bgr, debug=debug)
        if img_points is None:
            return None

        half = self.marker_size / 2.0
        # Build 3D ArUco corners in marker frame: +x right, +y up
        obj_points = np.array([
            [-half, -half, 0.0],   # top-left
            [half,  -half, 0.0],   # top-right
            [half,   half, 0.0],   # bottom-right
            [-half,  half, 0.0],   # bottom-left
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            if debug:
                print("solvePnP failed")
            return None

        if debug:
            print("rvec:", rvec.ravel())
            print("tvec:", tvec.ravel())

        T_cam_aruco = self.rvec_tvec_to_SE3(rvec, tvec)
        return T_cam_aruco

    def draw(self, image_bgr, detected_corners=None, T_cam_aruco=None):
        img = image_bgr.copy()

        # ---- 1. Draw detected ArUco ----
        if detected_corners is not None:
            pts = detected_corners.astype(np.int32)

            # Draw bounding box
            cv2.polylines(img, [pts], True, (0, 255, 0), 3)

            # Draw corner indices
            for i, (x, y) in enumerate(pts):
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(img, str(i), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            # ---- 2. Draw desired corners (blue) ----
            for (x, y) in self.desired_corners.astype(np.int32):
                cv2.drawMarker(
                    img,
                    (x, y),
                    (255, 0, 0),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=20,
                    thickness=2
                )
            cv2.imwrite(f"data/test_aruco_detect_{time.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png", img)

        # ---- 3. Draw 3D coordinate axes (pose visualization) ----
        if T_cam_aruco is not None:
            # Extract rotation matrix and translation vector from SE3
            R = T_cam_aruco.R
            t = T_cam_aruco.t.reshape(3, 1)

            # Define 3D coordinate axes (length=marker_size/2)
            marker_half_size = self.marker_size / 2.0
            axis_3d = np.array([
                [0, 0, 0],
                [marker_half_size, 0, 0],  # x-axis (red)
                [0, marker_half_size, 0],  # y-axis (green)
                [0, 0, marker_half_size]   # z-axis (blue)
            ], dtype=np.float32)

            # Project onto image plane
            axis_2d, _ = cv2.projectPoints(axis_3d, cv2.Rodrigues(R)[0], t,
                                           self.camera_matrix, self.dist_coeffs)
            axis_2d = axis_2d.astype(np.int32)

            # Draw axes
            cv2.line(img, tuple(axis_2d[0][0]), tuple(axis_2d[1][0]), (0, 0, 255), 3)
            cv2.line(img, tuple(axis_2d[0][0]), tuple(axis_2d[2][0]), (0, 255, 0), 3)
            cv2.line(img, tuple(axis_2d[0][0]), tuple(axis_2d[3][0]), (255, 0, 0), 3)

        cv2.imshow("Aruco Detector", img)
        cv2.waitKey(1)

        return img

    def get_desired_corners(self):
        return self.desired_corners.copy()


def print_se3_pose(T: sm.SE3, name: str):
    """Print rotation matrix and translation vector separately."""
    print(f"{name}:")
    print("  Rotation matrix R:")
    print(T.R.round(4))  # rotation matrix, keep 4 decimal places
    print("  Translation vector t:")
    print(T.t.round(4))  # translation vector, keep 4 decimal places


if __name__ == "__main__":
    # Set NumPy print options
    np.set_printoptions(
        precision=6,        # keep 6 decimal places
        suppress=True,      # disable scientific notation
        linewidth=100,      # line width
        floatmode='fixed'   # fixed decimal format
    )

    detector = ArucoDetector(720, 720, 0.1)
    img = cv2.imread("test_aruco.png")  # input BGR image

    # Test detect
    corners = detector.detect(img)
    print("Detected corners: ", corners)
    print("Desired corners: ", detector.get_desired_corners())
    out = detector.draw(img, corners)

    # Test estimate_pose
    T_cam_aruco = detector.estimate_pose(img, False)
    print_se3_pose(T_cam_aruco, 'T_cam_aruco')
    out = detector.draw(img, corners, T_cam_aruco)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# class ArucoDetector:
#     def __init__(self, image_width=720, image_height=720, marker_size = 0.1, desired_size=150.0):

#         # 字典（4x4_50）
#         self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
#         self.aruco_params = cv2.aruco.DetectorParameters()
#         self.detector = cv2.aruco.ArucoDetector(
#             self.aruco_dict, self.aruco_params
#         )
#         self.marker_size = marker_size

#         # 计算期望角点（蓝色十字）
#         cx, cy = image_width / 2.0, image_height / 2.0
#         half = desired_size / 2.0
#         self.desired_corners = np.array([
#             [cx - half, cy - half],
#             [cx + half, cy - half],
#             [cx + half, cy + half],
#             [cx - half, cy + half],
#         ], dtype=np.float32)


#         # 设置相机内参（由fovy=45计算得到）
#         if image_width == 640 and image_height == 480:
#             # 640x480 TODO
#             self.camera_matrix = np.array([
#                 [772.55,   0.0, 320.0],
#                 [  0.0 , 579.41, 240.0],
#                 [  0.0 ,   0.0,   1.0 ]
#             ], dtype=np.float32)
#         elif image_width == 1280 and image_height == 720:
#             # 1280x720 TODO
#             self.camera_matrix = np.array([
#                 [1545.1,   0.0, 360.0],
#                 [  0.0 ,  869.12, 360.0],
#                 [  0.0 ,   0.0,   1.0 ]
#             ], dtype=np.float32)
#         elif image_width == 720 and image_height == 720:
#             # 720x720
#             self.camera_matrix = np.array([
#                 [869.12,   0.0, 360.0],
#                 [  0.0 ,  869.12, 360.0],
#                 [  0.0 ,   0.0,   1.0 ]
#             ], dtype=np.float32)
#         else:
#             self.camera_matrix = None

#         self.dist_coeffs = np.zeros(5, dtype=np.float32)

#     def set_camera_params(self, camera_matrix, dist_coeffs=None):
#         self.camera_matrix = camera_matrix
#         if dist_coeffs is not None : self.dist_coeffs = dist_coeffs

#     def detect(self, image_bgr, debug=False):
#         """
#         输入: BGR 图像 (H,W,3)
#         输出: 四个角点 (4,2) 或 None
#         """
#         if image_bgr is None:
#             return None

#         gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

#         corners, ids, _ = self.detector.detectMarkers(gray)

#         if ids is None or len(corners) == 0:
#             if debug:
#                 print("No ArUco detected")
#             return None
        
#         if debug:
#             print("Detected corners: ", corners)

#         return corners[0][0].astype(np.float32)  # shape (4, 2)
    
#     @staticmethod
#     def rvec_tvec_to_SE3(rvec, tvec):
#         """
#         将 OpenCV solvePnP 输出的 (rvec, tvec) 转为 spatialmath.SE3
#         """
#         R, _ = cv2.Rodrigues(rvec)          # (3,3)
#         t = tvec.reshape(3)                 # (3,)
#         T = sm.SE3.Rt(R, t, False) 
#         return T

#     def estimate_pose(self, image_bgr, debug=False):
#         """
#         使用 PnP 估计 ArUco 相对相机的位姿，并返回 SE3（相机坐标系下的 ArUco 位姿）

#         :param image_bgr: 输入 BGR 图像
#         :return:
#             success: SE3 对象
#             failed:  None
#         """
#         if self.camera_matrix is None or self.dist_coeffs is None:
#             raise ValueError("camera_matrix / dist_coeffs 未设置，请先调用 set_camera_params()")

#         img_points = self.detect(image_bgr, debug=debug)
#         if img_points is None:
#             return None

#         half = self.marker_size / 2.0
#         # 构造 ArUco 在自身坐标系下的 3D 角点， 向右为 +x 轴，向上为 +y 轴：
#         obj_points = np.array([
#             [ -half,  -half, 0.0],  # 左上
#             [ half,  -half, 0.0],  # 右上
#             [ half, half, 0.0],  # 右下
#             [-half, half, 0.0],  # 左下
#         ], dtype=np.float32)

#         success, rvec, tvec = cv2.solvePnP(
#             obj_points,
#             img_points,
#             self.camera_matrix,
#             self.dist_coeffs,
#             flags=cv2.SOLVEPNP_ITERATIVE
#         )

#         if not success:
#             if debug:
#                 print("solvePnP failed")
#             return None

#         if debug:
#             print("rvec:", rvec.ravel())
#             print("tvec:", tvec.ravel())

#         T_cam_aruco = self.rvec_tvec_to_SE3(rvec, tvec)
#         return T_cam_aruco

#     def draw(self, image_bgr, detected_corners=None, T_cam_aruco=None):
#         img = image_bgr.copy()

#         # ---- 1. 绘制检测到的 ArUco ----
#         if detected_corners is not None:
#             pts = detected_corners.astype(np.int32)

#             # 画框
#             cv2.polylines(img, [pts], True, (0, 255, 0), 3)

#             # 画角点编号
#             for i, (x, y) in enumerate(pts):
#                 cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
#                 cv2.putText(img, str(i), (x + 10, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7, (0, 0, 255), 2)

#             # ---- 2. 绘制期望角点（蓝色） ----
#             for (x, y) in self.desired_corners.astype(np.int32):
#                 cv2.drawMarker(
#                     img,
#                     (x, y),
#                     (255, 0, 0),
#                     markerType=cv2.MARKER_TILTED_CROSS,
#                     markerSize=20,
#                     thickness=2
#                 )
#             cv2.imwrite(f"data/test_aruco_detect_{time.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png", img)

#         # ---- 3. 绘制 3D 坐标轴（位姿可视化） ----
#         if T_cam_aruco is not None:
#             # 从 SE3 中提取旋转矩阵和平移向量
#             R = T_cam_aruco.R
#             t = T_cam_aruco.t.reshape(3, 1)
#             # 定义 3D 坐标轴（长度=marker_size/2，此处用检测到的标记半边长近似）
#             # marker_half_size = np.linalg.norm(detected_corners[0] - detected_corners[1]) / 2.0
#             marker_half_size = self.marker_size / 2.0
#             axis_3d = np.array([
#                 [0, 0, 0],
#                 [marker_half_size, 0, 0],  # x轴（红）
#                 [0, marker_half_size, 0],  # y轴（绿）
#                 [0, 0, marker_half_size]   # z轴（蓝）
#             ], dtype=np.float32)
#             # 投影到图像平面
#             axis_2d, _ = cv2.projectPoints(axis_3d, cv2.Rodrigues(R)[0], t,
#                                         self.camera_matrix, self.dist_coeffs)
#             axis_2d = axis_2d.astype(np.int32)
#             # 绘制坐标轴
#             cv2.line(img, tuple(axis_2d[0][0]), tuple(axis_2d[1][0]), (0, 0, 255), 3)
#             cv2.line(img, tuple(axis_2d[0][0]), tuple(axis_2d[2][0]), (0, 255, 0), 3)
#             cv2.line(img, tuple(axis_2d[0][0]), tuple(axis_2d[3][0]), (255, 0, 0), 3)

#         cv2.imshow("Aruco Detector", img)
#         cv2.waitKey(1)

#         return img

#     def get_desired_corners(self):
#         return self.desired_corners.copy()


# def print_se3_pose(T: sm.SE3, name : str):
#     """分别打印旋转矩阵和平移向量"""
#     print(f"{name}:")
#     print("  旋转矩阵 R:")
#     print(T.R.round(4))  # 旋转矩阵，保留4位小数
#     print("  平移向量 t:")
#     print(T.t.round(4))  # 平移向量，保留4位小数




# if __name__ == "__main__":
#     # 设置numpy打印参数
#     np.set_printoptions(
#         precision=6,        # 保留4位小数
#         suppress=True,      # 禁用科学计数法
#         linewidth=100,      # 每行宽度
#         floatmode='fixed'   # 固定小数位数
#     )

#     detector = ArucoDetector(720,720,0.1)
#     img = cv2.imread("test_aruco.png")  # 输入 BGR 图像

#     # 测试 detect
#     corners = detector.detect(img)
#     print("Detected corners: ", corners)
#     print("Desired corners: ", detector.get_desired_corners())
#     out = detector.draw(img, corners)


#     # 测试 estimate_pose
#     T_cam_aruco = detector.estimate_pose(img, False)
#     print_se3_pose(T_cam_aruco,'T_cam_aruco')
#     out = detector.draw(img, corners, T_cam_aruco)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

