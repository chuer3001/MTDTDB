# A Motion Target Detection and Tracking Algorithm in Dynamic Background Based on Improved Optical Flow Method
import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment





# 读取原视频参数
def video_processor(input_path):
    # 初始化视频捕获对象
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {input_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS) # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 宽
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 高
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 总帧数


    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束或读取失败时退出
        frames.append(frame)
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    return frames,fps,width,height,total_frames

# ORB检测器
def non_max_suppression(keypoints, radius=5.0):
    """非极大值抑制，保留响应值高的关键点"""
    if not keypoints:
        return []
    # 按响应值降序排序
    keypoints_sorted = sorted(keypoints, key=lambda x: -x.response)
    suppressed = []
    for kp in keypoints_sorted:
        # 检查是否与已保留的关键点距离过近
        keep = True
        for kept_kp in suppressed:
            dx = kp.pt[0] - kept_kp.pt[0]
            dy = kp.pt[1] - kept_kp.pt[1]
            distance = (dx**2 + dy**2)**0.5
            if distance < radius:
                keep = False
                break
        if keep:
            suppressed.append(kp)
    return suppressed
def ORB(frame,nfeature):
    orb = cv2.ORB_create(
        nfeatures=nfeature,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE
    )

    # 检测关键点（不计算描述符）
    keypoints = orb.detect(frame, None)

    # 应用非极大值抑制
    keypoints = non_max_suppression(keypoints, radius=15.0)#   15

    # 计算抑制后的关键点描述符
    keypoints, descriptors = orb.compute(frame, keypoints)

    # 提取关键点坐标矩阵 (N, 2) 格式
    if keypoints:
        # 将 KeyPoint 对象列表转换为 (x, y) 坐标矩阵
        keypoints_matrix = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    else:
        keypoints_matrix = np.empty((0, 2), dtype=np.float32)  # 空矩阵

    return np.array(keypoints_matrix)  # 返回坐标矩阵 (N, 2)


# lk金字塔
def build_pyramid(image, max_level):
    pyramid = [image]
    for _ in range(max_level - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid
def lucas_kanade(prev_img, next_img, points, window_size=15):
    """
    单层Lucas-Kanade光流计算
    :param prev_img: 前一帧图像
    :param next_img: 后一帧图像
    :param points: 待跟踪的特征点坐标 (Nx2数组)
    :param window_size: 窗口大小
    :return: 新的特征点坐标 (Nx2数组), 状态标志 (1表示成功, 0表示失败)
    """
    new_points = []
    status = []

    # 转换为浮点型以支持梯度计算
    prev_img = prev_img.astype(np.float32)
    next_img = next_img.astype(np.float32)

    # 计算空间梯度 (Ix, Iy) 和时间梯度 (It)
    Ix = cv2.Sobel(prev_img, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_img, cv2.CV_32F, 0, 1, ksize=3)
    It = next_img - prev_img

    half_win = window_size // 2

    for (x, y) in points:
        x, y = int(x), int(y)
        # 检查是否超出图像边界
        if x < half_win or x >= prev_img.shape[1] - half_win or \
                y < half_win or y >= prev_img.shape[0] - half_win:
            new_points.append((x, y))
            status.append(0)
            continue

        # 提取窗口内的梯度
        win_Ix = Ix[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1].flatten()
        win_Iy = Iy[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1].flatten()
        win_It = -It[y - half_win:y + half_win + 1, x - half_win:x + half_win + 1].flatten()

        # 构建方程 A^T A d = A^T b
        A = np.vstack((win_Ix, win_Iy)).T
        b = win_It

        # 计算光流向量 (dx, dy)
        try:
            ATA = A.T @ A
            eigenvalues = np.linalg.eigvals(ATA)
            if np.min(eigenvalues) < 1e-6:  # 检查矩阵是否可逆
                raise np.linalg.LinAlgError
            dx_dy = np.linalg.inv(ATA) @ (A.T @ b)
            dx, dy = dx_dy
        except np.linalg.LinAlgError:
            dx, dy = 0, 0
            status.append(0)
        else:
            status.append(1)

        new_x = x + dx
        new_y = y + dy
        new_points.append((new_x, new_y))

    return np.array(new_points), np.array(status)
def pyramid_lk(prev_img, next_img, points, max_level=5, window_size=15):
    # 构建金字塔
    pyramid_prev = build_pyramid(prev_img, max_level)
    pyramid_next = build_pyramid(next_img, max_level)

    # 从顶层（低分辨率）开始初始化光流
    current_points = points / (2 ** (max_level - 1))  # 坐标缩放

    for level in range(max_level - 1, -1, -1):
        # 获取当前层的图像
        prev = pyramid_prev[level]
        next = pyramid_next[level]

        # 计算当前层的光流
        current_points, status = lucas_kanade(prev, next, current_points, window_size)

        # 向下一层（更高分辨率）传递坐标，并进行双线性插值
        if level > 0:
            current_points = current_points * 2

    return current_points, status


# 仿射矩阵变换
def CHANGE(good_prev, good_curr, prev_frame):
    # 计算仿射变换矩阵
    if len(good_prev) >= 3 and len(good_curr) >= 3:
        # 使用RANSAC算法估计仿射矩阵
        affine_matrix, inliers = cv2.estimateAffine2D(
            good_prev, good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000
        )

        if affine_matrix is not None:
            print("仿射变换矩阵:")
            print(affine_matrix)

            # 分解平移、旋转、缩放参数
            tx = affine_matrix[0, 2]
            ty = affine_matrix[1, 2]
            rotation = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0]) * 180 / np.pi
            scale = np.sqrt(affine_matrix[0, 0] ** 2 + affine_matrix[1, 0] ** 2)

            print(f"\n分解参数:\n平移: ({tx:.2f}, {ty:.2f})\n"
                  f"旋转: {rotation:.2f}°\n缩放因子: {scale:.2f}\n")

        else:
            print("仿射矩阵估计失败！")
    else:
        print("有效跟踪点不足（至少需要3对点）")

    return tx, ty, rotation, scale, affine_matrix

# 卡尔曼滤波器(去噪）
class MotionKalmanFilter:
    def __init__(self, freq=30, process_noise=1e-4, measure_noise=1e-2):
        """
        初始化卡尔曼滤波器
        :param freq: 采样频率(Hz)，用于计算速度积分
        :param process_noise: 过程噪声系数 (0.0001~0.1)
        :param measure_noise: 测量噪声系数 (0.01~1.0)
        """
        self.kf = cv2.KalmanFilter(8, 4)  # 状态8维(tx,ty,θ,s,tx_v,ty_v,θ_v,s_v)，观测4维

        # ===== 状态转移矩阵 =====
        # 假设匀速运动模型：x(t+1) = x(t) + v(t)*dt
        dt = 1.0 / freq
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],  # tx
            [0, 1, 0, 0, 0, dt, 0, 0],  # ty
            [0, 0, 1, 0, 0, 0, dt, 0],  # θ (角度)
            [0, 0, 0, 1, 0, 0, 0, dt],  # s (缩放)
            [0, 0, 0, 0, 1, 0, 0, 0],  # tx_v
            [0, 0, 0, 0, 0, 1, 0, 0],  # ty_v
            [0, 0, 0, 0, 0, 0, 1, 0],  # θ_v
            [0, 0, 0, 0, 0, 0, 0, 1]  # s_v
        ], dtype=np.float32)

        # ===== 观测矩阵 =====
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # 观测tx
            [0, 1, 0, 0, 0, 0, 0, 0],  # 观测ty
            [0, 0, 1, 0, 0, 0, 0, 0],  # 观测θ
            [0, 0, 0, 1, 0, 0, 0, 0]  # 观测s
        ], dtype=np.float32)

        # ===== 噪声协方差矩阵 =====
        self.kf.processNoiseCov = process_noise * np.eye(8, dtype=np.float32)
        self.kf.measurementNoiseCov = measure_noise * np.eye(4, dtype=np.float32)

        # 初始化状态和误差协方差
        self.kf.errorCovPost = 0.1 * np.ones((8, 8), dtype=np.float32)
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)

        # 角度处理参数（避免360°跳变）
        self.last_angle = 0.0

    def _angle_warp(self, angle):
        """处理角度周期性（-180°~180°）"""
        delta = angle - self.last_angle
        if delta > 180:
            angle -= 360
        elif delta < -180:
            angle += 360
        self.last_angle = angle
        return angle

    def update(self, tx, ty, theta, scale):
        """输入新观测值并返回滤波后的参数"""
        # 预处理角度
        theta = self._angle_warp(theta)

        # 预测步骤
        prediction = self.kf.predict()

        # 构建测量向量（注意角度转换为弧度）
        measurement = np.array([[tx], [ty],
                                [np.radians(theta)],  # 转弧度
                                [scale]], dtype=np.float32)

        # 更新步骤
        self.kf.correct(measurement)

        # 获取滤波后状态
        state = self.kf.statePost.ravel()

        # 后处理角度（转回角度制）
        filtered_theta = np.degrees(state[2]) % 360
        return state[0], state[1], filtered_theta, state[3]

# 背景补偿
def background_compensation(frame, affine_matrix, prev_frame_shape):
    """
    使用仿射变换进行背景补偿
    参数:
        frame: 当前帧图像（BGR格式）
        affine_matrix: 经过卡尔曼滤波的2x3仿射变换矩阵
        prev_frame_shape: 参考帧的尺寸（用于调整输出大小）
    """
    # 使用逆向变换矩阵补偿当前帧（若需要对齐到参考坐标系）
    # 如果affine_matrix是将参考帧变换到当前帧的矩阵，则需要取其逆矩阵
    # compensation_matrix = cv2.invertAffineTransform(affine_matrix)

    # 直接应用滤波后的仿射矩阵（假设已对齐参考坐标系）
    compensation_matrix = affine_matrix

    # 执行仿射变换（输出尺寸与参考帧一致）
    rows, cols = prev_frame_shape[:2]
    compensated_frame = cv2.warpAffine(
        frame,
        compensation_matrix,
        (cols, rows),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE
    )

    return compensated_frame





# 卡尔曼滤波器（目标追踪）
class KalmanTracker:
    def __init__(self, bbox):
        # 初始化卡尔曼滤波器（状态：x, y, w, h, dx, dy）
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]], dtype=np.float32)

        # 初始化噪声协方差（根据实际场景调整）
        self.kf.processNoiseCov = 1e-4 * np.eye(6, dtype=np.float32)
        self.kf.measurementNoiseCov = 1e-2 * np.eye(4, dtype=np.float32)

        # 初始状态
        self.kf.statePost = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0], dtype=np.float32)
        self.time_since_update = 0
        self.id = KalmanTracker.count
        self.hits = 1  # 连续匹配次数
        KalmanTracker.count += 1

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.statePost[:4]

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.correct(np.array(bbox, dtype=np.float32))
KalmanTracker.count = 0

# 置信比检测
def iou(bbox1, bbox2):
    # 计算交并比（支持多种格式输入）
    if isinstance(bbox1, np.ndarray):
        x1, y1, w1, h1 = bbox1.reshape(-1)
    else:
        x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1 + w1, x2 + w2) - xi
    hi = min(y1 + h1, y2 + h2) - yi

    if wi > 0 and hi > 0:
        intersection = wi * hi
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union
    return 0
# 多目标追踪（匈牙利）
class MultiObjectTracker:
    def __init__(self, max_age=5, iou_threshold=0.3):
        self.trackers = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update(self, detections):
        # 阶段1：预测所有追踪器的当前位置
        for trk in self.trackers:
            trk.predict()

        # 阶段2：数据关联（匈牙利算法）
        if len(detections) > 0 and len(self.trackers) > 0:
            # 构建代价矩阵（1 - IOU）
            cost_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
            for t, trk in enumerate(self.trackers):
                for d, det in enumerate(detections):
                    cost_matrix[t, d] = 1 - iou(trk.predict(), det)

            # 执行匈牙利算法匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 处理匹配结果
            matched = set()
            for t, d in zip(row_ind, col_ind):
                if cost_matrix[t, d] < 1 - self.iou_threshold:
                    self.trackers[t].update(detections[d])
                    matched.add(d)

            # 处理未匹配的检测（新建追踪器）
            for d in range(len(detections)):
                if d not in matched:
                    self.trackers.append(KalmanTracker(detections[d]))
        else:
            # 没有现有追踪器时，全部新建
            for d in detections:
                self.trackers.append(KalmanTracker(d))

        # 阶段3：清理失效的追踪器
        self.trackers = [trk for trk in self.trackers
                         if trk.time_since_update <= self.max_age]

        # 返回有效追踪结果（仅返回最近更新的）
        return [{
            'id': trk.id,
            'bbox': trk.kf.statePost[:4].astype(int),
            'hits': trk.hits
        } for trk in self.trackers if trk.time_since_update == 0]



# 可视化跟踪目标
def visualize_tracks(frame, tracks):


    for trk in tracks:
        x, y, w, h = trk['bbox']
        track_id = trk['id']

        # 绘制边界框
        cv2.rectangle(frame,
                      (x, y),
                      (x + w, y + h),
                      (100, 200, 100), 2)

        # 绘制ID文本背景
        cv2.rectangle(frame,
                      (x, y - 30),
                      (x + 190, y+10),
                      (100, 200, 100), -1)


        cv2.putText(frame,
                    f"RUNNING-OBJECT",
                    (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),  # 黑色文字
                    2)


    return frame

# 拉普拉斯金字塔
def build_laplacian_pyramid(img, levels=3):
    """构建拉普拉斯金字塔"""
    pyramid = []
    current = img.copy()
    for _ in range(levels):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        pyramid.append(current - up)
        current = down
    return pyramid
def estimate_affine_params(points1, points2):
    """使用RANSAC估计仿射矩阵"""
    if len(points1) < 3:
        return None
    matrix, inliers = cv2.estimateAffine2D(
        points1.astype(np.float32),
        points2.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99
    )
    return matrix
def multi_scale_affine_estimation(img1, img2, points1, points2, levels=3):
    """多尺度仿射参数估计（适配光流法特征点输入）"""
    # 构建金字塔
    pyramid1 = build_laplacian_pyramid(img1, levels)
    pyramid2 = build_laplacian_pyramid(img2, levels)

    # 初始化齐次坐标矩阵 (3x3)
    affine_matrix = np.eye(3)  # 修改为3x3单位矩阵

    # 从粗到细处理金字塔
    for level in reversed(range(levels)):
        # 计算当前层缩放因子
        scale = 2 ** level

        # 将光流点坐标缩放到当前金字塔层
        scaled_pts1 = points1 / scale
        scaled_pts2 = points2 / scale

        # 估计当前层仿射参数（直接使用坐标数组）
        if level == levels - 1:  # 最粗层初始化
            current_matrix = estimate_affine_params(scaled_pts1.reshape(-1, 2), scaled_pts2.reshape(-1, 2))
            if current_matrix is None:
                current_matrix = np.eye(2, 3)
        else:  # 精细层优化
            # 应用累积的仿射变换
            transformed_pts = cv2.transform(scaled_pts1.reshape(-1, 1, 2), affine_matrix[:2]).squeeze()
            current_matrix = estimate_affine_params(transformed_pts, scaled_pts2.reshape(-1, 2))
            if current_matrix is None:
                current_matrix = np.eye(2, 3)

        # 将当前层矩阵转换为齐次形式
        current_matrix_hom = np.vstack([current_matrix, [0, 0, 1]])  # 扩展为3x3

        # 矩阵组合（使用齐次坐标乘法）
        affine_matrix = current_matrix_hom @ affine_matrix

    # 返回OpenCV标准的2x3仿射矩阵
    return affine_matrix[:2]



if __name__ == '__main__':
    # 初始化读取及输出视频地址
    input_path=r"xxx.mp4"
    output_path=r"xxx.mp4"

    # 提取目标视频参数
    frames, fps, width, height, total_frames = video_processor(input_path)
    # 视频文件输入初始化
    camera = cv2.VideoCapture(input_path)
    # 视频写入器初始化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器设置
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )
    # 追踪器初始化
    tracker = MultiObjectTracker(max_age=5, iou_threshold=0.8)
    # 初始化当前帧的前两帧
    lastFrame1 = None
    lastFrame2 = None

    a=0
    # 遍历视频的每一帧
    while camera.isOpened():
        # 读取下一帧
        (ret, frame) = camera.read()

        # 如果不能抓取到一帧，说明到了视频的结尾
        if not ret:
            break
        # 如果第一二帧是None，对其进行初始化
        if lastFrame2 is None:
            if lastFrame1 is None:
                lastFrame1 = frame
            else:
                lastFrame2 = frame
            continue

        lastFrame1_gray = cv2.cvtColor(lastFrame1, cv2.COLOR_BGR2GRAY)
        lastFrame2_gray = cv2.cvtColor(lastFrame2, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # orb角点检测
        prev_pts1 = ORB(lastFrame1_gray, 200)
        prev_pts2 = ORB(lastFrame2_gray, 200)
        #   利用lk金字塔对角点进行跟踪
        curr_pts1, status1 = pyramid_lk(lastFrame1_gray, lastFrame2_gray, prev_pts1)
        curr_pts2, status2 = pyramid_lk(lastFrame2_gray, frame_gray, prev_pts2)
        # 筛选有效跟踪点
        good_p1_2 = prev_pts1[status1 == 1]
        good_p2_1 = curr_pts1[status1 == 1]
        good_p2_f = prev_pts2[status2 == 1]
        good_pf_2 = curr_pts2[status2 == 1]

        # 计算仿射矩阵
        affine_params1 = multi_scale_affine_estimation(lastFrame1_gray, lastFrame2_gray, good_p1_2, good_p2_1,5)
        affine_params2 = multi_scale_affine_estimation(lastFrame2_gray, frame_gray,  good_p2_f, good_pf_2,5)
        # 背景补偿
        e_last2 = background_compensation(
            frame=lastFrame2_gray,
            affine_matrix=affine_params1,
            prev_frame_shape=lastFrame1_gray.shape
        )
        e_last1 = background_compensation(
            frame=lastFrame1_gray,
            affine_matrix=affine_params1,
            prev_frame_shape=lastFrame1_gray.shape
        )
        e_frame = background_compensation(
            frame=frame_gray,
            affine_matrix=affine_params2,
            prev_frame_shape=lastFrame2_gray.shape
        )


        # 三帧差
        # 计算当前帧和前帧的不同,计算三帧差分
        frameDelta1 = cv2.absdiff(e_last1, e_last2)
        frameDelta2 = cv2.absdiff(e_last2, e_frame)  # 帧差二
        thresh = cv2.bitwise_and(frameDelta1, frameDelta2)  # 图像与运算
        thresh2 = thresh.copy()

        # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧
        lastFrame1 = lastFrame2
        lastFrame2 = frame.copy()

        # 图像二值化
        thresh = cv2.threshold(thresh, 40, 255, cv2.THRESH_BINARY)[1]


        # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
        thresh = cv2.dilate(thresh, None, iterations=11)
        thresh = cv2.erode(thresh, None, iterations=3)


        # 阀值图像上的轮廓位置
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历轮廓
        bbox = []
        for c in cnts:
            # 忽略小轮廓，排除误差
            if cv2.contourArea(c) < 9000:
                continue

            # 计算轮廓的边界框，在当前帧中画出该框
            (x, y, w, h) = cv2.boundingRect(c)
            bbox.append([x, y, w, h])

        # 进行目标追踪
        tracks = tracker.update(bbox)
        frame = visualize_tracks(frame, tracks)
        output_path1 = os.path.join(r"xxx", f"frame_{a:06d}.jpg")
        cv2.imwrite(output_path1, frame)
        video_writer.write(frame)
        a+=1
    # 释放资源
    video_writer.release()
    camera.release()
    cv2.destroyAllWindows()
