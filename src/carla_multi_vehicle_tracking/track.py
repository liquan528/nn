import carla
import random
import cv2
import numpy as np
from ultralytics import YOLO

HOST = '127.0.0.1'
PORT = 2000
W = 640
H = 480
FPS = 10

# 【新增：碰撞预警模块】配置变量
# 碰撞检测阈值：画面宽度的百分比（0.1表示10%）
COLLISION_DISTANCE_THRESHOLD_RATIO = 0.10
# 根据画面宽度计算实际距离阈值
COLLISION_DISTANCE_THRESHOLD = int(W * COLLISION_DISTANCE_THRESHOLD_RATIO)
# 预警框和文字的颜色（BGR格式）
COLLISION_WARNING_COLOR = (0, 0, 255)  # 红色
COLLISION_WARNING_TEXT_COLOR = (0, 0, 255)  # 红色
# 预警框的线条粗细
COLLISION_WARNING_BOX_THICKNESS = 3
# 预警文字大小
COLLISION_WARNING_TEXT_SCALE = 0.8
# 预警文字粗细
COLLISION_WARNING_TEXT_THICKNESS = 2
# 预警文字位置
COLLISION_WARNING_TEXT_POSITION = (10, 30)
# 预警消息
COLLISION_WARNING_MESSAGE = "COLLISION WARNING!"

model = YOLO('yolov8n.pt')
print("Model loaded")

# 【新增：碰撞预警模块】功能函数
def check_vehicle_collision(bbox_data, frame_width):
    """
    【新增：碰撞预警模块】检测车辆之间的碰撞风险
    
    参数:
        bbox_data: 检测到的车辆边界框数据列表
                  每个元素格式：[x1, y1, x2, y2, track_id, class_id, confidence]
                  - x1, y1: 边界框左上角坐标
                  - x2, y2: 边界框右下角坐标
                  - track_id: 跟踪ID（如果使用DeepSORT）
                  - class_id: 类别ID（车辆类别）
                  - confidence: 置信度
        frame_width: 视频帧的宽度，用于计算相对距离
    
    返回:
        collision_pairs: 有碰撞风险的车辆对列表
                        每个元素格式：[(x1, y1, x2, y2, id1), (x1, y1, x2, y2, id2)]
                        包含两辆车的边界框和ID
    """
    collision_pairs = []
    
    # 筛选出车辆类别的检测结果（class_id 2 or 3 or 7 对应 car, motorcycle, bus等常见车辆）
    vehicle_class_ids = [2, 3, 7]  # COCO数据集中车辆相关类别
    
    # 提取所有车辆的中心点和边界框信息
    vehicles = []
    for det in bbox_data:
        if len(det) >= 6:
            class_id = int(det[5]) if len(det) > 5 else -1
            # 检查是否为车辆类别
            if class_id in vehicle_class_ids:
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                # 计算中心点坐标
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 获取跟踪ID（如果有的话）
                track_id = int(det[4]) if len(det) > 4 else -1
                
                vehicles.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'track_id': track_id,
                    'class_id': class_id
                })
    
    # 如果车辆数量少于2辆，不可能发生碰撞
    if len(vehicles) < 2:
        return collision_pairs
    
    # 计算所有车辆对之间的距离
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            vehicle1 = vehicles[i]
            vehicle2 = vehicles[j]
            
            # 计算两车中心点之间的欧几里得距离
            dx = vehicle1['center'][0] - vehicle2['center'][0]
            dy = vehicle1['center'][1] - vehicle2['center'][1]
            distance = np.sqrt(dx * dx + dy * dy)
            
            # 如果距离小于阈值，判定为有碰撞风险
            if distance < COLLISION_DISTANCE_THRESHOLD:
                # 获取两辆车的ID
                id1 = vehicle1['track_id'] if vehicle1['track_id'] != -1 else i
                id2 = vehicle2['track_id'] if vehicle2['track_id'] != -1 else j
                
                # 添加到碰撞对列表
                collision_pairs.append((vehicle1['bbox'], vehicle2['bbox'], id1, id2))
                
                # 在控制台打印警告日志
                print(f"[警告] 车辆ID:{id1} 和 车辆ID:{id2} 距离过近，存在碰撞风险")
    
    return collision_pairs

# 【新增：碰撞预警模块】绘制预警效果
def draw_collision_warning(frame, collision_pairs):
    """
    【新增：碰撞预警模块】在视频帧上绘制碰撞预警效果
    
    参数:
        frame: 视频帧图像（numpy数组）
        collision_pairs: 碰撞对列表，每个元素为 (bbox1, bbox2, id1, id2)
    
    返回:
        frame: 绘制了预警效果的视频帧
    """
    # 如果有碰撞风险，显示预警文字
    if len(collision_pairs) > 0:
        # 在帧顶部绘制红色警告文字
        cv2.putText(
            frame,
            COLLISION_WARNING_MESSAGE,
            COLLISION_WARNING_TEXT_POSITION,
            cv2.FONT_HERSHEY_SIMPLEX,
            COLLISION_WARNING_TEXT_SCALE,
            COLLISION_WARNING_TEXT_COLOR,
            COLLISION_WARNING_TEXT_THICKNESS,
            cv2.LINE_AA
        )
        
        # 为每对碰撞风险的车辆绘制红色边框
        for bbox1, bbox2, id1, id2 in collision_pairs:
            # 绘制第一辆车的边界框
            x1, y1, x2, y2 = map(int, bbox1)
            cv2.rectangle(
                frame, 
                (x1, y1), 
                (x2, y2), 
                COLLISION_WARNING_COLOR, 
                COLLISION_WARNING_BOX_THICKNESS
            )
            
            # 绘制第二辆车的边界框
            x1, y1, x2, y2 = map(int, bbox2)
            cv2.rectangle(
                frame, 
                (x1, y1), 
                (x2, y2), 
                COLLISION_WARNING_COLOR, 
                COLLISION_WARNING_BOX_THICKNESS
            )
    
    return frame

video_out = None
output_file = 'output.mp4'

def handle_image(image):
    global video_out
    data = np.frombuffer(image.raw_data, dtype=np.uint8)
    data = data.reshape((H, W, 4))
    img = data[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    res = model(img, imgsz=640, conf=0.25)
    show = res[0].plot()
    
    # 【新增：碰撞预警模块】提取检测结果并检查碰撞风险
    # 获取检测到的边界框数据
    if res[0].boxes is not None and len(res[0].boxes) > 0:
        # 将检测结果转换为列表格式
        bbox_data = res[0].boxes.xyxy.cpu().numpy()
        # 获取类别信息
        class_data = res[0].boxes.cls.cpu().numpy()
        # 获取置信度
        conf_data = res[0].boxes.conf.cpu().numpy()
        
        # 组合所有检测数据 [x1, y1, x2, y2, class_id, confidence]
        detections = []
        for i in range(len(bbox_data)):
            x1, y1, x2, y2 = bbox_data[i]
            class_id = class_data[i]
            conf = conf_data[i]
            detections.append([x1, y1, x2, y2, -1, class_id, conf])  # -1表示没有跟踪ID
        
        # 【新增：碰撞预警模块】检查车辆碰撞风险
        collision_pairs = check_vehicle_collision(detections, W)
        
        # 【新增：碰撞预警模块】绘制预警效果
        show = draw_collision_warning(show, collision_pairs)
    
    if video_out is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(output_file, fourcc, FPS, (W, H))
    video_out.write(show)

    cv2.imshow('CARLA Tracking', show)
    cv2.waitKey(1)

def main():
    global video_out
    client = None
    actors = []

    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        print("Connected to CARLA")

        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(W))
        cam_bp.set_attribute('image_size_y', str(H))
        cam_bp.set_attribute('fov', '100')

        spawn_point = carla.Transform(
            carla.Location(x=-50, y=15, z=3),
            carla.Rotation(pitch=-20)
        )
        camera = world.spawn_actor(cam_bp, spawn_point)
        camera.listen(handle_image)
        actors.append(camera)
        print("Camera started")

        spawn_points = world.get_map().get_spawn_points()
        vehicle_bps = bp_lib.filter('vehicle.*')

        for i in range(4):
            if i < len(spawn_points):
                bp = random.choice(vehicle_bps)
                car = world.try_spawn_actor(bp, spawn_points[i])
                if car:
                    car.set_autopilot(True)
                    actors.append(car)

        print("Tracking started! Press q to quit")

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        if video_out:
            video_out.release()
            print(f"Video saved to {output_file}")

        for a in actors:
            if a:
                a.destroy()
        print("Cleanup done")

if __name__ == '__main__':
    main()