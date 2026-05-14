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

model = YOLO('yolov8n.pt')
print("Model loaded")

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