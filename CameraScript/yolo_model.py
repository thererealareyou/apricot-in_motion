import os
import cv2
from ultralytics import YOLO, solutions


def run_model(image_folder):
    model = YOLO("yolo11n.pt")

    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            results = model(image_path)

            detected_objects = results[0].boxes
            count = len(detected_objects)

            if count > 0:
                classes = [model.names[int(box.cls)] for box in detected_objects]
                classes = [cls for cls in classes if cls != "airplane"] # Фильтруем классы, исключая "airplane"
                unique_classes = set(classes)
                class_count = {cls: classes.count(cls) for cls in unique_classes}

            print(f"Image: {filename}, Detected objects: {class_count}")


def run_model_video():
    model = YOLO("yolo11n.pt")
    video_path = 'video/2024-12-07 00-46-38.mp4'
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Video', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'video/2024-12-07 00-46-38-Y.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        cv2.imshow('YOLO Video', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def count_objects_in_region(video_path, output_video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Файл повреждён"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(100, 500), (540, 500), (540, 680), (20, 680)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Что-то пошло не так :/")
            break

        results = counter.model(im0)
        detected_objects = results[0].boxes

        filtered_objects = [box for box in detected_objects if counter.model.names[int(box.cls)] != "airplane"]

        im0 = counter.count(im0)
        print(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cameras_link = ['Camera1.mp4', 'Camera2.mp4', 'Camera3.mp4']
    for link in cameras_link:
        run_model("video/sources/" + link, "video/yoloed/" + link, "yolo11n.pt")