from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(data="coco128.yaml", epochs=3)


if __name__ == '__main__':
    main()
