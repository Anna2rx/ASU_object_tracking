import argparse

import cv2


CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple object tracking')
    parser.add_argument('--weights', type=str, help='Path to weights file')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--classes', type=str, help='Path to classes file')
    parser.add_argument('--video', type=str, help='Path to video file')

    args = parser.parse_args()

    with open(args.classes, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    capture = cv2.VideoCapture(str(args.video))

    net = cv2.dnn.readNet(args.weights, args.config)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    while True:
        success, frame = capture.read()

        if not success:
            break

        # производим детекцию объектов
        tic = cv2.getTickCount()
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        toc = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (toc - tic)

        for class_id, score, (x1, y1, w, h) in zip(classes, scores, boxes):
            class_id, score = class_id.item(), score.item()

            label = class_names[class_id]
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f'fps = {fps:.0f}', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Отображаем результат детекции
        cv2.imshow('Press "Q" for exit', frame)

        key = cv2.waitKey(42) & 0xFF

        if key == ord('q'):
            break
