import argparse

import cv2


tracker_types = ['boosting', 'mil', 'kcf', 'tld', 'medianflow', 'mosse', 'csrt']

TrackersFactory = {
    'boosting': cv2.legacy.TrackerBoosting_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'kcf': cv2.legacy.TrackerKCF_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'medialflow': cv2.legacy.TrackerMedianFlow_create,
    'mosse': cv2.legacy.TrackerMOSSE_create,
    'csrt': cv2.legacy.TrackerCSRT_create,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple object tracking')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--tracker-type', type=str, choices=tracker_types, help='Type of tracker to use')

    args = parser.parse_args()

    tracker = TrackersFactory[args.tracker_type]()

    # Чтение видео
    capture = cv2.VideoCapture(args.video)

    # Если видео не открылось, выходим
    if not capture.isOpened():
        raise Exception('Could not open video')

    # Читаем первый кадр
    success, frame = capture.read()
    if not success:
        raise Exception('Could not read video file')

    bbox = cv2.selectROI(frame, True)

    # Инициализируем трекер
    ok = tracker.init(frame, bbox)

    while True:
        # Читаем кадр
        success, frame = capture.read()

        if not success:
            break

        # Отметка времени до начала обработки кадра
        tic = cv2.getTickCount()
        # Обновляем позицию объекта с помощью трекера
        ok, bbox = tracker.update(frame)
        # Отметка времени после обработки кадра
        toc = cv2.getTickCount()

        fps = cv2.getTickFrequency() / (toc - tic)

        # Если объект найден
        if ok:
            # Рисуем прямоугольник вокруг объекта
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Трекер не нашел объект
            cv2.putText(frame, 'Object not found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Отображем тип трекера
        cv2.putText(frame, f'Tracker: {args.tracker_type}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, f'FPS : {fps:.0f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Отображаем результат трекинга
        cv2.imshow('Press "Q" for exit', frame)

        key = cv2.waitKey(42) & 0xFF

        if key == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
