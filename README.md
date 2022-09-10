# Simple object tracking

Полезные ссылки:
- [Object Tracking with OpenCV](https://livecodestream.dev/post/object-tracking-with-opencv)
- [OpenCV Object Tracking](https://pyimagesearch.com/2018/07/30/opencv-object-tracking/)
- [Free HD Stock Footage & 4K Videos!](https://www.videezy.com/)

### Настройка окружения
```bash
git clone git@github.com:Anna2rx/ASU_object_tracking.git
cd ASU_object_tracking
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Запуск трекинга объекта
```bash
cd src
python3 simple_object_tracking.py --video ../assets/videos/input.mp4 --tracker-type csrt
```

### Запуск детекции объектов
```bash
cd src
python3 simple_yolov4_object_detection.py \
  --weights ../assets/object-detection-model/yolov4-tiny.weights \
  --config ../assets/object-detection-model/yolov4-tiny.cfg \
  --classes ../assets/object-detection-model/classes.txt \
  --video ../assets/videos/input.mp4
```
