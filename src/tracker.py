from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class ObjectMiddlePointTracker:
    def __init__(self, threshold: float):
        # если расстояние между центрами детекций на соседних кадрах:
        # - меньше порога, то объекты считаются одним и тем же
        # - больше порога, то объекты считаются разными
        self._threshold = threshold
        self._next_track_id: int = 0
        self._tracks: Dict[int, Tuple[int, int]] = {}

    def track(self, boxes: List[Tuple[int, int, int, int]]) -> List[int]:
        tracking_results: List[int] = []

        # если нет отслеживаемых объектов
        # если на предыдущем кадре не было найдено объектов
        if len(self._tracks.keys()) == 0:
            for x, y, w, h in boxes:
                # регистрируем новый объект
                self._tracks[self._next_track_id] = (x + w // 2, y + h // 2)
                tracking_results.append(self._next_track_id)
                self._next_track_id += 1

            return tracking_results

        # если не найдено ни одного объекта
        # если на текущем кадре отсутствуют объекты
        if len(boxes) == 0:
            # удаляем все отслеживаемые объекты
            self._tracks.clear()
            return tracking_results

        pairwise_euclidean_distance = euclidean_distances(
            X=np.array([[x + w // 2, y + h // 2] for x, y, w, h in boxes]),
            Y=np.array([[x_mid, y_mid] for x_mid, y_mid in self._tracks.values()]),
        )
        # pairwise_euclidean_distance - матрица попарных расстояний между найденными и отслеживаемыми объектами
        # строчки pairwise_euclidean_distance - это найденные объекты на текущем кадре
        # столбцы pairwise_euclidean_distance - это отслеживаемые объекты с предыдущего кадра

        for i, (x, y, w, h) in enumerate(boxes):
            # если расстояние до ближайшего объекта больше порога
            if pairwise_euclidean_distance[i].min() > self._threshold:
                self._tracks[self._next_track_id] = (x + w // 2, y + h // 2)
                tracking_results.append(self._next_track_id)
                self._next_track_id += 1
            else:
                tracking_results.append(
                    list(self._tracks.keys())[np.argmin(pairwise_euclidean_distance[i])]
                )
        # для всех объектов, которые существовали на предыдущем кадре, но пропали на текущем
        for track_id in set(self._tracks.keys()) - set(tracking_results):
            del self._tracks[track_id]

        return tracking_results
