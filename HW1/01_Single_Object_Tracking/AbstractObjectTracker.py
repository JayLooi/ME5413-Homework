import os
import cv2
import numpy as np
from utils import get_bbox_points, draw_bbox


class AbstractObjectTraker:
    def __init__(self, template=None):
        self.set_template(template)
        self._frame_seq = None
        self._seq_track_result = []

    def set_template(self, template):
        self._template = template

    def track(self, frame):
        raise NotImplementedError('track() method is not implemented by the derived class')

    def track_seq(self, frame_seq):
        self._frame_seq = frame_seq
        self._seq_track_result.clear()
        for img_path in frame_seq:
            frame = cv2.imread(img_path)
            bbox, _ = self.track(frame)
            self._seq_track_result.append(bbox)

    def save_tracking_video(self, directory, vidname, groundtruth_fp=None):
        if self._frame_seq is None:
            print('[ERROR] No sequence of frames given')
            return

        if self._seq_track_result is None:
            print('[ERROR] No tracking result available')
            return

        if len(self._frame_seq) != len(self._seq_track_result):
            print('[ERROR] Number tracking results and the number of frames are not same')
            return

        if groundtruth_fp is not None:
            groundtruth = self._read_bbox(groundtruth_fp)
            with_gt = True
            gt_bbox_colour = (237, 161, 9)

        else:
            with_gt = False

        video_path = os.path.join(directory, f'{vidname}.avi')
        codec = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
        h, w = cv2.imread(self._frame_seq[0], cv2.IMREAD_GRAYSCALE).shape
        writer = cv2.VideoWriter(video_path, codec, 30.0, (w, h))

        tracked_bbox_colour = (0, 255, 0)
        for i, img_path in enumerate(self._frame_seq):
            frame = cv2.imread(img_path)
            top_left, bottom_right = get_bbox_points(self._seq_track_result[i])
            frame = draw_bbox(frame, top_left, bottom_right, tracked_bbox_colour, 2, 'Tracked')

            if with_gt:
                top_left, bottom_right = get_bbox_points(groundtruth[i])
                frame = draw_bbox(frame, top_left, bottom_right, gt_bbox_colour, 2, 'Ground truth')

            writer.write(frame)

        writer.release()

    def save_track_bbox(self, filepath):
        np.savetxt(filepath, self._seq_track_result, fmt='%d', delimiter=',')

    def evaluate(self, groundtruth_fp, metric, threshold=None):
        if self._seq_track_result is None:
            print('[ERROR] No tracking result available')
            return

        metric = metric.lower()
        gt = self._read_bbox(groundtruth_fp)
        if metric == 'precision':
            result = self._calculate_precision(gt, threshold)

        elif metric == 'success':
            result = self._calculate_success(gt, threshold)

        else:
            pass

        return result

    def _calculate_precision(self, groundtruth, threshold=None):
        compute_centre = lambda bbox: (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        gt = np.array(list(map(compute_centre, groundtruth)))
        tracked = np.array(list(map(compute_centre, self._seq_track_result)))
        precision = np.linalg.norm(tracked - gt, axis=1)
        if threshold is not None:
            precision = np.array(precision <= threshold, dtype=np.float32)

        return precision

    def _calculate_success(self, groundtruth, threshold=None):
        results = []
        for gt, tracked in zip(groundtruth, self._seq_track_result):
            intersection = self._find_intersection(gt, tracked)
            if intersection is not None:
                _, _, w, h = intersection
                intersect_area = w * h
                union = gt[2] * gt[3] + tracked[2] * tracked[3] - intersect_area
                results.append(intersect_area / union)

            else:
                results.append(0)

        results = np.array(results)

        if threshold is not None:
            results = np.array(results >= threshold, dtype=np.float32)

        return results

    @staticmethod
    def _read_bbox(bbox_fp):
        bbox = np.loadtxt(bbox_fp, dtype=np.uint32, delimiter=',')
        return bbox

    @staticmethod
    def _find_intersection(bbox_1, bbox_2):
        x1, y1, w1, h1 = bbox_1
        x2, y2, w2, h2 = bbox_2
        if x1 < x2:
            if (x1 + w1) < x2:
                return
            else:
                x = x2

        else:
            if (x2 + w2) < x1:
                return
            else:
                x = x1

        if y1 < y2:
            if (y1 + h1) < y2:
                return
            else:
                y = y2

        else:
            if (y2 + h2) < y1:
                return
            else:
                y = y1

        w = min(x1 + w1, x2 + w2) - x + 1
        h = min(y1 + h1, y2 + h2) - y + 1

        return x, y, w, h
