import os
import cv2
import numpy as np


class AbstractObjectTraker:
    def __init__(self):
        self._frame_seq = None
        self._seq_track_result = []

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
            bbox_pts = self._get_bbox_points(self._seq_track_result[i])
            if bbox_pts is not None:
                top_left, bottom_right = bbox_pts
                frame = self._draw_bbox(frame, top_left, bottom_right, tracked_bbox_colour, 2, 'Tracked')

            if with_gt:
                top_left, bottom_right = self._get_bbox_points(groundtruth[i])
                frame = self._draw_bbox(frame, top_left, bottom_right, gt_bbox_colour, 2, 'Ground truth')

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
            iou = self._iou(gt, tracked)
            results.append(iou)

        results = np.array(results)

        if threshold is not None:
            results = np.array(results >= threshold, dtype=np.float32)

        return results

    @staticmethod
    def _read_bbox(bbox_fp):
        bbox = np.loadtxt(bbox_fp, dtype=np.uint32, delimiter=',')
        return bbox

    @staticmethod
    def _get_bbox_points(bbox):
        top_left = bbox[0:2]
        bbox_w, bbox_h = bbox[2:4]
        if (bbox_w == 0) or (bbox_h == 0):
            return

        bottom_right = (top_left[0] + bbox_w - 1, top_left[1] + bbox_h - 1)
        return top_left, bottom_right

    @staticmethod
    def _draw_bbox(img, top_left, bottom_right, colour, thickness, label=None):
        img_h, img_w = img.shape[0:2]
        img = cv2.rectangle(img, top_left, bottom_right, colour, thickness)
        if label is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            label_thickness = 2
            label_bg_colour = (0, 0, 0)
            label_text_colour = (255, 255, 255)
            (label_width, label_height), baseline = cv2.getTextSize(label, font, 
                                                                    font_scale, 
                                                                    label_thickness)

            # Draw background of label
            lbl_box_topleft = (top_left[0], max(0, top_left[1] - label_height - baseline))
            lbl_box_bottomright = (min(img_w - 1, top_left[0] + label_width), top_left[1])
            img = cv2.rectangle(img, lbl_box_topleft, lbl_box_bottomright, label_bg_colour, -1)  # -1 implies filled rectangle

            # Draw label text
            label_org = (top_left[0], max(0, lbl_box_bottomright[1] - baseline // 2))
            img = cv2.putText(img, label, label_org, font, font_scale, 
                                label_text_colour, label_thickness)

            return img

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

    @staticmethod
    def _iou(bbox_1, bbox_2):
        intersection = AbstractObjectTraker._find_intersection(bbox_1, bbox_2)
        if intersection is None:
            return 0

        _, _, w, h = intersection
        intersect_area = w * h
        union = bbox_1[2] * bbox_1[3] + bbox_2[2] * bbox_2[3] - intersect_area
        iou = intersect_area / union
        return iou
