import os
from transformers import AutoImageProcessor, DetrForObjectDetection
from AbstractObjectTracker import AbstractObjectTraker


class NeuralNetObjectTracker(AbstractObjectTraker):
    def __init__(self):
        super().__init__()
        module_dir = os.path.dirname(os.path.realpath(__file__))
        self._image_processor = AutoImageProcessor.from_pretrained(f'{module_dir}/detr-resnet-50')
        self._model = DetrForObjectDetection.from_pretrained(f'{module_dir}/detr-resnet-50')
        self._prev_bbox = None
        self._detection_thres = 0.9

    def set_starting_bbox(self, bbox):
        self._prev_bbox = bbox

    def set_detection_threshold(self, threshold):
        self._detection_thres = threshold

    def set_object_of_interest(self, obj_label):
        self._obj_of_interest = obj_label

    def track(self, frame):
        img_processed = self._image_processor(frame, return_tensors='pt')
        outputs = self._model(**img_processed)

        h, w, *_ = frame.shape
        target_sizes = [(h, w)]
        results = self._image_processor.post_process_object_detection(outputs,
                                                                      threshold=self._detection_thres,
                                                                      target_sizes=target_sizes)[0]

        max_iou = 0
        det_bbox = [0, 0, 0, 0]
        det_score = 0
        for label, box, score in zip(results['labels'], results['boxes'], results['scores']):
            label = self._model.config.id2label[label.item()]
            if label == self._obj_of_interest:
                xmin, ymin, xmax, ymax = [int(i) for i in box.tolist()]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                bbox = (xmin, ymin, w, h)
                iou = self._iou(bbox, self._prev_bbox)
                if iou > max_iou:
                    max_iou = iou
                    det_bbox = bbox
                    det_score = score.item()

        if max_iou > 0:
            self._prev_bbox = det_bbox

        return det_bbox, det_score


if __name__ == '__main__':
    import argparse as ap
    from NeuralNetObjectTracker import NeuralNetObjectTracker
    from run import run

    parser = ap.ArgumentParser()
    parser.add_argument('--seq', '-s', type=int, required=True)
    parser.add_argument('--obj', '-o', required=True)
    parser.add_argument('--thres', '-t', type=float, default=0.9)
    args = parser.parse_args()

    tracker = NeuralNetObjectTracker()
    run('neural-net', tracker, args.seq, obj_of_interest=args.obj, detection_thres=args.thres, precision_thres=20, iou_thres=0.5)
