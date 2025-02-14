import os
import glob
import cv2
from TemplateMatchingTracker import TemplateMatchingTracker
from NeuralNetObjectTracker import NeuralNetObjectTracker


def _parse_kwargs(kwargs, params, default_values):
    if len(params) != len(default_values):
        raise ValueError('params and default_values must have equal length')

    keys = kwargs.keys()
    args = []
    for p, default in zip(params, default_values):
        if p in keys:
            args.append(kwargs[p])
        else:
            args.append(default)

    return args


# Driver program for running the SOT algorithms
def run(method, tracker, data_id, **kwargs):
    TMPL_MATCH = 'template-matching'
    NEURAL_NET = 'neural-net'
    method_choices = (TMPL_MATCH, NEURAL_NET)
    if method not in method_choices:
        raise ValueError(f'Unknown SOT method {method}')

    if (data_id < 0) or (data_id > 5):
        raise ValueError('data_id must be an integer from 1 to 5')

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Paths of images and template bounding box data
    data_folder = f'{script_dir}/data/seq{data_id}'
    images_fp = sorted(glob.glob(os.path.join(data_folder, 'img/*.jpg')))
    firsttrack_fp = f'{data_folder}/firsttrack.txt'
    groundtruth_fp = f'{data_folder}/groundtruth.txt'

    with open(firsttrack_fp, 'r') as f:
        bbox = list(map(lambda n: int(n.strip()), f.readline().split(',')))

    # parse common args
    precision_thres, iou_thres = _parse_kwargs(kwargs, ('precision_thres', 'iou_thres'), (20, 0.5))

    if method == TMPL_MATCH:
        if not isinstance(tracker, TemplateMatchingTracker):
            raise TypeError(f'Incorrect tracker type for {method} method, '
                            'expects \'TemplateMatchingTracker\' but got '
                            f'\'{type(tracker).__name__}\'')

        mode_choices = tracker.list_mode()
        mode = _parse_kwargs(kwargs, ('mode',), (mode_choices[0],))[0]
        if mode not in mode_choices:
            raise ValueError(f'Unknown template matching mode {mode}')

        # Results path
        results_folder = f'{script_dir}/results/1_template_matching/{mode}'
        if not os.path.isdir(results_folder):
            os.makedirs(results_folder)

        result_fp = f'{results_folder}/trackresults_TM_seq{data_id}.txt'

        # Crop template from firsttrack bbox
        x, y, w, h = bbox
        first_frame = cv2.cvtColor(cv2.imread(images_fp[0]), cv2.COLOR_BGR2RGB)
        template = first_frame[y:y+h, x:x+w, :]

        # Run the tracking algorithm
        tracker.set_template(template)
        tracker.set_mode(mode)

    elif method == NEURAL_NET:
        if not isinstance(tracker, NeuralNetObjectTracker):
            raise TypeError(f'Incorrect tracker type for {method} method, '
                            'expects \'NeuralNetObjectTracker\' but got '
                            f'\'{type(tracker).__name__}\'')

        obj, det_thres = _parse_kwargs(kwargs, ('obj_of_interest', 'detection_thres'), (None, 0.9))

        # Results path
        results_folder = f'{script_dir}/results/2_objectdetection_withassociation/det_thres={det_thres}'
        if not os.path.isdir(results_folder):
            os.makedirs(results_folder)

        result_fp = f'{results_folder}/trackresults_TM_seq{data_id}.txt'

        tracker.set_starting_bbox(bbox)
        tracker.set_detection_threshold(det_thres)
        tracker.set_object_of_interest(obj)

    else:
        raise ValueError('Unknown tracking algorithm {method}')

    tracker.track_seq(images_fp)
    tracker.save_track_bbox(result_fp)
    tracker.save_tracking_video(results_folder, f'trackresults_TM_seq{data_id}', groundtruth_fp)
    precision = tracker.evaluate(groundtruth_fp, 'precision', threshold=precision_thres).mean()
    success = tracker.evaluate(groundtruth_fp, 'success', threshold=iou_thres).mean()

    if method == TMPL_MATCH:
        print(f'Seq={data_id}, tm mode={mode}, precision={precision}, success={success}')
    elif method == NEURAL_NET:
        print(f'Seq={data_id}, detection_thres={det_thres}, precision={precision}, success={success}')
