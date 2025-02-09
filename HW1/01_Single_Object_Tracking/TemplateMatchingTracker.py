import os
import cv2
from AbstractObjectTracker import AbstractObjectTraker


class TemplateMatchingTracker(AbstractObjectTraker):
    _TM_MODES = {
        'sqdiff': cv2.TM_SQDIFF,
        'sqdiff_normed': cv2.TM_SQDIFF_NORMED,
        'cross_correl': cv2.TM_CCORR,
        'cross_correl_normed': cv2.TM_CCORR_NORMED,
        'correl_coeff': cv2.TM_CCOEFF,
        'correl_coeff_normed': cv2.TM_CCOEFF_NORMED
    }

    def __init__(self, template=None, mode='sqdiff'):
        super().__init__(template)
        self.set_mode(mode)

    def set_mode(self, mode):
        mode = mode.lower()
        if mode not in self._TM_MODES.keys():
            print(f'[ERROR] Unknown template matching mode {mode}, please select either:')
            print(',\n'.join(self._TM_MODES.keys()))
            return

        self._mode = self._TM_MODES[mode]

    def track(self, frame):
        if self._template is None:
            print('[ERROR] No template provided')
            return

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(frame, templ=self._template, method=self._mode)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if self._mode in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            top_left = min_loc
            score = min_val

        else:
            top_left = max_loc
            score = max_val

        bbox_width = self._template.shape[1]
        bbox_height = self._template.shape[0]

        return (top_left[0], top_left[1], bbox_width, bbox_height), score


if __name__ == '__main__':
    import argparse as ap
    import glob
    import matplotlib.pyplot as plt
    parser = ap.ArgumentParser()
    parser.add_argument('--seq', '-s', type=int, required=True)
    parser.add_argument('--mode', '-m', required=True)
    args = parser.parse_args()

    mode = args.mode.strip()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Paths of images and template bounding box data
    data_folder = f'{script_dir}/data/seq{args.seq}'
    images_fp = sorted(glob.glob(os.path.join(data_folder, 'img/*.jpg')))
    firsttrack_fp = f'{data_folder}/firsttrack.txt'
    groundtruth_fp = f'{data_folder}/groundtruth.txt'

    # Results path
    results_folder = f'{script_dir}/results/1_template_matching/{mode}'
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    result_fp = f'{results_folder}/trackresults_TM_seq{args.seq}.txt'

    # Crop the template object
    with open(firsttrack_fp, 'r') as f:
        x, y, w, h = list(map(lambda n: int(n.strip()), f.readline().split(',')))
        first_frame = cv2.cvtColor(cv2.imread(images_fp[0]), cv2.COLOR_BGR2RGB)
        template = first_frame[y:y+h, x:x+w, :]
        # first_frame = cv2.cvtColor(cv2.imread(images_fp[0]), cv2.COLOR_BGR2GRAY)
        # template = first_frame[y:y+h, x:x+w]
        # plt.imshow(template)
        # plt.show()

    tracker = TemplateMatchingTracker(template, mode)
    tracker.track_seq(images_fp)
    tracker.save_track_bbox(result_fp)
    tracker.save_tracking_video(results_folder, f'trackresults_TM_seq{args.seq}', groundtruth_fp)
    precision = tracker.evaluate(groundtruth_fp, 'precision', threshold=20).mean()
    success = tracker.evaluate(groundtruth_fp, 'success', threshold=0.5).mean()
    print(f'Seq={args.seq}, tm mode={mode}, precision={precision}, success={success}')
