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
        super().__init__()
        self.set_template(template)
        self.set_mode(mode)

    def set_template(self, template):
        self._template = template

    def set_mode(self, mode):
        mode = mode.lower()
        if mode not in self._TM_MODES.keys():
            print(f'[ERROR] Unknown template matching mode {mode}, please select either:')
            print(',\n'.join(self._TM_MODES.keys()))
            return

        self._mode = self._TM_MODES[mode]

    def list_mode(self):
        return list(self._TM_MODES.keys())

    def track(self, frame):
        if self._template is None:
            print('[ERROR] No template provided')
            return

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
    from TemplateMatchingTracker import TemplateMatchingTracker
    from run import run

    parser = ap.ArgumentParser()
    parser.add_argument('--seq', '-s', type=int, required=True)
    parser.add_argument('--mode', '-m', required=True)
    args = parser.parse_args()

    tracker = TemplateMatchingTracker()
    run('template-matching', tracker, args.seq, mode=args.mode.strip(), precision_thres=20, iou_thres=0.5)
