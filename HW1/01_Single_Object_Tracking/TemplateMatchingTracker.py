import cv2
from AbstractObjectTracker import AbstractObjectTracker


class TemplateMatchingTracker(AbstractObjectTracker):
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
        self._prev_bbox = None
        self._h_offset = -1
        self._v_offset = -1

    def set_template(self, template):
        self._template = template

    def set_search_region(self, starting_bbox, h_offset, v_offset):
        self._prev_bbox = starting_bbox
        self._h_offset = h_offset
        self._v_offset = v_offset

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

        # Determine the region of frame to be searched for matching
        x0_search = 0
        y0_search = 0
        x1_search = frame.shape[1] - 1
        y1_search = frame.shape[0] - 1
        if (self._v_offset > 0) or (self._h_offset > 0):
            x0_tracked, y0_tracked, w_tracked, h_tracked = self._prev_bbox

            if self._h_offset > 0:
                x0_search = max(x0_tracked - self._h_offset, 0)
                x1_search = min(x0_tracked + w_tracked - 1 + self._h_offset, x1_search)

            if self._v_offset > 0:
                y0_search = max(y0_tracked - self._v_offset, 0)
                y1_search = min(y0_tracked + h_tracked - 1 + self._v_offset, y1_search)

            frame = frame[y0_search:y1_search+1, x0_search:x1_search+1, :]

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

        tracked = (top_left[0] + x0_search, top_left[1] + y0_search, bbox_width, bbox_height)
        self._prev_bbox = tracked

        return tracked, score


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
