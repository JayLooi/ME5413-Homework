import numpy as np


class MotionModel:
    def __init__(self, mode):
        self._mode = mode.lower()
        self._accel = None
        self._vel = None
        self._dt = None
        self._initial_pos = None

    def set_current_trajectory(self, trajectory, dt):
        self._initial_pos = trajectory[:, -1, 0:2]  # x, y coordinates
        self._vel = trajectory[:, -1, 7:9]          # vel_x, vel_y
        self._dt = dt
        if self._mode == 'cv':
            self._accel = np.array([0., 0.])

        elif self._mode == 'ca':
            self._accel = (self._vel - trajectory[:, -2, 7:9]) / dt

        else:
            raise ValueError(f'Unknown motion model mode {self._mode}')

    def _model(self, time_step):
        t = time_step * self._dt
        return 0.5 * self._accel * (t**2) + self._vel * t + self._initial_pos

    def predict(self, seconds):
        if self._accel is None or self._vel is None or \
           self._dt is None or self._initial_pos is None:
            raise RuntimeError(f'Past trajectory was not specified, '
                               'please invoke \'set_current_trajectory\' method')

        horizon = int(seconds / self._dt)
        predicted = [np.expand_dims(self._model(step), axis=1) for step in range(1, horizon + 1)]

        return np.concatenate(predicted, axis=1)

    @staticmethod
    def evaluate(metric, predicted, groundtruth):
        metric = metric.lower()
        norm = np.linalg.norm(predicted - groundtruth, axis=1)
        if metric == 'ade':
            result = np.average(norm)

        elif metric == 'fde':
            result = norm[-1]

        else:
            raise ValueError(f'Unknown evaluation metric {metric}')

        return result


if __name__ == '__main__':
    import argparse as ap
    import matplotlib.pyplot as plt

    parser = ap.ArgumentParser()
    parser.add_argument('-d', '--data-path', required=True)
    parser.add_argument('-t', '--type', required=True)
    parser.add_argument('-s', '--seconds', type=float, required=True)
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    tracks = data['predict_list']
    agents_of_interest = data['all_agent'][tracks]
    past_trajs = agents_of_interest[:, :11]
    road_polylines = data['road_polylines']
    scenario_id = data['scenario_id']

    dt = 0.1

    if args.type.lower() not in ('cv', 'ca'):
        raise ValueError(f'Unknown MotionModel type {args.type}')

    model = MotionModel(args.type)
    model.set_current_trajectory(past_trajs, dt)
    predicted = model.predict(args.seconds)
    horizon = int(args.seconds / dt)
    groundtruth = agents_of_interest[:, 11:11+horizon, :2]
    valids = agents_of_interest[:, 11:11+horizon, 9] == 1

    print(f'scenario_id={scenario_id}')
    for i, agent in enumerate(tracks):
        pred = predicted[i][valids[i]]
        gt = groundtruth[i][valids[i]]
        ade = model.evaluate('ade', pred, gt)
        fde = model.evaluate('fde', pred, gt)
        print(f'agent={agent}, ADE={ade}, FDE={fde}')
