import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from MotionModel import MotionModel


def run(mode, data, current_timestep, dt, future_seconds):
    model = MotionModel(mode)
    tracks = data['predict_list']
    agents_of_interest = data['all_agent'][tracks]
    past_trajs = agents_of_interest[:, :11]

    horizon = int(future_seconds / dt)
    start_predict_timestep = current_timestep + 1

    model.set_current_trajectory(past_trajs, dt)
    predicted = model.predict(future_seconds)
    groundtruth = agents_of_interest[:, start_predict_timestep:start_predict_timestep+horizon, :2]
    valids = agents_of_interest[:, start_predict_timestep:start_predict_timestep+horizon, 9] == 1

    valid_pred = []
    valid_gt = []

    ade = []
    fde = []

    for i in range(len(tracks)):
        pred = predicted[i][valids[i]]
        gt = groundtruth[i][valids[i]]
        valid_pred.append(pred)
        valid_gt.append(gt)
        ade.append(model.evaluate('ade', pred, gt))
        fde.append(model.evaluate('fde', pred, gt))

    return ade, fde, valid_pred, valid_gt


def visualise(data, current_timestep, predicted, groundtruth, save=False):
    road_polylines = data['road_polylines']
    scenario_id = data['scenario_id']
    sdc_track_id = data['sdc_track_index']
    all_agent = data['all_agent']
    sdc_current_state = all_agent[sdc_track_id][current_timestep]
    tracks = data['predict_list']

    ax = plt.gca()
    fig = plt.gcf()
    fig.set_facecolor('xkcd:grey')
    ax.set_facecolor('xkcd:grey')
    for polyline in road_polylines:
        map_type = polyline[0,6]
        if map_type == 6:
            plt.plot(polyline[:, 0], polyline[:, 1], 'w', linestyle='dashed', linewidth=1)
        elif map_type == 7:
            plt.plot(polyline[:, 0], polyline[:, 1], 'w', linestyle='solid', linewidth=1)
        elif map_type == 8:
            plt.plot(polyline[:, 0], polyline[:, 1], 'w', linestyle='solid', linewidth=1)
        elif map_type == 9:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=1)
        elif map_type == 10:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=1)
        elif map_type == 11:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=1)
        elif map_type == 12:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=1)
        elif map_type == 13:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='dotted', linewidth=1)
        elif map_type == 15:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)
        elif map_type == 16:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)

    # Draw bounding box of each agent
    # Blue rectangle denotes tracked agents
    for i, agent in enumerate(all_agent):
        x, y = agent[0, :2]
        w, h = agent[0, 3:5]
        heading = np.rad2deg(agent[0, 6])
        bottom_left = (x - (w / 2), y - (h / 2))
        
        if i in tracks:
            colour = 'b'
        else:
            colour = 'k'

        ax.add_artist(Rectangle(bottom_left, w, h, heading, rotation_point='center', color=colour))

    past_trajs = all_agent[tracks][:, :current_timestep+1]

    for agent in range(len(past_trajs)):
        past_valids = past_trajs[agent][:, 9] == 1
        plt.plot(past_trajs[agent][past_valids][:, 0], past_trajs[agent][past_valids][:, 1], 'b', linestyle='solid', linewidth=1)
        plt.plot(groundtruth[agent][:, 0], groundtruth[agent][:, 1], 'b', linestyle='solid', linewidth=1, label='Ground truth')
        plt.plot(predicted[agent][:, 0], predicted[agent][:, 1], 'r', linestyle='dashed', linewidth=1, label='Predicted')

    ax.axis([-70+ sdc_current_state[0], 70+ sdc_current_state[0], -70+ sdc_current_state[1], 70 + sdc_current_state[1]])

    if save:
        filename = f'visualization/{scenario_id}.png'
        plt.savefig(filename)

    plt.show()
