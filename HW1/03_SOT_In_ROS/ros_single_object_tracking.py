#!/usr/bin/python3

import argparse as ap
import sys
import os
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, BoundingBox2D
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-s', '--seq', type=int, required=True)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    sot_folder = os.path.join(script_dir, '../01_Single_Object_Tracking')
    sys.path.append(sot_folder)
    from TemplateMatchingTracker import TemplateMatchingTracker


    data_folder = f'{sot_folder}/data/seq{args.seq}'
    firsttrack_fp = f'{data_folder}/firsttrack.txt'
    groundtruth_fp = f'{data_folder}/groundtruth.txt'
    template_img_fp = f'{data_folder}/img/00000001.jpg'

    with open(firsttrack_fp, 'r') as f:
        bbox = list(map(lambda n: int(n.strip()), f.readline().split(',')))

    groundtruth = np.loadtxt(groundtruth_fp, dtype=np.uint32, delimiter=',')

    tracker = TemplateMatchingTracker()

    # Crop template from firsttrack bbox
    x, y, w, h = bbox
    first_frame = cv2.imread(template_img_fp)
    template = first_frame[y:y+h, x:x+w, :]

    tracker.set_template(template)
    tracker.set_mode('correl_coeff')
    tracker.set_search_region(bbox, 100, 50)

    rospy.init_node('single-object-tracker')
    rviz_img_pub = rospy.Publisher('/me5413/viz_output', Image)
    gt_pub = rospy.Publisher('/me5413/groundtruth', Detection2D)
    tracked_pub = rospy.Publisher('/me5413/track', Detection2D)
    nusnetID_pub = rospy.Publisher('/me5413/nusnetID', String)

    frame_index = 0

    def on_frame_received(data):
        global frame_index
        cv_bridge = CvBridge()
        cv_image = cv_bridge.imgmsg_to_cv2(data)
        tracked_bbox, _ = tracker.track(cv_image)
        bbox_pts = tracker._get_bbox_points(tracked_bbox)
        tracked_bbox_colour = (0, 255, 0)
        gt_bbox_colour = (237, 161, 9)

        if bbox_pts is not None:
            top_left, bottom_right = bbox_pts
            cv_image = tracker._draw_bbox(cv_image, top_left, bottom_right, tracked_bbox_colour, 2, 'Tracked')

        gt_bbox = groundtruth[frame_index]
        top_left, bottom_right = tracker._get_bbox_points(gt_bbox)
        cv_image = tracker._draw_bbox(cv_image, top_left, bottom_right, gt_bbox_colour, 2, 'Ground truth')

        img_msg = cv_bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        rviz_img_pub.publish(img_msg)

        gt_msg = Detection2D()
        gt_msg.bbox = BoundingBox2D()
        gt_msg.bbox.center = Pose2D()
        gt_x, gt_y = (gt_bbox[0] + gt_bbox[2] // 2, gt_bbox[1] + gt_bbox[3] // 2)
        gt_msg.bbox.center.x = gt_x
        gt_msg.bbox.center.y = gt_y
        gt_msg.bbox.size_x = gt_bbox[2]
        gt_msg.bbox.size_y = gt_bbox[3]
        gt_pub.publish(gt_msg)

        tracked_msg = Detection2D()
        tracked_msg.bbox = BoundingBox2D()
        tracked_msg.bbox.center = Pose2D()
        tracked_x, tracked_y = (tracked_bbox[0] + tracked_bbox[2] // 2, tracked_bbox[1] + tracked_bbox[3] // 2)
        tracked_msg.bbox.center.x = tracked_x
        tracked_msg.bbox.center.y = tracked_y
        tracked_msg.bbox.size_x = tracked_bbox[2]
        tracked_msg.bbox.size_y = tracked_bbox[3]
        tracked_pub.publish(tracked_msg)

        if (frame_index % 10) == 0:
            nusnetID_pub.publish('E1192863')

        frame_index = frame_index + 1


    image_sub = rospy.Subscriber('/me5413/image_raw', Image, on_frame_received)

    rospy.spin()
