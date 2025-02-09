import cv2


def get_bbox_points(bbox):
    top_left = bbox[0:2]
    bbox_w, bbox_h = bbox[2:4]
    bottom_right = (top_left[0] + bbox_w, top_left[1] + bbox_h)
    return top_left, bottom_right


def draw_bbox(img, top_left, bottom_right, colour, thickness, label=None):
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
