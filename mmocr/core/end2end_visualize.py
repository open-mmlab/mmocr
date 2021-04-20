import json

import cv2
import mmcv
import numpy as np


def write_json(obj, fpath):
    """Write json object to file."""
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '), ensure_ascii=False)


def gen_color():
    """Generate BGR color schemes."""
    color_list = [(101, 67, 254), (154, 157, 252), (173, 205, 249),
                  (123, 151, 138), (187, 200, 178), (148, 137, 69),
                  (169, 200, 200), (155, 175, 131), (154, 194, 182),
                  (178, 190, 137), (140, 211, 222), (83, 156, 222)]
    return color_list


def draw_polygons(img, polys):
    """Draw polygons on image.

    Args:
        img (np.ndarray): The original image.
        polys (list[list[float]]): Detected polygons.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    dst_img = img.copy()
    color_list = gen_color()
    out_img = dst_img
    for idx, poly in enumerate(polys):
        poly = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(
            img,
            np.array([poly]),
            -1,
            color_list[idx % len(color_list)],
            thickness=cv2.FILLED)
        out_img = cv2.addWeighted(dst_img, 0.5, img, 0.5, 0)
    return out_img


def get_optimal_font_scale(text, width):
    """Get optimal font scale for cv2.putText.

    Args:
        text (str): Text in box.
        width (int): Width of box.
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale / 10,
            thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale / 10
    return 1


def draw_texts(img, boxes, texts):
    """Draw boxes and texts on empty img.

    Args:
        img (np.ndarray): The original image.
        boxes (list[list[float]]): Detected bounding boxes.
        texts (list[str]): Recognized texts.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    color_list = gen_color()
    h, w = img.shape[:2]
    out_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        new_box = [[x, y] for x, y in zip(box[0:8:2], box[1:9:2])]
        Pts = np.array([new_box], np.int32)
        cv2.polylines(
            out_img, [Pts.reshape((-1, 1, 2))],
            True,
            color=color_list[idx % len(color_list)],
            thickness=1)
        min_x = int(min(box[0:8:2]))
        max_y = int(
            np.mean(np.array(box[1:9:2])) + 0.2 *
            (max(box[1:9:2]) - min(box[1:9:2])))
        font_scale = get_optimal_font_scale(
            text, int(max(box[0:8:2]) - min(box[0:8:2])))
        cv2.putText(out_img, text, (min_x, max_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1)

    return out_img


def end2end_show_result(img, end2end_res):
    """Draw `result`(boxes and texts) on `img`.
    Args:
        img (str or np.ndarray): The image to be displayed.
        end2end_res (dict): Text detect and recognize results.

    Return:
        out_img (np.ndarray): Visualized image.
    """
    img = mmcv.imread(img)
    boxes, texts = [], []
    for res in end2end_res['result']:
        boxes.append(res['box'])
        texts.append(res['text'])
    box_vis_img = draw_polygons(img, boxes)
    text_vis_img = draw_texts(img, boxes, texts)

    h, w = img.shape[:2]
    out_img = np.ones((h, w * 2, 3), dtype=np.uint8)
    out_img[:, :w, :] = box_vis_img
    out_img[:, w:, :] = text_vis_img

    return out_img
