# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import shutil
import urllib
import warnings

import cv2
import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import mmocr.utils as utils


def overlay_mask_img(img, mask):
    """Draw mask boundaries on image for visualization.

    Args:
        img (ndarray): The input image.
        mask (ndarray): The instance mask.

    Returns:
        img (ndarray): The output image with instance boundaries on it.
    """
    assert isinstance(img, np.ndarray)
    assert isinstance(mask, np.ndarray)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    return img


def show_feature(features, names, to_uint8, out_file=None):
    """Visualize a list of feature maps.

    Args:
        features (list(ndarray)): The feature map list.
        names (list(str)): The visualized title list.
        to_uint8 (list(1|0)): The list indicating whether to convent
            feature maps to uint8.
        out_file (str): The output file name. If set to None,
            the output image will be shown without saving.
    """
    assert utils.is_ndarray_list(features)
    assert utils.is_type_list(names, str)
    assert utils.is_type_list(to_uint8, int)
    assert utils.is_none_or_type(out_file, str)
    assert utils.equal_len(features, names, to_uint8)

    num = len(features)
    row = col = math.ceil(math.sqrt(num))

    for i, (f, n) in enumerate(zip(features, names)):
        plt.subplot(row, col, i + 1)
        plt.title(n)
        if to_uint8[i]:
            f = f.astype(np.uint8)
        plt.imshow(f)
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)


def show_img_boundary(img, boundary):
    """Show image and instance boundaires.

    Args:
        img (ndarray): The input image.
        boundary (list[float or int]): The input boundary.
    """
    assert isinstance(img, np.ndarray)
    assert utils.is_type_list(boundary, int) or utils.is_type_list(
        boundary, float)

    cv2.polylines(
        img, [np.array(boundary).astype(np.int32).reshape(-1, 1, 2)],
        True,
        color=(0, 255, 0),
        thickness=1)
    plt.imshow(img)
    plt.show()


def show_pred_gt(preds,
                 gts,
                 show=False,
                 win_name='',
                 wait_time=0,
                 out_file=None):
    """Show detection and ground truth for one image.

    Args:
        preds (list[list[float]]): The detection boundary list.
        gts (list[list[float]]): The ground truth boundary list.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): The value of waitKey param.
        out_file (str): The filename of the output.
    """
    assert utils.is_2dlist(preds)
    assert utils.is_2dlist(gts)
    assert isinstance(show, bool)
    assert isinstance(win_name, str)
    assert isinstance(wait_time, int)
    assert utils.is_none_or_type(out_file, str)

    p_xy = [p for boundary in preds for p in boundary]
    gt_xy = [g for gt in gts for g in gt]

    max_xy = np.max(np.array(p_xy + gt_xy).reshape(-1, 2), axis=0)

    width = int(max_xy[0]) + 100
    height = int(max_xy[1]) + 100

    img = np.ones((height, width, 3), np.int8) * 255
    pred_color = mmcv.color_val('red')
    gt_color = mmcv.color_val('blue')
    thickness = 1

    for boundary in preds:
        cv2.polylines(
            img, [np.array(boundary).astype(np.int32).reshape(-1, 1, 2)],
            True,
            color=pred_color,
            thickness=thickness)
    for gt in gts:
        cv2.polylines(
            img, [np.array(gt).astype(np.int32).reshape(-1, 1, 2)],
            True,
            color=gt_color,
            thickness=thickness)
    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def imshow_pred_boundary(img,
                         boundaries_with_scores,
                         labels,
                         score_thr=0,
                         boundary_color='blue',
                         text_color='blue',
                         thickness=1,
                         font_scale=0.5,
                         show=True,
                         win_name='',
                         wait_time=0,
                         out_file=None,
                         show_score=False):
    """Draw boundaries and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        boundaries_with_scores (list[list[float]]): Boundaries with scores.
        labels (list[int]): Labels of boundaries.
        score_thr (float): Minimum score of boundaries to be shown.
        boundary_color (str or tuple or :obj:`Color`): Color of boundaries.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename of the output.
        show_score (bool): Whether to show text instance score.
    """
    assert isinstance(img, (str, np.ndarray))
    assert utils.is_2dlist(boundaries_with_scores)
    assert utils.is_type_list(labels, int)
    assert utils.equal_len(boundaries_with_scores, labels)
    if len(boundaries_with_scores) == 0:
        warnings.warn('0 text found in ' + out_file)
        return None

    utils.valid_boundary(boundaries_with_scores[0])
    img = mmcv.imread(img)

    scores = np.array([b[-1] for b in boundaries_with_scores])
    inds = scores > score_thr
    boundaries = [boundaries_with_scores[i][:-1] for i in np.where(inds)[0]]
    scores = [scores[i] for i in np.where(inds)[0]]
    labels = [labels[i] for i in np.where(inds)[0]]

    boundary_color = mmcv.color_val(boundary_color)
    text_color = mmcv.color_val(text_color)
    font_scale = 0.5

    for boundary, score in zip(boundaries, scores):
        boundary_int = np.array(boundary).astype(np.int32)

        cv2.polylines(
            img, [boundary_int.reshape(-1, 1, 2)],
            True,
            color=boundary_color,
            thickness=thickness)

        if show_score:
            label_text = f'{score:.02f}'
            cv2.putText(img, label_text,
                        (boundary_int[0], boundary_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def imshow_text_char_boundary(img,
                              text_quads,
                              boundaries,
                              char_quads,
                              chars,
                              show=False,
                              thickness=1,
                              font_scale=0.5,
                              win_name='',
                              wait_time=-1,
                              out_file=None):
    """Draw text boxes and char boxes on img.

    Args:
        img (str or ndarray): The img to be displayed.
        text_quads (list[list[int|float]]): The text boxes.
        boundaries (list[list[int|float]]): The boundary list.
        char_quads (list[list[list[int|float]]]): A 2d list of char boxes.
            char_quads[i] is for the ith text, and char_quads[i][j] is the jth
            char of the ith text.
        chars (list[list[char]]). The string for each text box.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename of the output.
    """
    assert isinstance(img, (np.ndarray, str))
    assert utils.is_2dlist(text_quads)
    assert utils.is_2dlist(boundaries)
    assert utils.is_3dlist(char_quads)
    assert utils.is_2dlist(chars)
    assert utils.equal_len(text_quads, char_quads, boundaries)

    img = mmcv.imread(img)
    char_color = [mmcv.color_val('blue'), mmcv.color_val('green')]
    text_color = mmcv.color_val('red')
    text_inx = 0
    for text_box, boundary, char_box, txt in zip(text_quads, boundaries,
                                                 char_quads, chars):
        text_box = np.array(text_box)
        boundary = np.array(boundary)

        text_box = text_box.reshape(-1, 2).astype(np.int32)
        cv2.polylines(
            img, [text_box.reshape(-1, 1, 2)],
            True,
            color=text_color,
            thickness=thickness)
        if boundary.shape[0] > 0:
            cv2.polylines(
                img, [boundary.reshape(-1, 1, 2)],
                True,
                color=text_color,
                thickness=thickness)

        for b in char_box:
            b = np.array(b)
            c = char_color[text_inx % 2]
            b = b.astype(np.int32)
            cv2.polylines(
                img, [b.reshape(-1, 1, 2)], True, color=c, thickness=thickness)

        label_text = ''.join(txt)
        cv2.putText(img, label_text, (text_box[0, 0], text_box[0, 1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        text_inx = text_inx + 1

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def tile_image(images):
    """Combined multiple images to one vertically.

    Args:
        images (list[np.ndarray]): Images to be combined.
    """
    assert isinstance(images, list)
    assert len(images) > 0

    for i, _ in enumerate(images):
        if len(images[i].shape) == 2:
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)

    widths = [img.shape[1] for img in images]
    heights = [img.shape[0] for img in images]
    h, w = sum(heights), max(widths)
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)

    offset_y = 0
    for image in images:
        img_h, img_w = image.shape[:2]
        vis_img[offset_y:(offset_y + img_h), 0:img_w, :] = image
        offset_y += img_h

    return vis_img


def imshow_text_label(img,
                      pred_label,
                      gt_label,
                      show=False,
                      win_name='',
                      wait_time=-1,
                      out_file=None):
    """Draw predicted texts and ground truth texts on images.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        pred_label (str): Predicted texts.
        gt_label (str): Ground truth texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str): The filename of the output.
    """
    assert isinstance(img, (np.ndarray, str))
    assert isinstance(pred_label, str)
    assert isinstance(gt_label, str)
    assert isinstance(show, bool)
    assert isinstance(win_name, str)
    assert isinstance(wait_time, int)

    img = mmcv.imread(img)

    src_h, src_w = img.shape[:2]
    resize_height = 64
    resize_width = int(1.0 * src_w / src_h * resize_height)
    img = cv2.resize(img, (resize_width, resize_height))
    h, w = img.shape[:2]

    if is_contain_chinese(pred_label):
        pred_img = draw_texts_by_pil(img, [pred_label], None)
    else:
        pred_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        cv2.putText(pred_img, pred_label, (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
    images = [pred_img, img]

    if gt_label != '':
        if is_contain_chinese(gt_label):
            gt_img = draw_texts_by_pil(img, [gt_label], None)
        else:
            gt_img = np.ones((h, w, 3), dtype=np.uint8) * 255
            cv2.putText(gt_img, gt_label, (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 0, 0), 2)
        images.append(gt_img)

    img = tile_image(images)

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def imshow_edge_node(img,
                     result,
                     boxes,
                     idx_to_cls={},
                     show=False,
                     win_name='',
                     wait_time=-1,
                     out_file=None):

    img = mmcv.imread(img)
    h, w = img.shape[:2]

    max_value, max_idx = torch.max(result['nodes'].detach().cpu(), -1)
    node_pred_label = max_idx.numpy().tolist()
    node_pred_score = max_value.numpy().tolist()

    texts, text_boxes = [], []
    for i, box in enumerate(boxes):
        new_box = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]],
                   [box[0], box[3]]]
        Pts = np.array([new_box], np.int32)
        cv2.polylines(
            img, [Pts.reshape((-1, 1, 2))],
            True,
            color=(255, 255, 0),
            thickness=1)
        x_min = int(min([point[0] for point in new_box]))
        y_min = int(min([point[1] for point in new_box]))

        # text
        pred_label = str(node_pred_label[i])
        if pred_label in idx_to_cls:
            pred_label = idx_to_cls[pred_label]
        pred_score = '{:.2f}'.format(node_pred_score[i])
        text = pred_label + '(' + pred_score + ')'
        texts.append(text)

        # text box
        font_size = int(
            min(
                abs(new_box[3][1] - new_box[0][1]),
                abs(new_box[1][0] - new_box[0][0])))
        char_num = len(text)
        text_box = [
            x_min * 2, y_min, x_min * 2 + font_size * char_num, y_min,
            x_min * 2 + font_size * char_num, y_min + font_size, x_min * 2,
            y_min + font_size
        ]
        text_boxes.append(text_box)

    pred_img = np.ones((h, w * 2, 3), dtype=np.uint8) * 255
    pred_img = draw_texts_by_pil(
        pred_img, texts, text_boxes, draw_box=False, on_ori_img=True)

    vis_img = np.ones((h, w * 3, 3), dtype=np.uint8) * 255
    vis_img[:, :w] = img
    vis_img[:, w:] = cv2.cvtColor(np.asarray(pred_img), cv2.COLOR_RGB2BGR)

    if show:
        mmcv.imshow(vis_img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(vis_img, out_file)

    return vis_img


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
        text (str): Text in one box.
        width (int): The box width.
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale / 10,
            thickness=1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1


def draw_texts(img, texts, boxes=None, draw_box=True, on_ori_img=False):
    """Draw boxes and texts on empty img.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else, on a new empty image.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    color_list = gen_color()
    h, w = img.shape[:2]
    if boxes is None:
        boxes = [[0, 0, w, 0, w, h, 0, h]]
    assert len(texts) == len(boxes)

    if on_ori_img:
        out_img = img
    else:
        out_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        if draw_box:
            new_box = [[x, y] for x, y in zip(box[0::2], box[1::2])]
            Pts = np.array([new_box], np.int32)
            cv2.polylines(
                out_img, [Pts.reshape((-1, 1, 2))],
                True,
                color=color_list[idx % len(color_list)],
                thickness=1)
        min_x = int(min(box[0::2]))
        max_y = int(
            np.mean(np.array(box[1::2])) + 0.2 *
            (max(box[1::2]) - min(box[1::2])))
        font_scale = get_optimal_font_scale(
            text, int(max(box[0::2]) - min(box[0::2])))
        cv2.putText(out_img, text, (min_x, max_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1)

    return out_img


def draw_texts_by_pil(img,
                      texts,
                      boxes=None,
                      draw_box=True,
                      on_ori_img=False,
                      font_size=None,
                      font_color=None,
                      draw_pos=None,
                      return_text_size=False):
    """Draw boxes and texts on empty image, especially for Chinese.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else, on a new empty image.
    Return:
        out_img (np.ndarray): Visualized text image.
    """

    color_list = gen_color()
    h, w = img.shape[:2]
    if boxes is None:
        boxes = [[0, 0, w, 0, w, h, 0, h]]
    assert len(boxes) == len(texts)

    if on_ori_img:
        out_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        out_img = Image.new('RGB', (w, h), color=(255, 255, 255))
    out_draw = ImageDraw.Draw(out_img)

    text_sizes = []
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        if len(text) == 0:
            continue
        min_x, max_x = min(box[0::2]), max(box[0::2])
        min_y, max_y = min(box[1::2]), max(box[1::2])
        color = tuple(list(color_list[idx % len(color_list)])[::-1])
        if draw_box:
            out_draw.line(box, fill=color, width=1)
        dirname, _ = os.path.split(os.path.abspath(__file__))
        font_path = os.path.join(dirname, 'font.TTF')
        if not os.path.exists(font_path):
            url = ('http://download.openmmlab.com/mmocr/data/font.TTF')
            print(f'Downloading {url} ...')
            local_filename, _ = urllib.request.urlretrieve(url)
            shutil.move(local_filename, font_path)
        if font_size is None:
            box_width = max(max_x - min_x, max_y - min_y)
            font_size = int(0.9 * box_width / len(text))
        fnt = ImageFont.truetype(font_path, font_size)
        if draw_pos is None:
            draw_pos = (min_x + 1, min_y + 1)
        if font_color is None:
            font_color = (0, 0, 0)
        out_draw.text(draw_pos, text, font=fnt, fill=font_color)
        text_sizes.append(fnt.getsize(text))

    del out_draw

    out_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)

    if return_text_size:
        return out_img, text_sizes

    return out_img


def is_contain_chinese(check_str):
    """Check whether string contains Chinese or not.

    Args:
        check_str (str): String to be checked.

    Return True if contains Chinese, else False.
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def det_recog_show_result(img, end2end_res, out_file=None):
    """Draw `result`(boxes and texts) on `img`.

    Args:
        img (str or np.ndarray): The image to be displayed.
        end2end_res (dict): Text detect and recognize results.
        out_file (str): Image path where the visualized image should be saved.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    img = mmcv.imread(img)
    boxes, texts = [], []
    for res in end2end_res['result']:
        boxes.append(res['box'])
        texts.append(res['text'])
    box_vis_img = draw_polygons(img, boxes)

    if is_contain_chinese(''.join(texts)):
        text_vis_img = draw_texts_by_pil(img, texts, boxes)
    else:
        text_vis_img = draw_texts(img, texts, boxes)

    h, w = img.shape[:2]
    out_img = np.ones((h, w * 2, 3), dtype=np.uint8)
    out_img[:, :w, :] = box_vis_img
    out_img[:, w:, :] = text_vis_img

    if out_file:
        mmcv.imwrite(out_img, out_file)

    return out_img


def draw_edge_result(img, result, edge_thresh=0.5, keynode_thresh=0.5):
    """Draw text and their relationship on empty images.

    Args:
        img (np.ndarray): The original image.
        result (dict): The result of model forward prediction.
        edge_thresh (float): Score threshold for edge classification.
        keynode_thresh (float): Score threshold for node
            (``key``) classification.
    Return:
        out_img (np.ndarray): Visualized relationship image.
    """

    h, w = img.shape[:2]

    vis_area_width = w // 3 * 2
    vis_area_height = h
    dist_key_to_value = vis_area_width // 2
    dist_pair_to_pair = 30

    bbox_x1 = dist_pair_to_pair
    bbox_y1 = 0

    new_w = vis_area_width
    new_h = vis_area_height
    pred_edge_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255

    nodes = result['nodes'].detach().cpu()
    texts = result['img_metas'][0]['ori_texts']
    num_nodes = result['nodes'].size(0)
    edges = result['edges'].detach().cpu()[:, -1].view(num_nodes, num_nodes)
    pairs = (torch.max(edges, edges.T) > edge_thresh).nonzero(as_tuple=True)
    result_pairs = [(n1, n2) if nodes[n1, 1] > nodes[n1, 2] else (n2, n1)
                    for n1, n2 in zip(*pairs) if n1 < n2]
    result_pairs.sort()
    keynow = -1
    posnow = (-1, -1)
    newline_flag = False

    key_font_size = 15
    value_font_size = 15
    key_font_color = (0, 0, 0)
    value_font_color = (0, 0, 255)
    arrow_color = (0, 0, 255)
    for pair in result_pairs:
        key_idx = int(pair[0].item())
        if nodes[key_idx, 1] < keynode_thresh:
            continue
        if key_idx != keynow:
            bbox_y1 += 10
            if newline_flag:
                bbox_x1 += vis_area_width
                pred_edge_img_t = np.ones(
                    (new_h, new_w + vis_area_width, 3), dtype=np.uint8) * 255
                pred_edge_img_t[:new_h, :new_w] = pred_edge_img
                pred_edge_img = pred_edge_img_t
                new_w += vis_area_width
                newline_flag = False
                bbox_y1 = 10
        key_text = texts[key_idx]
        key_pos = (bbox_x1, bbox_y1)
        value_idx = pair[1].item()
        value_text = texts[value_idx]
        value_pos = bbox_x1 + dist_key_to_value, bbox_y1
        if key_idx != keynow:
            keynow = key_idx
            pred_edge_img, text_sizes = draw_texts_by_pil(
                pred_edge_img, [key_text],
                draw_box=False,
                on_ori_img=True,
                font_size=key_font_size,
                font_color=key_font_color,
                draw_pos=key_pos,
                return_text_size=True)
            pos_right_bottom = (key_pos[0] + text_sizes[0][0],
                                key_pos[1] + text_sizes[0][1])
            posnow = (pos_right_bottom[0] + 5, bbox_y1 + 10)
            pred_edge_img = cv2.arrowedLine(
                pred_edge_img, (pos_right_bottom[0] + 5, bbox_y1 + 10),
                (bbox_x1 + dist_key_to_value - 5, bbox_y1 + 10), arrow_color,
                1)
        else:
            if newline_flag:
                pred_edge_img_t = np.ones(
                    (new_h + dist_pair_to_pair, new_w, 3),
                    dtype=np.uint8) * 255
                pred_edge_img_t[:new_h, :new_w] = pred_edge_img
                pred_edge_img = pred_edge_img_t
                new_h += dist_pair_to_pair
            pred_edge_img = cv2.arrowedLine(pred_edge_img, posnow,
                                            (bbox_x1 + dist_key_to_value - 5,
                                             bbox_y1 + 10), arrow_color, 1)
        pred_edge_img = draw_texts_by_pil(
            pred_edge_img, [value_text],
            draw_box=False,
            on_ori_img=True,
            font_size=value_font_size,
            font_color=value_font_color,
            draw_pos=value_pos,
            return_text_size=False)
        bbox_y1 += dist_pair_to_pair
        if bbox_y1 + dist_pair_to_pair >= new_h:
            newline_flag = True

    return pred_edge_img


def imshow_edge(img,
                result,
                boxes,
                show=False,
                win_name='',
                wait_time=-1,
                out_file=None):
    """Display the prediction results of the nodes and edges of the KIE model.

    Args:
        img (np.ndarray): The original image.
        result (dic): The result of model forward prediction.
        boxes (list): The text boxes corresponding to the nodes.
        show (bool): Whether to show the image. Default: False.
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Return:
        out_img (np.ndarray): Visualized result image.
    """
    img = mmcv.imread(img)
    h, w = img.shape[:2]
    color_list = gen_color()

    for i, box in enumerate(boxes):
        new_box = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]],
                   [box[0], box[3]]]
        Pts = np.array([new_box], np.int32)
        cv2.polylines(
            img, [Pts.reshape((-1, 1, 2))],
            True,
            color=color_list[i % len(color_list)],
            thickness=1)

    pred_img_h = h
    pred_img_w = w

    pred_edge_img = draw_edge_result(img, result)
    pred_img_h = max(pred_img_h, pred_edge_img.shape[0])
    pred_img_w += pred_edge_img.shape[1]

    vis_img = np.zeros((pred_img_h, pred_img_w, 3), dtype=np.uint8)
    vis_img[:h, :w] = img
    vis_img[:, w:] = 255

    height_t, width_t = pred_edge_img.shape[:2]
    vis_img[:height_t, w:(w + width_t)] = pred_edge_img

    if show:
        mmcv.imshow(vis_img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(vis_img, out_file)
        res_dic = {
            'boxes': boxes,
            'nodes': result['nodes'].detach().cpu(),
            'edges': result['edges'].detach().cpu(),
            'metas': result['img_metas'][0]
        }
        mmcv.dump(res_dic, f'{out_file}_res.pkl')

    return vis_img
