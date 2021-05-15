import cv2
import numpy as np
import pyclipper
import torch
from mmcv.ops import contour_expand, pixel_group
from numpy.linalg import norm
from shapely.geometry import Polygon
from skimage.morphology import skeletonize

from mmocr.core import points2boundary
from mmocr.core.evaluation.utils import boundary_iou


def filter_instance(area, confidence, min_area, min_confidence):
    return bool(area < min_area or confidence < min_confidence)


def decode(
        decoding_type='pan',  # 'pan' or 'pse'
        **kwargs):
    if decoding_type == 'pan':
        return pan_decode(**kwargs)
    if decoding_type == 'pse':
        return pse_decode(**kwargs)
    if decoding_type == 'db':
        return db_decode(**kwargs)
    if decoding_type == 'textsnake':
        return textsnake_decode(**kwargs)
    if decoding_type == 'fcenet':
        return fcenet_decode(**kwargs)

    raise NotImplementedError


def pan_decode(preds,
               text_repr_type='poly',
               min_text_confidence=0.5,
               min_kernel_confidence=0.5,
               min_text_avg_confidence=0.85,
               min_text_area=16):
    """Convert scores to quadrangles via post processing in PANet. This is
    partially adapted from https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        preds (tensor): The head output tensor of size 6xHxW.
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_confidence (float): The minimal text confidence.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_text_area (int): The minimal text instance region area.
    Returns:
        boundaries: (list[list[float]]): The instance boundary and its
            instance confidence list.
    """
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()

    text_score = preds[0].astype(np.float32)
    text = preds[0] > min_text_confidence
    kernel = (preds[1] > min_kernel_confidence) * text
    embeddings = preds[2:].transpose((1, 2, 0))  # (h, w, 4)

    region_num, labels = cv2.connectedComponents(
        kernel.astype(np.uint8), connectivity=4)
    contours, _ = cv2.findContours((kernel * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    kernel_contours = np.zeros(text.shape, dtype='uint8')
    cv2.drawContours(kernel_contours, contours, -1, 255)
    text_points = pixel_group(text_score, text, embeddings, labels,
                              kernel_contours, region_num,
                              min_text_avg_confidence)

    boundaries = []
    for text_inx, text_point in enumerate(text_points):
        text_confidence = text_point[0]
        text_point = text_point[2:]
        text_point = np.array(text_point, dtype=int).reshape(-1, 2)
        area = text_point.shape[0]

        if filter_instance(area, text_confidence, min_text_area,
                           min_text_avg_confidence):
            continue
        vertices_confidence = points2boundary(text_point, text_repr_type,
                                              text_confidence)
        if vertices_confidence is not None:
            boundaries.append(vertices_confidence)

    return boundaries


def pse_decode(preds,
               text_repr_type='poly',
               min_kernel_confidence=0.5,
               min_text_avg_confidence=0.85,
               min_kernel_area=0,
               min_text_area=16):
    """Decoding predictions of PSENet to instances. This is partially adapted
    from https://github.com/whai362/PSENet.

    Args:
        preds (tensor): The head output tensor of size nxHxW.
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_confidence (float): The minimal text confidence.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_kernel_area (int): The minimal text kernel area.
        min_text_area (int): The minimal text instance region area.
    Returns:
        boundaries: (list[list[float]]): The instance boundary and its
            instance confidence list.
    """
    preds = torch.sigmoid(preds)  # text confidence

    score = preds[0, :, :]
    masks = preds > min_kernel_confidence
    text_mask = masks[0, :, :]
    kernel_masks = masks[0:, :, :] * text_mask

    score = score.data.cpu().numpy().astype(np.float32)  # to numpy

    kernel_masks = kernel_masks.data.cpu().numpy().astype(np.uint8)  # to numpy

    region_num, labels = cv2.connectedComponents(
        kernel_masks[-1], connectivity=4)

    # labels = pse(kernel_masks, min_kernel_area)
    labels = contour_expand(kernel_masks, labels, min_kernel_area, region_num)
    labels = np.array(labels)
    label_num = np.max(labels)
    boundaries = []
    for i in range(1, label_num + 1):
        points = np.array(np.where(labels == i)).transpose((1, 0))[:, ::-1]
        area = points.shape[0]
        score_instance = np.mean(score[labels == i])
        if filter_instance(area, score_instance, min_text_area,
                           min_text_avg_confidence):
            continue

        vertices_confidence = points2boundary(points, text_repr_type,
                                              score_instance)
        if vertices_confidence is not None:
            boundaries.append(vertices_confidence)

    return boundaries


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def db_decode(preds,
              text_repr_type='poly',
              mask_thr=0.3,
              min_text_score=0.3,
              min_text_width=5,
              unclip_ratio=1.5,
              max_candidates=3000):
    """Decoding predictions of DbNet to instances. This is partially adapted
    from https://github.com/MhLiao/DB.

    Args:
        preds (Tensor): The head output tensor of size nxHxW.
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        mask_thr (float): The mask threshold value for binarization.
        min_text_score (float): The threshold value for converting binary map
            to shrink text regions.
        min_text_width (int): The minimum width of boundary polygon/box
            predicted.
        unclip_ratio (float): The unclip ratio for text regions dilation.
        max_candidates (int): The maximum candidate number.

    Returns:
        boundaries: (list[list[float]]): The predicted text boundaries.
    """
    prob_map = preds[0, :, :]
    text_mask = prob_map > mask_thr

    score_map = prob_map.data.cpu().numpy().astype(np.float32)
    text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

    contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boundaries = []
    for i, poly in enumerate(contours):
        if i > max_candidates:
            break
        epsilon = 0.01 * cv2.arcLength(poly, True)
        approx = cv2.approxPolyDP(poly, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(score_map, points)
        if score < min_text_score:
            continue
        poly = unclip(points, unclip_ratio=unclip_ratio)
        if len(poly) == 0 or isinstance(poly[0], list):
            continue
        poly = poly.reshape(-1, 2)
        poly = points2boundary(poly, text_repr_type, score, min_text_width)
        if poly is not None:
            boundaries.append(poly)
    return boundaries


def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return ~canvas | input_mask


def centralize(points_yx,
               normal_sin,
               normal_cos,
               radius,
               contour_mask,
               step_ratio=0.03):

    h, w = contour_mask.shape
    top_yx = bot_yx = points_yx
    step_flags = np.ones((len(points_yx), 1), dtype=np.bool)
    step = step_ratio * radius * np.hstack([normal_sin, normal_cos])
    while np.any(step_flags):
        next_yx = np.array(top_yx + step, dtype=np.int32)
        next_y, next_x = next_yx[:, 0], next_yx[:, 1]
        step_flags = (next_y >= 0) & (next_y < h) & (next_x > 0) & (
            next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                       np.clip(next_x, 0, w - 1)]
        top_yx = top_yx + step_flags.reshape((-1, 1)) * step
    step_flags = np.ones((len(points_yx), 1), dtype=np.bool)
    while np.any(step_flags):
        next_yx = np.array(bot_yx - step, dtype=np.int32)
        next_y, next_x = next_yx[:, 0], next_yx[:, 1]
        step_flags = (next_y >= 0) & (next_y < h) & (next_x > 0) & (
            next_x < w) & contour_mask[np.clip(next_y, 0, h - 1),
                                       np.clip(next_x, 0, w - 1)]
        bot_yx = bot_yx - step_flags.reshape((-1, 1)) * step
    centers = np.array((top_yx + bot_yx) * 0.5, dtype=np.int32)
    return centers


def merge_disks(disks, disk_overlap_thr):
    xy = disks[:, 0:2]
    radius = disks[:, 2]
    scores = disks[:, 3]
    order = scores.argsort()[::-1]

    merged_disks = []
    while order.size > 0:
        if order.size == 1:
            merged_disks.append(disks[order])
            break
        i = order[0]
        d = norm(xy[i] - xy[order[1:]], axis=1)
        ri = radius[i]
        r = radius[order[1:]]
        d_thr = (ri + r) * disk_overlap_thr

        merge_inds = np.where(d <= d_thr)[0] + 1
        if merge_inds.size > 0:
            merge_order = np.hstack([i, order[merge_inds]])
            merged_disks.append(np.mean(disks[merge_order], axis=0))
        else:
            merged_disks.append(disks[i])

        inds = np.where(d > d_thr)[0] + 1
        order = order[inds]
    merged_disks = np.vstack(merged_disks)

    return merged_disks


def textsnake_decode(preds,
                     text_repr_type='poly',
                     min_text_region_confidence=0.6,
                     min_center_region_confidence=0.2,
                     min_center_area=30,
                     disk_overlap_thr=0.03,
                     radius_shrink_ratio=1.03):
    """Decoding predictions of TextSnake to instances. This was partially
    adapted from https://github.com/princewang1994/TextSnake.pytorch.

    Args:
        preds (tensor): The head output tensor of size 6xHxW.
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_region_confidence (float): The confidence threshold of text
            region in TextSnake.
        min_center_region_confidence (float): The confidence threshold of text
            center region in TextSnake.
        min_center_area (int): The minimal text center region area.
        disk_overlap_thr (float): The radius overlap threshold for merging
            disks.
        radius_shrink_ratio (float): The shrink ratio of ordered disks radii.

    Returns:
        boundaries (list[list[float]]): The instance boundary and its
            instance confidence list.
    """
    assert text_repr_type == 'poly'
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()

    pred_text_score = preds[0]
    pred_text_mask = pred_text_score > min_text_region_confidence
    pred_center_score = preds[1] * pred_text_score
    pred_center_mask = pred_center_score > min_center_region_confidence
    pred_sin = preds[2]
    pred_cos = preds[3]
    pred_radius = preds[4]
    mask_sz = pred_text_mask.shape

    scale = np.sqrt(1.0 / (pred_sin**2 + pred_cos**2 + 1e-8))
    pred_sin = pred_sin * scale
    pred_cos = pred_cos * scale

    pred_center_mask = fill_hole(pred_center_mask).astype(np.uint8)
    center_contours, _ = cv2.findContours(pred_center_mask, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

    boundaries = []
    for contour in center_contours:
        if cv2.contourArea(contour) < min_center_area:
            continue
        instance_center_mask = np.zeros(mask_sz, dtype=np.uint8)
        cv2.drawContours(instance_center_mask, [contour], -1, 1, -1)
        skeleton = skeletonize(instance_center_mask)
        skeleton_yx = np.argwhere(skeleton > 0)
        y, x = skeleton_yx[:, 0], skeleton_yx[:, 1]
        cos = pred_cos[y, x].reshape((-1, 1))
        sin = pred_sin[y, x].reshape((-1, 1))
        radius = pred_radius[y, x].reshape((-1, 1))

        center_line_yx = centralize(skeleton_yx, cos, -sin, radius,
                                    instance_center_mask)
        y, x = center_line_yx[:, 0], center_line_yx[:, 1]
        radius = (pred_radius[y, x] * radius_shrink_ratio).reshape((-1, 1))
        score = pred_center_score[y, x].reshape((-1, 1))
        instance_disks = np.hstack([np.fliplr(center_line_yx), radius, score])
        instance_disks = merge_disks(instance_disks, disk_overlap_thr)

        instance_mask = np.zeros(mask_sz, dtype=np.uint8)
        for x, y, radius, score in instance_disks:
            if radius > 0:
                cv2.circle(instance_mask, (int(x), int(y)), int(radius), 1, -1)
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        score = np.sum(instance_mask * pred_text_score) / (
            np.sum(instance_mask) + 1e-8)
        if len(contours) > 0:
            boundary = contours[0].flatten().tolist()
            boundaries.append(boundary + [score])

    return boundaries


def fcenet_decode(
    preds,
    fourier_degree,
    reconstr_points,
    scale,
    alpha=1.0,
    beta=2.0,
    text_repr_type='poly',
    score_thresh=0.8,
    nms_thresh=0.1,
):
    """Decoding predictions of FCENet to instances.

    Args:
        preds (list(Tensor)): The head output tensors.
        fourier_degree (int): The maximum Fourier transform degree k.
        reconstr_points (int): The points number of the polygon reconstructed
            from predicted Fourier coefficients.
        scale (int): The downsample scale of the prediction.
        alpha (float) : The parameter to calculate final scores. Score_{final}
                = (Score_{text region} ^ alpha)
                * (Score_{text center region}^ beta)
        beta (float) : The parameter to calculate final score.
        text_repr_type (str):  Boundary encoding type 'poly' or 'quad'.
        score_thresh (float) : The threshold used to filter out the final
            candidates.
        nms_thresh (float) :  The threshold of nms.

    Returns:
        boundaries (list[list[float]]): The instance boundary and confidence
            list.
    """
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert text_repr_type == 'poly'

    cls_pred = preds[0][0]
    tr_pred = cls_pred[0:2].softmax(dim=0).data.cpu().numpy()
    tcl_pred = cls_pred[2:].softmax(dim=0).data.cpu().numpy()

    reg_pred = preds[1][0].permute(1, 2, 0).data.cpu().numpy()
    x_pred = reg_pred[:, :, :2 * fourier_degree + 1]
    y_pred = reg_pred[:, :, 2 * fourier_degree + 1:]

    score_pred = (tr_pred[1]**alpha) * (tcl_pred[1]**beta)
    tr_pred_mask = (score_pred) > score_thresh
    tr_mask = fill_hole(tr_pred_mask)

    tr_contours, _ = cv2.findContours(
        tr_mask.astype(np.uint8), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)  # opencv4

    mask = np.zeros_like(tr_mask)
    exp_matrix = generate_exp_matrix(reconstr_points, fourier_degree)
    boundaries = []
    for cont in tr_contours:
        deal_map = mask.copy().astype(np.int8)
        cv2.drawContours(deal_map, [cont], -1, 1, -1)

        text_map = score_pred * deal_map
        polygons = contour_transfor_inv(fourier_degree, x_pred, y_pred,
                                        text_map, exp_matrix, scale)
        polygons = poly_nms(polygons, nms_thresh)
        boundaries = boundaries + polygons

    boundaries = poly_nms(boundaries, nms_thresh)
    return boundaries


def poly_nms(polygons, threshold):
    assert isinstance(polygons, list)

    polygons = np.array(sorted(polygons, key=lambda x: x[-1]))

    keep_poly = []
    index = [i for i in range(polygons.shape[0])]

    while len(index) > 0:
        keep_poly.append(polygons[index[-1]].tolist())
        A = polygons[index[-1]][:-1]
        index = np.delete(index, -1)

        iou_list = np.zeros((len(index), ))
        for i in range(len(index)):
            B = polygons[index[i]][:-1]

            iou_list[i] = boundary_iou(A, B)
        remove_index = np.where(iou_list > threshold)
        index = np.delete(index, remove_index)

    return keep_poly


def contour_transfor_inv(fourier_degree, x_pred, y_pred, score_map, exp_matrix,
                         scale):
    """Reconstruct polygon from predicts.

    Args:
        fourier_degree (int): The maximum Fourier degree K.
        x_pred (ndarray): The real part of predicted Fourier coefficients.
        y_pred (ndarray): The image part of predicted Fourier coefficients.
        score_map (ndarray): The final score of candidates.
        exp_matrix (ndarray): A matrix of e^x, where x = 2pi x ikt, and shape
            is (2k+1, n') where n' is reconstructed point number. See Eq.2
            in paper.
        scale (int): The down-sample scale.
    Returns:
        Polygons (list): The reconstructed polygons and scores.
    """
    mask = score_map > 0

    xy_text = np.argwhere(mask)
    dxy = xy_text[:, 1] + xy_text[:, 0] * 1j

    x = x_pred[mask]
    y = y_pred[mask]

    c = x + y * 1j
    c[:, fourier_degree] = c[:, fourier_degree] + dxy
    c *= scale

    polygons = fourier_inverse_matrix(c, exp_matrix=exp_matrix)
    score = score_map[mask].reshape(-1, 1)
    return np.hstack((polygons, score)).tolist()


def fourier_inverse_matrix(fourier_coeff, exp_matrix):
    """ Inverse Fourier transform
    Args:
        fourier_coeff (ndarray): Fourier coefficients shaped (n, 2k+1), with
            n and k being candidates number and Fourier degree respectively.
        exp_matrix (ndarray): A matrix of e^x, where x = 2pi x ikt and shape
            is (2k+1, n') where n' is reconstructed point number.
            See Eq.2 in paper.
    Returns:
        Polygons (ndarray): The reconstructed polygons shaped (n, n')
    """

    assert type(fourier_coeff) == np.ndarray
    assert fourier_coeff.shape[1] == exp_matrix.shape[0]

    n = exp_matrix.shape[1]
    polygons = np.zeros((fourier_coeff.shape[0], n, 2))

    points = np.matmul(fourier_coeff, exp_matrix)
    p_x = np.real(points)
    p_y = np.imag(points)
    polygons[:, :, 0] = p_x
    polygons[:, :, 1] = p_y
    return polygons.astype('int32').reshape(polygons.shape[0], -1)


def generate_exp_matrix(point_num, fourier_degree):
    """ Generate a matrix of e^x, where x = 2pi x ikt. See Eq.2 in paper.
        Args:
            point_num (int): Number of reconstruct points of polygon
            fourier_degree (int): Maximum Fourier degree k
        Returns:
            exp_matrix (ndarray): A matrix of e^x, where x = 2pi x ikt and
            shape is (2k+1, n') where n' is reconstructed point number.
    """
    e = complex(np.e)
    exp_matrix = np.zeros([2 * fourier_degree + 1, point_num], dtype='complex')

    temp = np.zeros([point_num], dtype='complex')
    for i in range(point_num):
        temp[i] = 2 * np.pi * 1j / point_num * i

    for i in range(2 * fourier_degree + 1):
        exp_matrix[i, :] = temp * (i - fourier_degree)

    return np.power(e, exp_matrix)
