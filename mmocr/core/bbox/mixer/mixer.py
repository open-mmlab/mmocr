# Copyright (c) OpenMMLab. All rights reserved.
import torch


class Mixer:
    """Prepare necessary inputs for text recognizers by mixing prediction
    results (text regions) from the text detector with their assigned ground
    truth properties including texts. Intended for text spotter models.

    Args:
        pred_keys (list[str]): The keys accessing the predicted regions in
            ``pred_results``.
        add_gt (bool): Whether to add all the ground truth data to the result.
    """

    def __init__(
        self,
        pred_keys=['bboxes', 'polygons'],
        add_gt=False,
    ):
        self.pred_keys = pred_keys
        self.add_gt = add_gt

    def __call__(self, img_metas, pred_results, gt_inds=None):
        """
        Args:
            img_metas (dict): The original "img_metas".
            pred_results (dict): The prediction results for one image generated
                from the text detector.
            gt_inds (torch.LongTensor): The ground truth assignment results
                from the assigner. Its length should be equal to the number of
                predicted boxes in ``pred_results``.

        Returns:
            dict: New ``img_metas`` for text recognizers.
        """

        self.new_img_metas = {}
        self.img_metas = img_metas
        if gt_inds is not None:
            self._extract_pred(pred_results, gt_inds)
        if self.add_gt:
            self._merge_img_metas(self.img_metas)
        return self.new_img_metas

    def _extract_pred(self, pred_results, gt_inds):
        # Since we will merge everything into datasample (img_metas)
        # Let's just treat it as a dictionary to be merged
        pred_pos_inds = torch.nonzero(gt_inds > 0, as_tuple=True)[0]
        gt_pos_inds = gt_inds[pred_pos_inds]
        gt_pos_inds -= 1  # minus one to sync with the indexes of img_metas

        # Get predictions
        self._merge_img_metas(pred_results, self.pred_keys, pred_pos_inds)

        # Get ground truth
        self._merge_img_metas(
            self.img_metas, self.pred_keys, gt_pos_inds, exclude_keys=True)

    def _merge_img_metas(self,
                         src_img_metas,
                         src_keys=None,
                         src_inds=None,
                         exclude_keys=False):
        """Merge certain part of ``src_img_metas`` with ``self.new_img_metas``.

        Args:
            src_img_metas (dict): Source img_metas from which elements are
                copied. Each fields to be copied must be either a list or a
                Tensor.
            src_keys (list[str] or None): Fields in ``src_img_metas`` that it
                copies from, if ``exclude_keys`` is False.
            src_inds (list[int] or None): Indexing elements at each field to
                be copied. If not specified, all elements will be used.
            exclude_keys (bool): If True, all keys will be copied to
                ``self.new_img_metas`` except for those in ``keys``.

        Warning:
            Setting ``exclude_keys`` as True while leaving ``keys`` empty would
            not make any change to ``self.new_img_metas``.
        """

        if src_keys is None:
            src_keys = src_img_metas.keys()
        all_keys = src_keys if not exclude_keys else src_img_metas.keys()
        for key in all_keys:
            if exclude_keys and key in src_keys:
                continue
            if type(src_img_metas[key]) in (list, tuple):
                assert type(self.new_img_metas.get(key, [])) in (list, tuple)
                if src_inds is not None:
                    self.new_img_metas[key] = self.new_img_metas.get(
                        key,
                        []) + [src_img_metas[key][idx] for idx in src_inds]
                else:
                    self.new_img_metas[key] = self.new_img_metas.get(
                        key, []) + src_img_metas[key]
            elif isinstance(src_img_metas[key], torch.Tensor):
                assert key not in self.new_img_metas or \
                    isinstance(self.new_img_metas[key], torch.Tensor)
                stack_list = []
                if key in self.new_img_metas:
                    stack_list.append(self.new_img_metas[key])
                if src_inds is not None:
                    stack_list.append(src_img_metas[key][src_inds])
                else:
                    stack_list.append(src_img_metas[key])
                self.new_img_metas[key] = torch.cat(stack_list, dim=0)
            else:
                raise NotImplementedError(f'Merging {type(src_img_metas[key])}'
                                          'is not supported yet')
