from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import (clusters2labels, comps2boundaries, connected_components,
                    graph_propagation, remove_single)


@POSTPROCESSOR.register_module()
class DRRGPostprocessor(BasePostprocessor):
    """Merge text components and construct boundaries of text instances.

    Args:
        link_thr (float): The edge score threshold.
    """

    def __init__(self, link_thr, **kwargs):
        assert isinstance(link_thr, float)
        self.link_thr = link_thr

    def __call__(self, edges, scores, text_comps):
        """
        Args:
            edges (ndarray): The edge array of shape N * 2, each row is a node
                index pair that makes up an edge in graph.
            scores (ndarray): The edge score array of shape (N,).
            text_comps (ndarray): The text components.

        Returns:
            List[list[float]]: The predicted boundaries of text instances.
        """
        assert len(edges) == len(scores)
        assert text_comps.ndim == 2
        assert text_comps.shape[1] == 9

        vertices, score_dict = graph_propagation(edges, scores, text_comps)
        clusters = connected_components(vertices, score_dict, self.link_thr)
        pred_labels = clusters2labels(clusters, text_comps.shape[0])
        text_comps, pred_labels = remove_single(text_comps, pred_labels)
        boundaries = comps2boundaries(text_comps, pred_labels)

        return boundaries
