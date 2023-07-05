import torch.nn as nn
from types import SimpleNamespace
from utils.math import calc_iou, cosine_similarity
from einops import repeat, rearrange
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torch
from utils.math import cw2se
from torchvision.ops import sigmoid_focal_loss
from .loss import l1_loss, iou_loss


class ThresholdMatcher(nn.Module):

    def __init__(self, threshold=0.3) -> None:
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, iou, **kwargs):
        cost = iou
        positive_mask = cost > self.threshold
        positive_mask[torch.argmax(cost, dim=0), torch.arange(cost.shape[1])] = True

        row_ind, col_ind = positive_mask.nonzero(as_tuple=True)
        return cost, row_ind, col_ind


class HungarianMatcher(nn.Module):

    def __init__(self, w_sim, w_iou) -> None:
        super().__init__()
        self.w_sim = w_sim
        self.w_iou = w_iou

    @torch.no_grad()
    def forward(self, sim, iou):
        """
        Args:
            sim: (Nq, Nt_i) cosine similarity matrix between moment and text
            iou: (Nq, Nt_i) IoU matrix between moment and timestamps
        Returns:
            cost_matrix: (Nq, Nt) cost matrix
            row_ind: (Nq, ) row indices
            col_ind: (Nt, ) column indices
        """

        cost = self.w_sim * sim + self.w_iou * iou  # B, Nq, Nt
        # logger.info(f"sim: {sim[0, :10]} \t iou: {iou[0, :10]}")
        row_ind, col_ind = linear_sum_assignment(cost.cpu(), maximize=True)

        device = sim.device
        return cost, torch.as_tensor(row_ind).to(device), torch.as_tensor(col_ind).to(device)


class Criterion(nn.Module):

    def __init__(self,
                 matcher,
                 w_intra=0.0,
                 w_inter=0.0,
                 w_l1=0.0,
                 w_iou=0.0,
                 contrastive_loss="bce",
                 use_aux=True,
                 disable_aux_cst=True,
                 temperature=1.0
                 ) -> None:
        super().__init__()
        self.matcher = matcher
        self.use_aux = use_aux
        self.w_intra = w_intra
        self.w_inter = w_inter
        self.w_l1 = w_l1
        self.w_iou = w_iou
        self.contrastive_loss = contrastive_loss
        self.temperature = temperature
        self.disable_aux_cst = disable_aux_cst

    @torch.no_grad()
    def get_assign_matrix(self, query, reference_boxes, gt, txt_feat, batch_split_size):
        """
        return assign_matrix: (B * Nq, Nt)
        Args:
            query: (Bv, Nq, d_model)
            reference_boxes: (Bv, Nq, 2)
            gt: (Nt, 2)
            txt_feat: (Bq, d_model)
            batch_split_size: (Bq, )
        Returns:
            assign_matrix: (Bv * Nq, Nt) representing which moment is assigned to which timestamp/text
            intra_video_mask: (Bv * Nq, Nt) representing the moment assigned to the same video
        """
        B, Nq = query.shape[:2]
        Nt = gt.shape[0]

        txt_feat_split = torch.split(txt_feat, batch_split_size, dim=0)
        gt_split = torch.split(gt, batch_split_size, dim=0)

        # assign inter video
        assign_matrix_list = []
        intra_video_mask_list = []
        for i, (gt_i, txt_feat_i) in enumerate(zip(gt_split, txt_feat_split)):
            Nt_i = gt_i.shape[0]
            cur_ref_boxes_single_vid = reference_boxes[i]  # single video, single level
            cur_query_single_vid = query[i]  # single video, single level
            ious = calc_iou(repeat(cur_ref_boxes_single_vid, "nq i -> (nq nt_i) i", nt_i=Nt_i),
                            repeat(gt_i, "nt_i i -> (nq nt_i) i", nq=Nq))
            ious = ious.reshape(Nq, Nt_i)  # Nq, Nt_i
            ious[ious < 0.0] = 0.0
            ious[ious > 1.0] = 1.0
            sim = cosine_similarity(cur_query_single_vid, txt_feat_i)  # (Nq, Nt_i)
            cost, row_inds, col_inds = self.matcher(sim=sim, iou=ious)

            cur_assign_matrix = torch.zeros(Nq, Nt_i).to(query.device)
            cur_assign_matrix[row_inds, col_inds] = 1

            assign_matrix_list.append(cur_assign_matrix)
            cur_intra_mask = torch.ones(Nq, Nt_i).to(query.device)
            intra_video_mask_list.append(cur_intra_mask)

        assign_matrix = torch.block_diag(*assign_matrix_list)  # (B * Nq, Nt)
        intra_video_mask = torch.block_diag(*intra_video_mask_list).bool()  # (B * Nq, Nt)
        intermediate = SimpleNamespace(assign_matrix=assign_matrix_list, intra_video_mask=intra_video_mask_list)

        return assign_matrix, intra_video_mask, intermediate

    def forward(self, reference_boxes, query, txt_feat, gt, batch_split_size):
        B = len(batch_split_size)
        Nq = query[0].shape[1]

        loss_dict = dict()
        loss = 0.0

        Nt = gt.shape[0]
        if not self.use_aux:
            reference_boxes = [reference_boxes[-1]]
            query = [query[-1]]
        
        num_lvls = len(reference_boxes)

        for lvl, (cur_query, cur_reference_boxes) in enumerate(zip(query, reference_boxes)):
            assign_matrix, intra_video_mask, intermediate = self.get_assign_matrix(
                query=cur_query,
                txt_feat=txt_feat,
                reference_boxes=cur_reference_boxes,
                gt=gt,
                batch_split_size=batch_split_size)  # (B * Nq, Nt)
            cur_query_flat = rearrange(cur_query, "b nq d -> (b nq) d")
            sim_all = cosine_similarity(cur_query_flat, txt_feat)  # (B * Nq, Nt)

            sim_intra = sim_all[intra_video_mask]
            assign_matrix_intra = assign_matrix[intra_video_mask]

            sim_inter = sim_all[~intra_video_mask]
            assign_matrix_inter = assign_matrix[~intra_video_mask]
            if self.contrastive_loss == "bce":
                loss_intra = F.binary_cross_entropy_with_logits(sim_intra, assign_matrix_intra, reduction="mean")
                loss_inter = F.binary_cross_entropy_with_logits(sim_inter, assign_matrix_inter, reduction="mean")
            elif self.contrastive_loss == "focal":
                loss_intra = sigmoid_focal_loss(sim_intra, assign_matrix_intra, reduction="mean")
                loss_inter = sigmoid_focal_loss(sim_inter, assign_matrix_inter, reduction="mean")
            elif self.contrastive_loss == "infonce":
                row_inds, col_inds = assign_matrix.nonzero(as_tuple=True)
                # moment to text
                sim_all /= self.temperature
                mask_m2t = assign_matrix.clone().bool()
                mask_m2t[row_inds, :] = True
                loss_m2t = F.cross_entropy(sim_all[mask_m2t], assign_matrix[mask_m2t], reduction="mean")

                # text to moment
                mask_t2m = assign_matrix.clone().bool()
                mask_t2m[:, col_inds] = True
                loss_t2m = F.cross_entropy(sim_all[mask_t2m], assign_matrix[mask_t2m], reduction="mean")

                #! simple hack
                loss_intra = 0.05 * (loss_m2t + loss_t2m)
                loss_inter = 0.0

            else:
                raise NotImplementedError()

            if self.disable_aux_cst and lvl < num_lvls - 1:
                loss_intra = loss_inter = 0.0
            loss += self.w_intra * loss_intra + self.w_inter * loss_inter

            cur_reference_boxes_flat = rearrange(cur_reference_boxes, "b nq d -> (b nq) d")
            cur_reference_boxes_flat = cw2se(cur_reference_boxes_flat)
            positive_ref_boxes = cur_reference_boxes_flat[assign_matrix.nonzero()[:, 0], :]

            positive_gt = gt[assign_matrix.nonzero()[:, 1]]

            loss_iou = iou_loss(positive_ref_boxes, positive_gt, iou="giou")
            loss_l1 = l1_loss(positive_ref_boxes, positive_gt)

            # localization loss
            loss_loc = self.w_iou * loss_iou + self.w_l1 * loss_l1
            coeff_los = 1.0 if lvl == num_lvls - 1 else 1 / (num_lvls - 1)
            loss += coeff_los * loss_loc

            loss_dict.update({
                f"loss_intra_{lvl}": loss_intra,
                f"loss_inter_{lvl}": loss_inter,
                f"loss_iou_{lvl}": loss_iou,
                f"loss_l1_{lvl}": loss_l1
            })

        loss_dict["loss"] = loss

        return loss_dict
