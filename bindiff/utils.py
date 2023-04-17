import torch
import torch.nn.functional as F
from einops import rearrange
from tmtools import tm_align


def kabsch(A, B):
    assert len(A) == len(B)
    centroid_A = torch.mean(A, dim=0)
    centroid_B = torch.mean(B, dim=0)
    AA = A - centroid_A[None, :]
    BB = B - centroid_B[None, :]
    H = torch.matmul(BB.T, AA)
    U, S, Vt = torch.linalg.svd(H)
    R = torch.matmul(Vt.T, U.T)
    t = torch.matmul(-R, centroid_B) + centroid_A
    return R, t


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    pos_1, pos_2 = pos_1.cpu(), pos_2.cpu()
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def calc_distmap_loss(pos_1, pos_2):

    def distmap(crd):
        return (
            rearrange(crd, "... i c -> ... i () c")
            - rearrange(crd, "... j c -> ... () j c")
        ).norm(dim=-1)

    dist1 = distmap(pos_1)
    dist2 = distmap(pos_2)
    return F.mse_loss(dist1, dist2)
