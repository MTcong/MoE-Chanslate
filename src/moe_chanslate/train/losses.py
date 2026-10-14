import torch
import torch.nn.functional as F


def distill_loss(student_logits, idx, lp, labels=None, ce_weight=0.0):
    """Forward KL on the teacher's top-k support (+ optional CE on next-token labels).

    student_logits [B,T,V]; idx/lp [B,T,k] teacher top-k ids and log-probs (position t
    predicts token t+1, same as the student, so no shift). labels [B,T] are the
    *already shifted* targets (labels[:, t] = ids[:, t+1]; last position = -100).
    """
    logq = F.log_softmax(student_logits.float(), dim=-1)
    q_at = logq.gather(-1, idx.long())
    p = lp.float().exp()                    # true teacher probs; tail mass beyond top-k is ignored
    kl = (p * (lp.float() - q_at)).sum(-1)  # [B,T], exactly 0 when student == teacher
    valid = torch.ones_like(kl, dtype=torch.bool)
    if labels is not None:
        valid = labels != -100
    loss = (kl * valid).sum() / valid.sum().clamp_min(1)
    if ce_weight and labels is not None:
        ce = F.nll_loss(logq.flatten(0, 1), labels.flatten(), ignore_index=-100)
        loss = loss + ce_weight * ce
    return loss
