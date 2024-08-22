import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class BiEncoderLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        # self.pooling_strategy = pooling_strategy

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """

        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)

        loss_rowwise = self.ce_loss(scores, torch.arange(scores.shape[0], device=scores.device))
        # loss_columnwise = self.ce_loss(scores.T, torch.arange(scores.shape[1], device=scores.device))
        # loss = (loss_rowwise + loss_columnwise) / 2
        return loss_rowwise


class MultiVectorLoss(torch.nn.Module):
    pass

class ColbertLoss(MultiVectorLoss):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        """

        scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)

        # scores = torch.zeros((query_embeddings.shape[0], doc_embeddings.shape[0]), device=query_embeddings.device)
        # for i in range(query_embeddings.shape[0]):
        #     for j in range(doc_embeddings.shape[0]):
        #         # step 1 - dot product --> (s1,s2)
        #         q2d_scores = torch.matmul(query_embeddings[i], doc_embeddings[j].T)
        #         # step 2 -> max on doc  --> (s1)
        #         q_scores = torch.max(q2d_scores, dim=1)[0]
        #         # step 3 --> sum the max score --> (1)
        #         sum_q_score = torch.sum(q_scores)
        #         # step 4 --> assert is scalar
        #         scores[i, j] = sum_q_score

        # assert (scores_einsum - scores < 0.0001).all().item()

        loss_rowwise = self.ce_loss(scores, torch.arange(scores.shape[0], device=scores.device))
        # TODO: comparing between queries might not make sense since it's a sum over the length of the query
        # loss_columnwise = self.ce_loss(scores.T, torch.arange(scores.shape[1], device=scores.device))
        # loss = (loss_rowwise + loss_columnwise) / 2
        return loss_rowwise


class ColbertPairwiseCELoss(MultiVectorLoss):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)

        Positive scores are the diagonal of the scores matrix.
        """

        # Compute the ColBERT scores
        scores = (
            torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
        )  # (batch_size, batch_size)

        # Positive scores are the diagonal of the scores matrix.
        pos_scores = scores.diagonal()  # (batch_size,)

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We exclude the diagonal by setting it to a very low value: since we know the maximum score is 1,
        # we can subtract 1 from the diagonal to exclude it from the maximum operation.
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
        neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores
        # relative to the negative scores.
        # This can be simplified to log-sum-exp of negative scores minus the positive score
        # for numerical stability.
        # torch.vstack((pos_scores, neg_scores)).T.softmax(1)[:, 0].log()*(-1)
        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss


class XtrPairwiseCELoss(MultiVectorLoss):
    def __init__(self):
        super().__init__()
        self.top_k = 3  # Better to use percentage of m(num_query_token)*B(num_document_token) for each batch?
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)

        Positive scores are the diagonal of the scores matrix.
        """

        # Compute the ColBERT scores
        # scores = (
        #     torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
        # )  # (batch_size, batch_size)
        pairwise_scores = torch.einsum(
            "bnd,csd->bcns", query_embeddings, doc_embeddings
        )  # (batch_size, batch_size, num_query_tokens, num_doc_tokens)

        topk_masks = self.generate_top_k_mask(
            pairwise_scores, self.top_k
        )  # (batch_size, batch_size, num_query_tokens, num_doc_tokens)
        aligned = pairwise_scores * topk_masks
        Z = topk_masks.float().max(-1)[0].sum(-1).clamp(min=1e-3)  # noqa: N806
        doc_tok_summed_normalized = aligned.max(-1)[0].sum(-1) / Z
        scores = doc_tok_summed_normalized  # (batch_size, batch_size)

        # Positive scores are the diagonal of the scores matrix.
        pos_scores = scores.diagonal()  # (batch_size,)

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We exclude the diagonal by setting it to a very low value: since we know the maximum score is 1,
        # we can subtract 1 from the diagonal to exclude it from the maximum operation.
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
        neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores
        # relative to the negative scores.
        # This can be simplified to log-sum-exp of negative scores minus the positive score
        # for numerical stability.
        # torch.vstack((pos_scores, neg_scores)).T.softmax(1)[:, 0].log()*(-1)
        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss

    def generate_top_k_mask(self, scores, k):
        batch_size = scores.shape[0]
        num_query_tokens = scores.shape[2]
        num_doc_tokens = scores.shape[3]

        # Reshape to combine the first two dimensions and flatten the last two
        scores_reshaped = scores.reshape(batch_size * batch_size, -1)

        # Get the top k values across both query and doc tokens
        _, top_k_indices = torch.topk(scores_reshaped, k=k, dim=-1)

        # Create a mask
        mask = torch.zeros_like(scores_reshaped, dtype=torch.bool, device=scores.device)
        batch_indices = torch.arange(scores_reshaped.shape[0], device=scores.device).unsqueeze(-1).expand(-1, k)

        mask[batch_indices, top_k_indices] = True

        # Reshape back to the original shape
        mask = mask.reshape(batch_size, batch_size, num_query_tokens, num_doc_tokens)

        return mask


class BiPairwiseCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, dim)
        doc_embeddings: (batch_size, dim)
        """

        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)

        pos_scores = scores.diagonal()
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6
        neg_scores = neg_scores.max(dim=1)[0]

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores
        # relative to the negative scores.
        # This can be simplified to log-sum-exp of negative scores minus the positive score
        # for numerical stability.
        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss


def main():
    num_query_tokens = 5
    num_doc_tokens = 10
    dim = 16
    batch_size = 7

    query_embeddings = torch.randn((batch_size, num_query_tokens, dim))
    doc_embeddings = torch.randn((batch_size, num_doc_tokens, dim))
    loss = XtrPairwiseCELoss()
    print("loss = XtrPairwiseCELoss()")
    loss(query_embeddings, doc_embeddings)


if __name__ == "__main__":
    main()
