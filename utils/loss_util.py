import torch
import torch.nn.functional as F
import torch.nn as nn


def infoNCE(feat_ori, feat_pos, tau=0.1):
    feat_ori = feat_ori.view(feat_ori.size(0), -1)
    feat_pos = feat_pos.view(feat_pos.size(0), -1)
    # L-2 normalization
    feat_ori = F.normalize(feat_ori, dim=-1)
    feat_pos = F.normalize(feat_pos, dim=-1)
    b = feat_pos.shape[0]
    # logits
    l_pos = torch.einsum('nc,nc->n', [feat_ori, feat_pos]).unsqueeze(-1)  # (N,1)
    l_neg = torch.einsum('nc,ck->nk', [feat_ori, feat_ori.transpose(0, 1)])  # (N, N)
    logits = (1 - torch.eye(b)).type_as(l_neg) * l_neg + torch.eye(b).type_as(l_pos) * l_pos
    # loss
    logits = logits / tau
    labels = torch.arange(b, dtype=torch.long)
    if torch.cuda.is_available():
        labels = labels.cuda()
    loss = F.cross_entropy(logits, labels)

    return loss


def getVoxelDistance(voxel1, voxel2):
    voxel1, voxel2 = voxel1, voxel2
    lis = []
    for i in range(voxel1.shape[0]):
        dist = (torch.sum((voxel1[i] - voxel2[i]) ** 2))
        lis.append(dist/(32 * 32 *32))
    return torch.stack(lis)


def getVoxelDistance2(voxel1_batch, voxel2_batch):
    voxel1_batch = voxel1_batch.detach()
    voxel2_batch = voxel2_batch.detach()
    voxel1_batch = voxel1_batch.view(voxel1_batch.size(0),-1)
    voxel2_batch = voxel2_batch.view(voxel2_batch.size(0),-1)
    criterion1 = nn.MSELoss(reduction='none')
    dists = criterion1(voxel1_batch, voxel2_batch)
    dists = torch.sum(dists,dim=-1)/ (32*32*32)
    return dists


def info3DNCE(feat_ori, feat_pos, label, tau=0.1):
    feat_ori = feat_ori.view(feat_ori.size(0),-1)
    feat_pos = feat_pos.view(feat_pos.size(0),-1)

    feat_ori = F.normalize(feat_ori, dim=-1)
    feat_pos = F.normalize(feat_pos, dim=-1)
    feat_all = feat_ori.clone()
    label_all = label.clone()
    b = feat_ori.shape[0]

    label_ori_rep = label.reshape(-1, 1, 32,32,32).repeat(1, b, 1, 1, 1)
    label_all_rep = label_all.reshape(1, -1, 32,32,32).repeat(b, 1, 1, 1, 1)
    dist = getVoxelDistance2(label_ori_rep.reshape(-1, 32,32,32), label_all_rep.reshape(-1, 32,32,32))
    dist = dist.reshape(b, b)
    threshold = 0.1

    mark_matrix = torch.zeros([b, b], dtype=torch.float)
    for index_1 in range(b):
        weight_sum = 1.0
        for index_2, d in enumerate(dist[index_1]):
            if index_2 == index_1:
                mark_matrix[index_1][index_2] = 1.0
                continue
            if d < threshold:
                mark_matrix[index_1][index_2] = (1-dist[index_1][index_2])*10
                weight_sum += 1
        # rescale weight
        mark_matrix[index_1] = mark_matrix[index_1] / weight_sum
    if torch.cuda.is_available():
        mark_matrix = mark_matrix.cuda()

    l_pos = torch.exp(torch.einsum('nc,ck->nk', [feat_ori, feat_pos.transpose(0, 1)]) / tau)  # (N, N)
    l_pos = torch.einsum('nc,nc->n', [l_pos, mark_matrix]).unsqueeze(-1)  # (N, 1)
    l_neg = torch.exp(torch.einsum('nc,ck->nk', [feat_ori, feat_all.transpose(0, 1)]) / tau)  # (N, N)

    logits = torch.cat([l_pos, l_neg], dim=1)

    # loss
    loss = - torch.log(logits[:, 0] / torch.sum(logits, -1))
    return loss.mean()

if __name__=='__main__':
    feat = torch.rand(5,2048)
    feat_pos = torch.rand(5,2048)
    label = torch.rand(5,1,32,32,32)
    print(info3DNCE(feat,feat_pos,label))
