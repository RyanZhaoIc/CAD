import torch
import torch.nn as nn
import torch.nn.functional as F

class Estimator(nn.Module):
    def __init__(self,args,base_encoder):
        super().__init__()

        self.encoder = base_encoder(num_class=args.num_class, feat_dim=128, name=args.arch)

        self.moco_queue=args.queue
        self.low_dim=128
        # create the queue_feature
        self.register_buffer("queue_feat", torch.randn(self.moco_queue, self.low_dim)) # embedding pool
        self.queue_feat = F.normalize(self.queue_feat, dim=0)
        # create the queue_distribution of label
        self.register_buffer("queue_dist", torch.randn(self.moco_queue, args.num_class)) # distribution pool
        self.register_buffer("queue_diff_targets",torch.randn(self.moco_queue)) # partial pool
        self.register_buffer("queue_target",torch.randn(self.moco_queue,1)) # target pool
        # create the queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_feat, keys_dist, diff_targets):
        batch_size=keys_feat.shape[0]

        ptr = int(self.queue_ptr)
        
        if ptr+batch_size <= self.moco_queue:
            self.queue_feat[ptr:ptr+batch_size]=keys_feat
            self.queue_dist[ptr:ptr+batch_size]=keys_dist
            self.queue_diff_targets[ptr:ptr+batch_size]=diff_targets
        else:
            in_ptr = self.moco_queue - ptr
            over_ptr = batch_size - in_ptr
            self.queue_feat[ptr:ptr+in_ptr]=keys_feat[:in_ptr]
            self.queue_dist[ptr:ptr+in_ptr]=keys_dist[:in_ptr]
            self.queue_diff_targets[ptr:ptr+in_ptr]=diff_targets[:in_ptr]
            self.queue_feat[:over_ptr]=keys_feat[in_ptr:]
            self.queue_dist[:over_ptr]=keys_dist[in_ptr:]
            self.queue_diff_targets[:over_ptr]=diff_targets[in_ptr:]
        ptr = (ptr + batch_size) % self.moco_queue # move pointer

        self.queue_ptr[0]=ptr


    def forward(self, img_distill, img_diff_w, img_diff_distill, partY=None, diff_target_w=None):

        # compute key_k features
        with torch.no_grad():
            # shuffle keys
            shuffle_ids, reverse_ids = get_shuffle_ids(img_distill.shape[0])
            shuffle_ids_diff, reverse_ids_diff = get_shuffle_ids(img_diff_w.shape[0])
            
            img_distill, partY = img_distill[shuffle_ids], partY[shuffle_ids]
            img_diff_w, img_diff_distill, diff_target_w = img_diff_w[shuffle_ids_diff], img_diff_distill[shuffle_ids_diff], diff_target_w[shuffle_ids_diff]
            
            output_k, _ = self.encoder(img_distill)
            output_k = torch.softmax(output_k, dim=1)
            output_k = pos_neg_norm(output_k, partY)
            
            _, feat_diff = self.encoder(img_diff_w)
            output_diff, _ = self.encoder(img_diff_distill)
            
            # undo shuffle
            omega, partY = output_k[reverse_ids], partY[reverse_ids]
            output_diff, feat_diff, diff_target_w = output_diff[reverse_ids_diff], feat_diff[reverse_ids_diff], diff_target_w[reverse_ids_diff]
        
        features_diff = torch.cat((feat_diff, self.queue_feat.clone().detach()), dim=0)
        diff_targets = torch.cat((diff_target_w, self.queue_diff_targets.clone().detach()), dim=0)
        dists_diff = torch.cat((output_diff, self.queue_dist.clone().detach()), dim=0)

        # dequeue and enqueue
        self._dequeue_and_enqueue(feat_diff, output_diff, diff_target_w)

        return features_diff, dists_diff, diff_targets, output_diff, omega

def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def pos_neg_norm(un_conf, partY):
    # candidate set
    part_confidence = un_conf * partY
    comp_confidence = un_conf * (1 - partY)
    
    part_confidence = part_confidence / part_confidence.sum(dim=1).unsqueeze(1).repeat(1, part_confidence.shape[1])
    comp_confidence = comp_confidence / (comp_confidence.sum(dim=1).unsqueeze(1).repeat(1, comp_confidence.shape[1]) + 1e-20)
    
    rec_confidence = part_confidence + comp_confidence

    return rec_confidence


class Classifier(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        self.encoder = base_encoder(name=args.arch,head='mlp',num_class=args.num_class)
    
    def forward(self, img_s, img_cls, eval_only=False, neg_logits=False):
        if eval_only:
            output_s, _=self.encoder(img_cls)
            return output_s
        
        _, feat_s = self.encoder(img_s)
        output_s, _ = self.encoder(img_cls)
        _, _, output_neg = self.encoder(img_cls, True)
        
        if neg_logits:
            return output_s, feat_s, output_neg
        return output_s, feat_s
