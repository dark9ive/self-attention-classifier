import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, device):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.device = device

        self.values = nn.Parameter(torch.empty(self.embed_dim*self.heads, self.embed_dim, device=device))
        self.keys = nn.Parameter(torch.empty(self.embed_dim*self.heads, self.embed_dim, device=device))
        self.queries = nn.Parameter(torch.empty(self.embed_dim*self.heads, self.embed_dim, device=device))

        # Initialize parameters
        nn.init.normal_(self.values)
        nn.init.normal_(self.keys)
        nn.init.normal_(self.queries)
        '''
        self.values = nn.Linear(self.embed_dim, self.heads*self.embed_dim, device=device)
        self.keys = nn.Linear(self.embed_dim, self.heads*self.embed_dim, device=device)
        self.queries = nn.Linear(self.embed_dim, self.heads*self.embed_dim, device=device)
        '''
        self.fc_out = nn.Linear(embed_dim * heads, embed_dim, device=device)

    def forward(self, x, seqlen):

        x = x[:seqlen]

        #   shape = (seq_len, dim*heads)
        q_mat = torch.mm(x, torch.t(self.queries))
        k_mat = torch.mm(x, torch.t(self.keys))
        v_mat = torch.mm(x, torch.t(self.values))
        '''
        #   shape = (seq_len, dim*heads)
        q_mat = self.queries(x)
        k_mat = self.keys(x)
        v_mat = self.values(x)
        '''

        result = torch.zeros((self.heads, self.embed_dim), device=self.device)

        for i in range(self.heads):

            #   shape = (seq_len, dim)
            q_mat_part = torch.t(torch.t(q_mat)[self.embed_dim*i:self.embed_dim*(i+1)])
            #   shape = (dim, seq_len)
            k_mat_part = torch.t(k_mat)[self.embed_dim*i:self.embed_dim*(i+1)]
            #   shape = (seq_len, dim)
            v_mat_part = torch.t(torch.t(v_mat)[self.embed_dim*i:self.embed_dim*(i+1)])
            
            #   shape = (seq_len, seq_len)
            SA_scores = torch.softmax(torch.mm(q_mat_part, k_mat_part), dim=1)
            
            result[i] = torch.sum(torch.t(torch.mm(SA_scores, v_mat_part)))

        result = self.fc_out(torch.reshape(result, (-1,)))
        
        return result


class SelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, heads, num_classes, device):
        super(SelfAttentionClassifier, self).__init__()
        self.dim = embed_dim
        self.device = device
        self.attention = SelfAttention(embed_dim, heads, device)
        self.classifier = nn.Linear(embed_dim, num_classes, device=device)
    
    def forward(self, xs, lens):
        embs = torch.zeros((len(lens), self.dim), device=self.device)
        for i, (x, seq_len) in enumerate(zip(xs, lens)):
            embs[i] = self.attention(x, seq_len)
        
        #embs = self.attention(xs, lens)

        x = self.classifier(embs)
        return F.log_softmax(x, dim=1)
        