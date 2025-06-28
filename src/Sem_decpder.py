import math
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class SeConvTransR(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim, semantic_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3):
        super(SeConvTransR, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)

        self.conv1 = torch.nn.Conv1d(5, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(5)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_relations*2)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

        self.w1 = torch.nn.Parameter(torch.Tensor(semantic_dim, semantic_dim)).float()
        torch.nn.init.normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(semantic_dim, embedding_dim)).float()
        torch.nn.init.normal_(self.w2)

        self.bias_r = torch.nn.Parameter(torch.Tensor(semantic_dim))
        torch.nn.init.zeros_(self.bias_r)   

    def forward(self, embedding, emb_rel, triplets, e_e_his_emb=None, partial_embeding=None):
        # e1_embedded_all = F.tanh(embedding)

        e1_embedded_all = embedding
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)

        

        r_e_e_his_emb = e_e_his_emb[len(e_e_his_emb)//2:, ]
        r_e_e_his_emb = torch.mm(r_e_e_his_emb, self.w1) + self.bias_r
        e_e_his_emb = torch.cat([e_e_his_emb[:len(e_e_his_emb)//2, ], r_e_e_his_emb], 0)
        e_e_his_emb = torch.mm(e_e_his_emb, self.w2).unsqueeze(1)
        
        stacked_inputs = torch.cat([e1_embedded, e2_embedded, e_e_his_emb], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            x = torch.mm(x, emb_rel.transpose(1, 0))
        else:
            x = torch.mm(x, emb_rel.transpose(1, 0))
            x = torch.mul(x, partial_embeding)
        return x


# class SeConvTransE(torch.nn.Module):
#     def __init__(self, num_entities, embedding_dim, semantic_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3):

#         super(SeConvTransE, self).__init__()

#         self.inp_drop = torch.nn.Dropout(input_dropout)
#         self.hidden_drop = torch.nn.Dropout(hidden_dropout)
#         self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
#         self.loss = torch.nn.BCELoss()

#         self.conv1 = torch.nn.Conv1d(3, channels, kernel_size, stride=1,
#                                padding=int(math.floor(kernel_size / 2)))
#         self.bn0 = torch.nn.BatchNorm1d(3)
#         self.bn1 = torch.nn.BatchNorm1d(channels)
#         self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
#         self.register_parameter('b', Parameter(torch.zeros(num_entities)))
#         self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
#         # print('==================')
#         # print('semantic_dim', semantic_dim)

#         # print('semantic_dim.shape:', semantic_dim.shape)
#         self.fc1 = torch.nn.Linear(semantic_dim, embedding_dim)

#     def forward(self, embedding, emb_rel, triplets, his_emb, e_r_his_emb, pre_weight, pre_type):


#         device = next(self.parameters()).device
#         embedding   = embedding.to(device)
#         emb_rel     = emb_rel.to(device)
#         triplets    = triplets.to(device)
#         his_emb     = his_emb.to(device)
#         e_r_his_emb = e_r_his_emb.to(device)

        

#         batch_entity_idx = triplets[:, 0]                # [batch_size]
#         batch_his_ent    = his_emb[batch_entity_idx]     # [batch_size, h_dim]
#         batch_his_rel    = e_r_his_emb[batch_entity_idx] # [batch_size, semantic_dim]

#         # batch_size = batch_entity_idx.size(0)
#         e1_all     = F.tanh(embedding) 

#         batch_size = len(triplets)
        
#         if pre_type =='all':
#             e1_embed_all = e1_all[batch_entity_idx] 
#             e1_embedded   = e1_embed_all.unsqueeze(1)
#             embedded_his  = F.tanh(batch_his_ent).unsqueeze(1)
#             # e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
#             # e1_his_embedded = embedded_his[triplets[:, 0]].unsqueeze(1)
#             e1_embed = pre_weight*e1_embedded + (1-pre_weight)*embedded_his
            

#         rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)  # batch_size,1,h_dim
#         e_r_his_emb = self.fc1(batch_his_rel).unsqueeze(1)  # batch_size,1,h_dim
#         stacked_inputs = torch.cat([e1_embed, rel_embedded, e_r_his_emb], 1)  # batch_size,2,h_dim
#         stacked_inputs = self.bn0(stacked_inputs)  # batch_size,2,h_dim
#         x = self.inp_drop(stacked_inputs)  # batch_size,2,h_dim
#         x = self.conv1(x)  # batch_size,2,h_dim
#         x = self.bn1(x)  # batch_size,channels,h_dim
#         x = F.relu(x)
#         x = self.feature_map_drop(x)
#         x = x.view(batch_size, -1)  # batch_size,channels*h_dim
#         x = self.fc(x)  # batch_size,channels*h_dim
#         x = self.hidden_drop(x)  # batch_size,h_dim
#         if batch_size > 1:
#             x = self.bn2(x)
#         x = F.relu(x)
        
#         x = torch.mm(x, e1_all.transpose(1, 0))
        
#         return x
    


class SeConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, semantic_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3):

        super(SeConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(3, channels, kernel_size, stride=1,
                            padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(3)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        # print('==================')
        # print('semantic_dim', semantic_dim)

        # llama
        # print('semantic_dim.shape:', semantic_dim.shape)
        self.fc1 = torch.nn.Linear(semantic_dim, embedding_dim)

        # bert
        # self.fc1 = torch.nn.Linear(768, embedding_dim)

    def forward(self, embedding, emb_rel, triplets, his_emb, e_r_his_emb, pre_weight, pre_type):


        device = next(self.parameters()).device
        embedding   = embedding.to(device)
        emb_rel     = emb_rel.to(device)
        triplets    = triplets.to(device)
        his_emb     = his_emb.to(device)
        e_r_his_emb = e_r_his_emb.to(device)

        

        batch_entity_idx = triplets[:, 0]                # [batch_size]
        batch_his_ent    = his_emb[batch_entity_idx]     # [batch_size, h_dim]
        batch_his_rel    = e_r_his_emb[batch_entity_idx] # [batch_size, semantic_dim]

        # batch_size = batch_entity_idx.size(0)
        e1_all     = F.tanh(embedding) 

        batch_size = len(triplets)
        
        if pre_type =='all':
            e1_embed_all = e1_all[batch_entity_idx] 
            e1_embedded   = e1_embed_all.unsqueeze(1)
            embedded_his  = F.tanh(batch_his_ent).unsqueeze(1)
            # e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
            # e1_his_embedded = embedded_his[triplets[:, 0]].unsqueeze(1)
            e1_embed = pre_weight*e1_embedded + (1-pre_weight)*embedded_his
            

        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)  # batch_size,1,h_dim
        e_r_his_emb = self.fc1(batch_his_rel).unsqueeze(1)  # batch_size,1,h_dim
        stacked_inputs = torch.cat([e1_embed, rel_embedded, e_r_his_emb], 1)  # batch_size,2,h_dim
        stacked_inputs = self.bn0(stacked_inputs)  # batch_size,2,h_dim
        x = self.inp_drop(stacked_inputs)  # batch_size,2,h_dim
        x = self.conv1(x)  # batch_size,2,h_dim
        x = self.bn1(x)  # batch_size,channels,h_dim
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)  # batch_size,channels*h_dim
        x = self.fc(x)  # batch_size,channels*h_dim
        x = self.hidden_drop(x)  # batch_size,h_dim
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        
        x = torch.mm(x, e1_all.transpose(1, 0))
        
        return x

