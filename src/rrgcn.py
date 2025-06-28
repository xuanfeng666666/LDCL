import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, RGAT, UnionRGCNLayer2, UnionRGATLayer, CompGCNLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
from collections import defaultdict
from src.Sem_decpder import SeConvTransE
class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))

        return x


class MLPLinear_new(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        # 如果未指定 hidden_dim，则默认为 input_dim
        h_dim = hidden_dim or input_dim
        self.linear1 = nn.Linear(input_dim, h_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(h_dim, output_dim)
    def forward(self, x):
        x = self.act(self.linear1(x))
        return self.linear2(x)



class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                            activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn" or self.encoder_name == "kbat" or self.encoder_name == "compgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RGCNCell2(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer2(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0, 
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1,pre_weight=0.7, discount=0, angle=0, use_static=False, pre_type = 'short', 
                 use_cl= False, temperature=0.007, sem_temperature=0.007, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False, hidden_size=None, ent_emb=None, rel_emb=None, e_e_his_emb_matrices=None,
        e_r_his_emb_matrices=None):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.pre_weight = pre_weight
        self.discount = discount
        self.use_static = use_static
        self.pre_type = pre_type
        self.use_cl = use_cl
        self.temp =temperature
        self.sem_temp = sem_temperature
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        self.w1 = nn.Linear(self.h_dim*2, self.h_dim)

        self.jiangwei_transform = nn.Linear(hidden_size, self.h_dim)

        self.w8 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim)).float()
        torch.nn.init.normal_(self.w8)
        self.w9 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim)).float()
        torch.nn.init.normal_(self.w9)

        self.w2 = nn.Linear(self.h_dim, self.h_dim)
        self.w3 = nn.Linear(self.h_dim, self.h_dim)
        self.w4 = nn.Linear(self.h_dim*2, self.h_dim)
        self.w5 = nn.Linear(self.h_dim, self.h_dim)
        self.w6 = nn.Linear(self.h_dim,self.h_dim)
        self.w7 = nn.Linear(self.h_dim, self.h_dim)
        self.w_cl = nn.Linear(self.h_dim*2, self.h_dim)

        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        self.weight_1 = nn.Linear(self.h_dim*2, self.h_dim)
        self.weight_2 = nn.Linear(self.h_dim*2, self.h_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        self.weight_3 = nn.Linear(self.h_dim, 1)
        self.weight_4 = nn.Linear(self.h_dim, 1)
        # self.bias_r = nn.Parameter(torch.zeros(1))
        # —— 新增：保存“按时间 t 的语义实体 / 关系矩阵” —— #
        self.e_e_his_emb_matrices = e_e_his_emb_matrices
        self.e_r_his_emb_matrices = e_r_his_emb_matrices

        self.bias_r = nn.Parameter(torch.Tensor(self.h_dim)).to(self.gpu)
        torch.nn.init.zeros_(self.bias_r)

        self.linear1 = nn.Linear(4096, 200).to(self.gpu)
        self.linear2 = nn.Linear(230, 4096).to(self.gpu)
        # self.rel_emb = nn.Parameter(rel_emb.clone().detach().requires_grad_(True)).to(self.gpu)
        self.ent_emb = nn.Parameter(ent_emb.clone().detach().requires_grad_(True)).to(self.gpu)

        linear_layer = nn.Linear(4096, 200).to(self.gpu)
        rel_emb_resized = linear_layer(rel_emb ).to(self.gpu)
        rel_emb_resized   = rel_emb_resized.repeat(2, 1).to(self.gpu)  # 复制两次，变为 [460, 200]
        self.rel_emb = nn.Parameter(rel_emb_resized).detach().requires_grad_(True).to(self.gpu)



        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.rel_emb,
                             use_cuda,
                             analysis)
        
        self.his_rgcn_layer = RGCNCell2(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.rel_emb,
                             use_cuda,
                             analysis)
        
        self.rgat_layer = RGAT(self.h_dim, self.h_dim, activation=F.rrelu, dropout=dropout, self_loop=True)
        self.projection_model = MLPLinear(self.h_dim, self.h_dim)
    
        
        
        self.dim_reduction = nn.Linear(4096, 200)
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)   

        self.pre_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))
       
        # GRU cell for relation evolving
        self.entity_cell = nn.GRUCell(self.h_dim, self.h_dim)
        self.relation_cell = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "seconvtranse":
            self.decoder_ob = SeConvTransE(num_ents, h_dim, hidden_size, input_dropout, hidden_dropout, feat_dropout)
            
        else:
            raise NotImplementedError 
        

        self.seq = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, h_dim),
        )
        

    def forward(self,sub_graph,T_idx, query_mask, g_list, static_graph, use_cuda):

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        #-----------------全局历史建模-------------------------------------
        self.his_ent, subg_index = self.all_GCN(self.h, sub_graph,use_cuda)
        his_r_emb = F.normalize(self.emb_rel)
        his_att = F.softmax(self.w5(query_mask+ self.his_ent),dim=1)
        his_emb = his_att*self.his_ent
        his_emb = F.normalize(his_emb)

        history_embs = []
        att_embs = []
        his_temp_embs =[]
        his_rel_embs =[]
        if self.pre_type=="all":
            for i, g in enumerate(g_list):
                g = g.to(self.gpu)
                t2 = len(g_list)-i+1
                h_t = torch.cos(self.weight_t2 * t2 + self.bias_t2).repeat(self.num_ents,1)
                self.h =self.w4(torch.concat([self.h,h_t],dim=1))
                temp_e = self.h[g.r_to_e]
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x = temp_e[span[0]:span[1],:]
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    x_input[r_idx] = x_mean
                x_input = self.emb_rel + x_input
                current_h = self.rgcn.forward(g, self.h, [self.emb_rel, self.emb_rel])
                current_h = F.normalize(current_h) if self.layer_norm else current_h
                
                att_e = F.softmax(self.w2(query_mask+current_h),dim=1)
                
                if i == 0:
                    self.h_0 = self.entity_cell(current_h, self.h)    # 第1层输入
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                    
                else:
                    self.h_0 = self.entity_cell(current_h, self.h_0)  # 第2层输出==下一时刻第一层输入
                    self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                   
                time_weight = F.sigmoid(torch.mm(x_input, self.time_gate_weight) + self.time_gate_bias)
                self.hr = time_weight * x_input + (1-time_weight) * self.emb_rel
                self.hr = F.normalize(self.hr) if self.layer_norm else self.hr
                history_embs.append(self.h_0)
                his_rel_embs.append(self.hr)
                his_temp_embs.append(self.h_0)
                self.h = self.h_0
                att_emb = att_e*self.h_0 
                att_embs.append(att_emb.unsqueeze(0))
            att_ent = torch.mean(torch.concat(att_embs,dim=0),dim=0)
            att_ent = F.normalize(att_ent)
            history_emb=  att_ent+history_embs[-1]
            history_emb = F.normalize(history_emb) if self.layer_norm else history_emb
        else:
            self.hr = None
            history_emb = None

        return history_emb, static_emb, self.hr, his_emb, his_r_emb,his_temp_embs,his_rel_embs

    def predict(self, que_pair, sub_graph, T_id, test_graph, num_rels, static_graph, test_triplets, e_e_his_emb=None, e_r_his_emb=None, use_cuda=True):
        with torch.no_grad():
            all_triples = test_triplets

            e_e_his_emb = e_e_his_emb.to(self.gpu)
            e_r_his_emb = e_r_his_emb.to(self.gpu)
            
            #-----------------查询数据处理-------------------------------------
            uniq_e = que_pair[0]
            r_len = que_pair[1]
            r_idx = que_pair[2]
            temp_r = self.rel_emb[r_idx]
            e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
            for span, e_idx in zip(r_len, uniq_e):
                x = temp_r[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                e_input[e_idx] = x_mean

            query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
            e1_emb = self.dynamic_emb[uniq_e]
            rel_emb = e_input[uniq_e] #实体所连的所有关系池化
            query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1))
            query_mask[uniq_e] = query_emb

            embedding, _, r_emb, his_emb, his_r_emb,_,_ = self.forward(sub_graph,T_id, query_mask,test_graph, static_graph, use_cuda)

            if self.pre_type == "all":

                scores_ob = self.decoder_ob.forward(embedding, r_emb, all_triples, his_emb, e_r_his_emb, self.pre_weight, self.pre_type)
                score_seq = F.softmax(scores_ob, dim=1)
                score_en =score_seq
            scores_en = torch.log(score_en)
            return all_triples, scores_en


    def get_loss(self, que_pair, sub_graph, T_idx, glist, triples, static_graph, use_cuda, e_e_his_emb=None, e_r_his_emb=None, semantic_ent_embedding=None):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """

        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_semantic_cl = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        all_triples = triples


        ### --------------查询数据处理-----------------------
        uniq_e = que_pair[0]
        r_len = que_pair[1]
        r_idx = que_pair[2]
        temp_r = self.rel_emb[r_idx]
        
        e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
        for span, e_idx in zip(r_len, uniq_e):
            x = temp_r[span[0]:span[1],:]
            x_mean = torch.mean(x, dim=0, keepdim=True)
            e_input[e_idx] = x_mean

        query_mask = torch.zeros((self.num_ents,self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
        t1 = torch.tensor(T_idx).cuda().to(self.gpu)
        q_t = torch.cos(self.weight_t2 * 0 + self.bias_t2).repeat(self.num_ents,1)
        qe_emb = self.w4(torch.concat([self.dynamic_emb,q_t],dim=1))
        
        e1_emb = qe_emb[uniq_e]

        rel_emb = e_input[uniq_e] 
        query_emb = self.w1(torch.concat([e1_emb,rel_emb],dim=1)) 
        query_mask[uniq_e] = query_emb

        embedding, static_emb, r_emb, his_emb, his_r_emb, his_temp_embs, his_rel_embs = self.forward(sub_graph, T_idx, query_mask, glist, static_graph, use_cuda)

        if self.pre_type == "all":
            scores_ob = self.decoder_ob.forward(embedding, r_emb, all_triples, his_emb, e_r_his_emb, self.pre_weight, self.pre_type)
            score_seq = F.softmax(scores_ob, dim=1)
            score_en = score_seq

        scores_en = torch.log(score_en)
        loss_ent += F.nll_loss(scores_en, triples[:, 2])

        if self.use_cl and self.pre_type=="all":
            
            e_r_his_emb_0_reduced = self.dim_reduction(e_r_his_emb[all_triples[:, 0]]).to(self.gpu)
            e_r_his_emb_1_reduced = self.dim_reduction(e_r_his_emb[all_triples[:, 1]]).to(self.gpu)
           
            # struct_view_list = []
            for id, evolve_emb in enumerate(his_temp_embs):
                
                semantic_ent_emb = torch.cat((e_r_his_emb_0_reduced, e_r_his_emb_1_reduced), dim=1) #0.494159, 0.382445, 0.553656, 0.710080
                
                structure_ent_emb = torch.concat([evolve_emb[all_triples[:, 0]], his_rel_embs[id][all_triples[:, 1]]],dim=1) #:0.499471, 0.389635, 0.556777, 0.710080
              
                x1 = self.w_cl(semantic_ent_emb).to(self.gpu)
                x2 = self.w_cl(structure_ent_emb).to(self.gpu)
                loss_semantic_cl += self.get_loss_conv(x1, x2)
     
        return loss_ent, None, loss_static, None, loss_semantic_cl

    def all_GCN(self,ent_emb, sub_graph, use_cuda):
        sub_graph = sub_graph.to(self.gpu)
        sub_graph.ndata['h'] = ent_emb 
        his_emb = self.his_rgcn_layer.forward(sub_graph, ent_emb, [self.rel_emb, self.rel_emb])
        subg_index = torch.masked_select(
                torch.arange(0, sub_graph.number_of_nodes(), dtype=torch.long).cuda(),
                (sub_graph.in_degrees(range(sub_graph.number_of_nodes())) > 0))
        return F.normalize(his_emb.detach()),subg_index
    
    
    def get_loss_conv(self, ent1_emb, ent2_emb):

        loss_fn = nn.CrossEntropyLoss().to(self.gpu)
        z1 = self.projection_model(ent1_emb)
        z2 = self.projection_model(ent2_emb)
        pred1 = torch.mm(z1, z2.T)
        pred2 = torch.mm(z2, z1.T)
        pred3 = torch.mm(z1, z1.T)
        pred4 = torch.mm(z2, z2.T)
        labels = torch.arange(pred1.shape[0]).to(self.gpu)
        train_cl_loss =(loss_fn(pred1 / self.sem_temp, labels) + loss_fn(pred2 / self.sem_temp, labels)+loss_fn(pred3 / self.sem_temp, labels) + loss_fn(pred4 / self.sem_temp, labels)) / 4
        return train_cl_loss
    
    def id2matrix(self, e_r_his_id, e_e_his_id, local_obj=None):
        """
        Convert IDs to matrices.
        """

        e_r_his_matrix = torch.zeros((len(e_r_his_id), self.num_ents), device=self.gpu).float()
        e_e_his_matrix = torch.zeros((len(e_e_his_id), self.num_rels*2), device=self.gpu).float()
        for i, e_id in enumerate(e_r_his_id):
            e_r_his_matrix[i, e_id] = 1
        for i, r_id in enumerate(e_e_his_id):
            e_e_his_matrix[i, r_id] = 1
        if local_obj:
            for i, obj_id in enumerate(local_obj):
                obj_id = list(obj_id)
                e_r_his_matrix[i, obj_id] = 1
        return e_e_his_matrix, e_r_his_matrix