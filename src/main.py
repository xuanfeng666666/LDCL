import csv
from datetime import datetime
import argparse
import itertools
import os
import sys
import time
import pickle
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append(".")
from rgcn import utils
from rgcn.utils import build_sub_graph, build_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import time
import pandas as pd
import warnings
from DSE import get_historical_embeddings 
import torch
from collections import defaultdict
from collections import OrderedDict


warnings.filterwarnings('ignore')

@torch.no_grad()
def _compute_semantic_matrices_for_t(
        t,
        ent_ent_his_emb,  ent_ent_pair_list,
        ent_rel_his_emb,  ent_rel_pair_list,
        num_nodes,        num_rels,
        device,           dtype=torch.float32):
    """
    返回 (semantic_ent_mat, semantic_rel_mat)，均 shape=[num_nodes, D]
    与原 build_semantic_matrices 里的“按实体聚合”公式完全一致。
    """
    # ---------- entity–entity ----------
    ee_patch = ent_ent_his_emb[t].to(dtype=dtype, device=device)      # [P1, D]
    ee_pairs = ent_ent_pair_list[t]                                   # [(s,o),...]
    D        = ee_patch.size(1)
    sem_ent  = torch.zeros(num_nodes, D, device=device, dtype=dtype)
    cnt_e    = torch.zeros(num_nodes, device=device, dtype=torch.int32)

    for i, (s, o) in enumerate(ee_pairs):
        v = ee_patch[i]
        sem_ent[s] += v;  sem_ent[o] += v
        cnt_e[s]  += 1;   cnt_e[o]  += 1
    mask = cnt_e > 0
    sem_ent[mask] = (sem_ent[mask].T / cnt_e[mask]).T

    # ---------- entity–relation (按实体聚合) ----------
    er_patch = ent_rel_his_emb[t].to(dtype=dtype, device=device)      # [P2, D]
    er_pairs = ent_rel_pair_list[t]                                   # [(s,r),...]
    sem_rel  = torch.zeros_like(sem_ent)
    cnt_r    = torch.zeros_like(cnt_e)

    for j, (s, _) in enumerate(er_pairs):
        sem_rel[s] += er_patch[j]
        cnt_r[s]   += 1
    mask = cnt_r > 0
    sem_rel[mask] = (sem_rel[mask].T / cnt_r[mask]).T

    return sem_ent, sem_rel


class SemanticCache:
    """
    最近 window_size 个时间步的 (semantic_ent_mat, semantic_rel_mat) 滑动缓存。
    cache[t] = (sem_ent, sem_rel)
    """
    def __init__(self, window_size,
                 ent_ent_his_emb,  ent_ent_pair_list,
                 ent_rel_his_emb,  ent_rel_pair_list,
                 num_nodes, num_rels,
                 device):
        self.window_size = window_size
        self.ent_ent_his_emb   = ent_ent_his_emb
        self.ent_ent_pair_list = ent_ent_pair_list
        self.ent_rel_his_emb   = ent_rel_his_emb
        self.ent_rel_pair_list = ent_rel_pair_list
        self.num_nodes = num_nodes
        self.num_rels  = num_rels
        self.device    = device
        self.cache     = OrderedDict()          # {t: (sem_ent, sem_rel)}

    def get(self, t):
        # 如果已缓存，移到队尾并返回
        if t in self.cache:
            self.cache.move_to_end(t)
            return self.cache[t]

        # 未缓存 ➜ 计算并插入
        sem_ent, sem_rel = _compute_semantic_matrices_for_t(
            t,
            self.ent_ent_his_emb,  self.ent_ent_pair_list,
            self.ent_rel_his_emb,  self.ent_rel_pair_list,
            self.num_nodes,        self.num_rels,
            self.device
        )
        self.cache[t] = (sem_ent, sem_rel)

        # 超窗：弹出最早元素并释放显存
        if len(self.cache) > self.window_size:
            _, (old_e, old_r) = self.cache.popitem(last=False)
            del old_e, old_r
            torch.cuda.empty_cache()

        return sem_ent, sem_rel





        
def e2r(triplets, num_rels):
    # 统计同一个查询实体连接不同的关系
    src, rel, dst = triplets.transpose()
    # get all relations
    # uniq_e = np.concatenate((src, dst))
    uniq_e = np.unique(src)
    # generate r2e
    e_to_r = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        e_to_r[src].add(rel)
        # e_to_r[dst].add(rel+num_rels)
    r_len = []
    r_idx = []
    idx = 0
    for e in uniq_e:
        r_len.append((idx,idx+len(e_to_r[e])))
        r_idx.extend(list(e_to_r[e]))
        idx += len(e_to_r[e])
    uniq_e = torch.from_numpy(np.array(uniq_e)).long().cuda()
    r_len = torch.from_numpy(np.array(r_len)).long().cuda()
    r_idx = torch.from_numpy(np.array(r_idx)).long().cuda()
    return [uniq_e, r_len, r_idx]

def get_sample_from_history_graph3(subg_arr, sr_to_sro, triples,num_nodes, num_rels, use_cuda, gpu):
    # q_to_sro = defaultdict(list)
    q_to_sro = set()
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
    all_triples = np.concatenate([triples, inverse_triples])
    # ent_set = set(all_triples[:, 0])
    src_set = set(triples[:, 0])
    dst_set = set(triples[:, 0])

    # ----------------二阶邻居采样-----------------------
    # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
    er_list = list(set([(tri[0],tri[1]) for tri in triples]))
    er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    # ent_list = list(ent_set)
    # rel_list = list(set(all_triples[:, 1]))

    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
    #整合重复三元组并统计三元组的频率，将三元组的频率作为第四列数据
    subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 
    keys = list(sr_to_sro.keys())
    values = list(sr_to_sro.values())
    df_dic =  pd.DataFrame({'sr': keys, 'dst': values}) #将查询字段转化为pandas

    dst_df = df_dic.query('sr in @er_list')  #获取查询实体和关系的pandas
    dst_get = dst_df['dst'].values    #获取目标尾实体
    two_ent = set().union(*dst_get)   #将头实体与尾实体进行整合
    all_ent = list(src_set|two_ent)   
    result = subg_df.query('src in @all_ent')

    dst_df_inv = df_dic.query('sr in @er_list_inv')  #获取查询实体和关系的pandas
    dst_get_inv = dst_df_inv['dst'].values    #获取目标尾实体
    two_ent_inv = set().union(*dst_get_inv)   #将头实体与尾实体进行整合
    all_ent_inv = list(dst_set|two_ent_inv)  
    result_inv = subg_df.query('src in @all_ent_inv')
    #----------------二阶邻居采样-----------------------
    # result = subg_df.query('src in @src_set')
    q_tri = result.to_numpy()
    q_tri_inv = result_inv.to_numpy()

    his_sub = build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu) 
    his_sub_inv = build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu)
    return  his_sub,his_sub_inv



def test(model,
         history_list,
         test_list,
         num_rels,
         num_nodes,
         use_cuda,
         all_ans_list,
         all_ans_r_list,
         model_name,
         static_graph,
         history_time,
         mode,
         cache,        # 新增：dict[int, Tensor]，每个 t 对应 [M_e×D] 的历史实体嵌入
                 # 新增：dict[int, Tensor]，每个 t 对应 [M_r×D] 的历史实体-关系嵌入
         ent_rel_pair_list      # 原本就有：dict[int, List[(s_id,r_id)]]
):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter = [], []
    ranks_raw_inv, ranks_filter_inv, mrr_raw_list_inv, mrr_filter_list_inv = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        print("------------store_path----------------",model_name)
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    his_list = history_list[:]
    subg_arr = np.concatenate(his_list)
    sr_to_sro = np.load('./data/{}/his_dict/train_s_r.npy'.format(args.dataset), allow_pickle=True).item()

    
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        inverse_triples =test_snap[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        que_pair =  e2r(test_snap, num_rels)
        que_pair_inv =  e2r(inverse_triples, num_rels)

        sub_snap,sub_snap_inv = get_sample_from_history_graph3(subg_arr, sr_to_sro, test_snap , num_nodes,num_rels,use_cuda, args.gpu)

        t = time_idx + history_time

        cur_e_e_his, cur_e_r_his = cache.get(t)


        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input_inv = torch.LongTensor(inverse_triples).cuda() if use_cuda else torch.LongTensor(inverse_triples)
        # cur_e_e_his = torch.tensor(ent_ent_his_emb[t], device=args.gpu)
        # cur_e_e_his = e_e_his_emb_matrices[t].to(args.gpu)
        # cur_e_r_his = e_r_his_ent_matrices[t].to(args.gpu)

        # cur_e_r_his = torch.tensor(e_r_his_ent_matrices[t], device=args.gpu)
        # … 接下来传给模型预测 …
        test_triples, final_score = model.predict(
            que_pair,
            sub_snap,
            time_idx,
            history_glist,
            num_rels,
            static_graph,
            test_triples_input,
            cur_e_e_his,
            cur_e_r_his,
            use_cuda
        )
        
        inv_test_triples, inv_final_score = model.predict(
                que_pair_inv, sub_snap_inv, time_idx, history_glist, num_rels, static_graph, test_triples_input_inv, cur_e_e_his,
                cur_e_r_his,
                use_cuda
                )


        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        mrr_filter_snap_inv, mrr_snap_inv, rank_raw_inv, rank_filter_inv = utils.get_total_rank(inv_test_triples, inv_final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
            # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_inv.append(rank_raw_inv)
        ranks_filter_inv.append(rank_filter_inv)
        
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
            
        idx += 1
    
    # print('alpha=', alpha)
    mrr_raw,hit_raw = utils.stat_ranks(ranks_raw, "raw")
    mrr_filter,hit_filter = utils.stat_ranks(ranks_filter, "filter")
    mrr_raw_inv,hit_raw_inv = utils.stat_ranks(ranks_raw_inv, "raw_inv")
    mrr_filter_inv,hit_filter_inv = utils.stat_ranks(ranks_filter_inv, "filter_inv")
    all_mrr_raw = (mrr_raw+mrr_raw_inv)/2
    all_mrr_filter = (mrr_filter+mrr_filter_inv)/2
    all_hit_raw, all_hit_filter,all_hit_raw_r, all_hit_filter_r = [],[],[],[]
    
    for hit_id in range(len(hit_raw)):
        all_hit_raw.append((hit_raw[hit_id]+hit_raw_inv[hit_id])/2)
        all_hit_filter.append((hit_filter[hit_id]+hit_filter_inv[hit_id])/2)
    print("(all_raw) MRR, Hits@ (1,3,10):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_raw.item(), all_hit_raw[0],all_hit_raw[1],all_hit_raw[2]))
    print("(all_filter) MRR, Hits@ (1,3,10):{:.6f}, {:.6f}, {:.6f}, {:.6f}".format( all_mrr_filter.item(), all_hit_filter[0],all_hit_filter[1],all_hit_filter[2]))
    
            
    # return mrr_raw, mrr_filter
    return all_mrr_raw, all_mrr_filter

    

def run_experiment(args,ent_embs, rel_embs, ent_ent_his_emb, ent_rel_his_emb,  ent_ent_pair_list, ent_rel_pair_list, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    history_test_time = len(train_list) + len(valid_list)  # test list start time
    history_val_time = len(train_list)  # valid list start time



    num_nodes = data.num_nodes
    num_rels = data.num_rels
    
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    device_for_cache = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    # device 选 cuda or cpu；这里按 GPU 来，与你原先用法保持一致
    cache = SemanticCache(
        window_size = 30,
        ent_ent_his_emb   = ent_ent_his_emb,
        ent_ent_pair_list = ent_ent_pair_list,
        ent_rel_his_emb   = ent_rel_his_emb,
        ent_rel_pair_list = ent_rel_pair_list,
        num_nodes         = num_nodes,
        num_rels          = num_rels,
        device            = device_for_cache
    )



    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    model_name = "{}-len{}-gpu{}-lr{}-{}-{}-{}-{}-{}-{}-{}"\
        .format(args.dataset, args.train_history_len, args.gpu, args.lr, args.temperature,args.pre_weight, args.use_cl, args.pre_type,  args.n_hidden, args.encoder,str(time.time()))
    # os.makedirs(f'{THIS_DIR.parent}/models', exist_ok=True)
    
    model_state_file = './models/' + model_name+ ".pt"
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))


    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list("./data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None


    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        weight=args.weight,
                        pre_weight = args.pre_weight,
                        discount=args.discount,
                        angle=args.angle,
                        use_static=args.add_static_graph,
                        pre_type = args.pre_type,
                        use_cl = args.use_cl,
                        temperature = args.temperature,
                        sem_temperature = args.sem_temperature,

                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis, hidden_size=embedding_dim, ent_emb=ent_embs, rel_emb=rel_embs)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter = test(model, 
                            train_list+valid_list,
                            test_list, 
                            num_rels, 
                            num_nodes, 
                            use_cuda, 
                            all_ans_list_test, 
                            all_ans_list_r_test, 
                            model_state_file, 
                            static_graph,
                            history_test_time, 
                            mode="test",
                                 cache=cache,          ### ← NEW
                                 ent_rel_pair_list=ent_rel_pair_list)
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")

        best_mrr = 0
        his_best = 0
        # alpha = 
        # print("alpha=", alpha)

        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx) 

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                subgraph_arr = np.load('./data/{}/his_graph_for/train_s_r_{}.npy'.format(args.dataset, train_sample_num))
                subgraph_arr_inv = np.load('./data/{}/his_graph_inv/train_o_r_{}.npy'.format(args.dataset, train_sample_num))
                subg_snap = build_graph(num_nodes, num_rels, subgraph_arr, use_cuda, args.gpu)   #取出采样子图
                subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv, use_cuda, args.gpu)

                inverse_triples = output[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                que_pair =  e2r(output[0], num_rels)
                que_pair_inv =  e2r(inverse_triples, num_rels)
                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                triples = torch.from_numpy(output[0]).long().cuda()
                inverse_triples = torch.from_numpy(inverse_triples).long().cuda() 

                

                semantic_ent_matrix, semantic_rel_matrix = cache.get(train_sample_num)

                           

                loss_e1, _, loss_static1, loss_stru_cl1, loss_sem_cl1  = model.get_loss(
                                    que_pair     = que_pair,
                                    sub_graph    = subg_snap,
                                    T_idx        = train_sample_num,
                                    glist        = history_glist,
                                    triples      = triples,
                                    static_graph = static_graph,
                                    use_cuda     = use_cuda,
                                    e_e_his_emb  = semantic_ent_matrix,             # [B, hidden_size]，只给对比损失用
                                    e_r_his_emb  = semantic_rel_matrix,      # 【改成】完整的 [num_nodes, hidden_size]
                                    
                    semantic_ent_embedding=ent_embs
                                )
                    

                # ---- 反向任务: t, r^{-1} -> h ----
                loss_e2, _, loss_static2, loss_stru_cl2, loss_sem_cl2 = model.get_loss(
                    que_pair     = que_pair_inv,
                    sub_graph    = subg_snap_inv,
                    T_idx        = train_sample_num,
                    glist        = history_glist,
                    triples      = inverse_triples,
                    static_graph = static_graph,
                    use_cuda     = use_cuda,
                    e_e_his_emb  = semantic_ent_matrix,
                    e_r_his_emb  = semantic_rel_matrix,
                    semantic_ent_embedding=ent_embs
                )
                
                loss_e = 0.5*(loss_e1+loss_e2)
                loss_static = 0.5*(loss_static1+loss_static2)
                loss_sem_cl = 0.5*(loss_sem_cl1+loss_sem_cl2)

                

                loss = loss_e+ loss_static + args.alpha*loss_sem_cl
                    
                losses.append(loss.item())
                losses_e.append(loss_e.item())
                # losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()
                # break
            
            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-static:{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_static), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter = test(model, 
                                    train_list, 
                                    valid_list, 
                                    num_rels, 
                                    num_nodes, 
                                    use_cuda, 
                                    all_ans_list_valid, 
                                    all_ans_list_r_valid, 
                                    model_state_file, 
                                    static_graph,
                                    history_val_time, 
                                    mode="train",  cache=cache, 
                                    ent_rel_pair_list=ent_rel_pair_list)
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter < best_mrr:
                        his_best += 1
                        if epoch >= args.n_epochs:
                            break
                        if his_best>=5:
                            break
                    else:
                        his_best=0
                        best_mrr = mrr_filter
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            torch.cuda.empty_cache()
        mrr_raw, mrr_filter = test(model, 
                            train_list+valid_list,
                            test_list, 
                            num_rels, 
                            num_nodes, 
                            use_cuda, 
                            all_ans_list_test, 
                            all_ans_list_r_test, 
                            model_state_file, 
                            static_graph,
                            history_test_time, 
                            mode="test", cache=cache, 
                                    ent_rel_pair_list=ent_rel_pair_list)
    return mrr_raw, mrr_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LogCL')

    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, default="GDELT",
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--pre-type",  type=str, default="short",
                        help=["long","short", "all"])
    parser.add_argument("--use-cl",  action='store_true', default=False,
                        help="use the info of  contrastive learning")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="the temperature of cl")
    parser.add_argument("--sem_temperature", type=float, default=0.07,
                        help="the sem_temperature of sem_cl")
    parser.add_argument("--alpha", type=float, default=1,
                        help="the weight of sem_cl")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--pre-weight", type=float, default=0.9,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")
    parser.add_argument("--encoder", type=str, default="uvrgcn", # {uvrgcn,kbat,compgcn}
                        help="method of encoder")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="seconvtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")
    parser.add_argument('--plm', type=str, default='llama3b', help="Large language model")
    parser.add_argument('--num-k', type=int, default=5, help="Number of neighbors for DSEP")
    parser.add_argument('--model-type', type=str, default='llama', help="Type of pre-trained model (e.g., bert, t5)")


    args = parser.parse_args()
    print(args)
    args.__dict__["test_history_len"] = args.__dict__["train_history_len"]



    ent_embs, rel_embs, ent_ent_his_emb, ent_rel_his_emb, ent_ent_his_triplets_id, ent_rel_his_triplets_id, ent_ent_pair_list, ent_rel_pair_list, new_entity_ids = get_historical_embeddings(
            args.dataset, args.num_k,
            plm=args.plm,
            model_type=args.model_type,
            batch_size=args.batch_size,
            gpu=args.gpu,
            save=True
        )
                
    embedding_dim = ent_embs.shape[1]

    run_experiment(
    args,
    ent_embs=ent_embs,
    rel_embs=rel_embs,
    ent_ent_his_emb=ent_ent_his_emb,
    ent_rel_his_emb=ent_rel_his_emb,
    ent_ent_pair_list=ent_ent_pair_list,
    ent_rel_pair_list=ent_rel_pair_list
)


