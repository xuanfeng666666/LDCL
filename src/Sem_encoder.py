from __future__ import annotations

import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5Model, GPT2Tokenizer, GPT2Model, LlamaTokenizer, LlamaModel, AutoTokenizer
from pathlib import Path
# from safetensors.torch import save_file, load_file
from safetensors.torch import save_file
from safetensors import safe_open
from typing import Dict, List
import gc            # 新增


THIS_DIR = Path(__file__).parent.resolve()





class DataPreprocessor:
    def __init__(self, dataset, num_k, plm='llama'):
        self.dataset = dataset
        self.num_k = num_k
        self.plm = plm
        
    def _build_path(self, *args):
        """
        Construct a file path based on dataset and model type.
        """
        return os.path.join(THIS_DIR.parent, 'data', self.dataset, 'text_emb', self.plm, *args)
    
    def get_paths(self):
        """
        Get file paths for saving embeddings based on the dataset and model.
        """
        base_path = self._build_path()
        path = base_path
        entities_emb_path = self._build_path('entities_emb.pt')
        relations_emb_path = self._build_path('relations_emb.pt')
        ent_ent_his_emb_path = self._build_path(f'ent_ent_his_emb_num{self.num_k}.safetensors')
        ent_rel_his_emb_path = self._build_path(f'ent_rel_his_emb_num{self.num_k}.safetensors')
        return {
            'path': path,
            'entities_emb_path': entities_emb_path, 
            'relations_emb_path': relations_emb_path, 
            'ent_ent_his_emb_path': ent_ent_his_emb_path, 
            'ent_rel_his_emb_path': ent_rel_his_emb_path
        }

    def load_id_data(self, fileNames=['train.txt', 'valid.txt', 'test.txt']):
        """
        Load data from the dataset files.
        """
        all_data = defaultdict(list)
        for fileName in fileNames:
            with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, fileName), 'r') as fr:
                # some datasets have 5 columns(s, r, o, [start_time, end_time]), some have 4(s, r, o, time)
                for line in fr:
                    parts = list(map(int, line.split()))
                    head, rel, tail, time = parts[:4]
                    if self.dataset == 'ICEWS14':  # ICEWS14 starts from 1
                        time -= 1
                    all_data.setdefault(time, []).append([head, rel, tail])
        return all_data
    
    

    def get_id2ent_id2rel(self):
        """
        Load mappings from ID to entity and relation names.
        """
        if self.dataset == 'GDELT':
            id2ent = self._load_id2ent_GDELT()
            id2rel = self._load_id2rel_GDELT()
        else:
            id2ent, id2rel = self._load_id2ent_id2rel_default()
        return id2ent, id2rel

    def _load_id2ent_GDELT(self):
        """
        Load entity mappings specifically for the GDELT dataset.
        """
        with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'entity2id.txt'), 'r') as f:
            return {int(line.split('\t')[1]): line.split('\t')[0].split('(')[0].strip() for line in f}

    def _load_id2rel_GDELT(self):
        """
        Load relation mappings specifically for the GDELT dataset.
        """
        with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'relation2id.txt'), 'r') as f1, \
             open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'CAMEO-eventcodes.txt'), 'r') as f2:
            id2cameoCode = {int(line.split('\t')[1]): line.split('\t')[0] for line in f1}
            cameoCode2rel = {line.split('\t')[0]: line.split('\t')[1].strip() for line in f2}
            return {id_: cameoCode2rel[id2cameoCode[id_]] for id_ in id2cameoCode.keys()}

    def _load_id2ent_id2rel_default(self):
        """
        Load entity and relation mappings for datasets other than GDELT.
        """
        with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'entity2id.txt'), 'r') as f:
            id2ent = {int(line.split('\t')[1]): line.split('\t')[0] for line in f}
        with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'relation2id.txt'), 'r') as f:
            id2rel = {int(line.split('\t')[1]): line.split('\t')[0] for line in f}
        return id2ent, id2rel

    def generate_historical_triplets(self):
        """
        Generate historical triplets for each time step, encoding them into 
        both sentence-based triplets and ID-based triplets for future usage.
        **同时额外记录：**
          - ent_ent_pair_list[t] = [(s_id, o_id), ...] 
          - ent_rel_pair_list[t] = [(s_id, r_id), ...]
        """
        all_data = self.load_id_data()
        id2ent, id2rel = self.get_id2ent_id2rel()
        rel_nums = len(id2rel)
        times = list(all_data.keys())
        
        # Determine time interval (for ICEWS or GDELT datasets)
        # Time interval: GDELT: 15mins ICEWS: 1day ---> 1day = 15mins *4 * 24
        time_interval = times[1] - times[0] if self.dataset != 'GDELT' else (times[1] - times[0]) * 24 * 4
        
        # Initialize containers for triplets and historical triplets
        ent_ent_triplets = defaultdict(list)
        ent_rel_triplets = defaultdict(list)
        ent_ent_his_triplets = {}
        ent_rel_his_triplets = {}
        ent_ent_triplets_id = defaultdict(list)
        ent_rel_triplets_id = defaultdict(list)
        ent_ent_his_triplets_id = {}
        ent_rel_his_triplets_id = {}
        seen_entity_ids = set()  # Track all seen s_id
        new_entity_ids = defaultdict(list)

        # **新增：用来记录每一行历史 embedding 对应的 (s_id,o_id) 和 (s_id,r_id)**
        ent_ent_pair_list = {}
        ent_rel_pair_list = {}
        
        for idx, t in enumerate(tqdm(all_data.keys(), desc="Generating historical triplets")):
            # Process current time step triplets
            train_new_data = torch.from_numpy(np.array(all_data[t]))

            # Generate inverse triplets
            inverse_train_data = train_new_data[:, [2, 1, 0]]
            inverse_train_data[:, 1] = inverse_train_data[:, 1] + rel_nums
            train_new_data = torch.cat([train_new_data, inverse_train_data])

            t //= time_interval
            date = self.convert_to_date(t)

            ent_ent_triplets_t, ent_rel_triplets_t = [], []
            ent_ent_update, ent_rel_update = defaultdict(list), defaultdict(list)
            ent_ent_triplets_t_id, ent_rel_triplets_t_id = [], []
            ent_ent_update_id, ent_rel_update_id = defaultdict(list), defaultdict(list)

            # **新增：用于本 time step 下记录 (s_id,o_id)/(s_id,r_id) 对应列表**
            ent_ent_pairs_t = []
            ent_rel_pairs_t = []

            s_ids = train_new_data[:, 0].unique().tolist()
            unseen_entity_ids = set(s_ids) - seen_entity_ids

            # Process each triplet in the current batch
            for k, (s_id, r_id, o_id) in enumerate(train_new_data.tolist()):
                if r_id < rel_nums:
                    # 正向三元组 (s,r,o)
                    s, r, o = id2ent[s_id], id2rel[r_id], id2ent[o_id]
                    sentence_e_e = f"{s} ? {o} on {date}"
                    sentence_e_r = f"{s} {r} ? on {date}"

                    ent_ent_triplets_t.append(ent_ent_triplets.get((s, o), []) + [sentence_e_e])
                    ent_ent_update[(s, o)].append(f"{s} {r} {o} on {date}")
                    ent_rel_triplets_t.append(ent_rel_triplets.get((s, r), []) + [sentence_e_r])
                    ent_rel_update[(s, r)].append(f"{s} {r} {o} on {date}")

                    # **记录这一行历史 embedding 对应的 (s_id, o_id) 和 (s_id, r_id)**
                    ent_ent_pairs_t.append((s_id, o_id))
                    ent_rel_pairs_t.append((s_id, r_id))
                else:
                    # 逆向三元组 (o, r', s)，r' = r_id - rel_nums
                    s, r, o = id2ent[s_id], id2rel[r_id - rel_nums], id2ent[o_id]
                    sentence_e_e_inv = f"{o} ? {s} on {date}"
                    sentence_e_r_inv = f"? {r} {s} on {date}"

                    ent_ent_triplets_t.append(ent_ent_triplets.get((s, o), []) + [sentence_e_e_inv])
                    ent_ent_update[(s, o)].append(f"{o} {r} {s} on {date}")
                    ent_rel_triplets_t.append(ent_rel_triplets.get((s, r), []) + [sentence_e_r_inv])
                    ent_rel_update[(s, r)].append(f"{o} {r} {s} on {date}")

                    ent_ent_pairs_t.append((s_id, o_id))
                    ent_rel_pairs_t.append((s_id, r_id))

                # store the id lists
                ent_ent_triplets_t_id.append(ent_ent_triplets_id.get((s_id, o_id), []))
                ent_ent_update_id[(s_id, o_id)].append(r_id)
                ent_rel_triplets_t_id.append(ent_rel_triplets_id.get((s_id, r_id), []))
                ent_rel_update_id[(s_id, r_id)].append(o_id)
                
                if s_id in unseen_entity_ids:
                    new_entity_ids[t].append(k)
            
            # Save historical triplets and ID lists for this time step
            ent_ent_his_triplets[idx]    = copy.deepcopy(ent_ent_triplets_t)
            ent_rel_his_triplets[idx]    = copy.deepcopy(ent_rel_triplets_t)
            ent_ent_his_triplets_id[idx] = copy.deepcopy(ent_ent_triplets_t_id)
            ent_rel_his_triplets_id[idx] = copy.deepcopy(ent_rel_triplets_t_id)

            # **同步保存 (s_id,o_id) 和 (s_id,r_id) 对应列表**
            ent_ent_pair_list[idx] = ent_ent_pairs_t[:]
            ent_rel_pair_list[idx] = ent_rel_pairs_t[:]

            # Update historical triplets for future steps
            self._update_historical_triplets(ent_ent_triplets, ent_ent_update)
            self._update_historical_triplets(ent_rel_triplets, ent_rel_update)
            self._update_triplet_ids(ent_ent_triplets_id, ent_ent_update_id)
            self._update_triplet_ids(ent_rel_triplets_id, ent_rel_update_id)

        # 返回时包含两套“pair list”
        return (
            ent_ent_his_triplets, ent_rel_his_triplets,
            ent_ent_his_triplets_id, ent_rel_his_triplets_id,
            ent_ent_pair_list, ent_rel_pair_list,
            new_entity_ids
        )

    def _update_historical_triplets(self, triplets, updates):
        """
        Update historical triplets with new sentences, ensuring no more than `num_k` triplets are stored.
        """
        for key, sentences in updates.items():
            triplets[key].extend(sentences)
            while len(triplets[key]) > self.num_k:
                triplets[key].pop(0)

    def _update_triplet_ids(self, triplets_id, updates_id):
        """
        Update triplet IDs, ensuring no duplicate IDs are stored.
        """
        for key, ids in updates_id.items():
            for r_id in ids:
                if r_id not in triplets_id[key]:
                    triplets_id[key].append(r_id)
        
    def get_local_entity(self, his_triplets, current_triplets):
        """
        Get the local entities connected to the current triplet, based on historical triplets.
        """
        local_entities = []
        his_entities = defaultdict(set)

        # Create a mapping from subject to object and relation
        for his_triplet in his_triplets:
            his_triplet = torch.from_numpy(his_triplet)
            inverse_his_triplet = his_triplet[:, [2, 1, 0, 3]]
            all_triplets = torch.cat([his_triplet, inverse_his_triplet], dim=0)
            
            for triplet in all_triplets:
                s1, r1, o1 = triplet[:3].tolist()
                his_entities[s1].add((o1, r1))
            
        # Find local entities related to the current triplet
        for triplet in current_triplets:
            s, r = triplet[:2].tolist()
            query = set()

            if s in his_entities:
                for o1, r1 in his_entities[s]:
                    query.add(o1)
                    if r1 == r:
                        query.update(o2 for o2, _ in his_entities[o1])

            local_entities.append(query)
        
        return local_entities
    
    def convert_to_date(self, number):
        """
        Convert the number to date string. For example, 0 -> '2014-01-01'.
        """
        if self.dataset in ['GDELT', 'ICEWS18']:
            self.year = 2018
        elif self.dataset in ['ICEWS05-15']:
            self.year = 2005
        elif self.dataset in ['ICEWS14s', 'ICEWS14']:
            self.year = 2014
        else:
            raise ValueError('Unsupported dataset')
        
        base_date = datetime(self.year, 1, 1)
        delta = timedelta(days=number)
        target_date = base_date + delta
        return target_date.strftime("%Y-%m-%d")


class DSEncoder(nn.Module):
    _SHARD_SIZE = 512  # tensors per shard; header stays well below 16 MiB

    def __init__(
        self,
        paths: Dict[str, str],
        plm: str = "llama",
        model_type: str = "llama",
        batch_size: int = 32,
        gpu: int = -1,
        save: bool = False,
    ):
        super().__init__()
        self.device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.save = save
        self.model_type = model_type

        # paths
        self.path = paths["path"]
        self.entities_emb_path = paths["entities_emb_path"]
        self.relations_emb_path = paths["relations_emb_path"]
        self.ent_ent_his_emb_path = paths["ent_ent_his_emb_path"]  # *.safetensors (base)
        self.ent_rel_his_emb_path = paths["ent_rel_his_emb_path"]

        # legacy *.pt paths (for auto‑migration)
        self.ent_ent_legacy_pt = self.ent_ent_his_emb_path.replace(".safetensors", ".pt")
        self.ent_rel_legacy_pt = self.ent_rel_his_emb_path.replace(".safetensors", ".pt")

        self._load_pretrained_model()  # Load the specified pre-trained model


    def _load_pretrained_model(self):
        """
        Load the specified pre-trained model (BERT, T5, GPT-2, LLaMA, etc.) and its tokenizer.
        """
        model_path = r'/xxx/xxx/work/XXX/tools/llama-3b'  # your large model path
       

        print(f'\nLoading pretrained model from {model_path}...\n')

        if self.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token for LLaMA
            self.model = LlamaModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map={'': str(self.device)}
            )
            self.hidden_size = self.model.config.hidden_size
        else:
            raise ValueError('Unsupported model type')


    def _get_sentence_embedding(self, sentences):
        """
        Generate sentence embeddings using the pre-trained model.
        """
        with torch.no_grad():
            inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            # 对 LLaMA，使用 last_hidden_state 在 token 维度做平均
            return torch.mean(outputs.last_hidden_state, dim=1)


    def initial_word_embedding(self, entities=None, relations=None):
        """
        Encode and save embeddings for entities and relations.
        """
        if os.path.exists(self.entities_emb_path) and os.path.exists(self.relations_emb_path):
            print('\nLoading existing entity and relation embeddings...\n')
            entities_embedding = torch.load(self.entities_emb_path).to(self.device)
            relations_embedding = torch.load(self.relations_emb_path).to(self.device)
            return entities_embedding, relations_embedding
        
        entities_embedding = torch.zeros(len(entities), self.hidden_size).to(self.device)
        relations_embedding = torch.zeros(len(relations), self.hidden_size).to(self.device)
        print('============================')
        
        print('模型不存在....')
        print('\n' + '-'*10 + 'Encoding entities and relations' + '-'*10)
        
        for idx, entity in tqdm(enumerate(entities), desc='Encoding entities'):
            # 单一实体句子列表传入 _get_sentence_embedding 返回 [1, hidden_size]
            entities_embedding[idx] = self._get_sentence_embedding([entity]).squeeze(0)
        
        for idx, relation in tqdm(enumerate(relations), desc='Encoding relations'):
            relations_embedding[idx] = self._get_sentence_embedding([relation]).squeeze(0)
        
        if self.save:
            os.makedirs(self.path, exist_ok=True)
            torch.save(entities_embedding.cpu(), self.entities_emb_path)
            print('The encoded entities are saved in:', self.entities_emb_path)

            torch.save(relations_embedding.cpu(), self.relations_emb_path)
            print('The encoded relations are saved in:', self.relations_emb_path)

        return entities_embedding, relations_embedding


   

    def _embed_sentences_in_batches(self, sentences: List[str]) -> torch.Tensor:
        num = len(sentences)
        out = torch.zeros(num, self.hidden_size, dtype=torch.float32)
        for i in range(0, num, self.batch_size):
            emb = self._get_sentence_embedding(sentences[i: i + self.batch_size]).cpu()
            out[i: i + self.batch_size] = emb
        return out



    def encode_his_triplets(self, his_triplets):
        """
        Encode historical triplets into embeddings.
        """

        # embeddings = defaultdict(dict)
        embeddings: Dict[int, torch.Tensor] = {}

        for t, his_triplet in tqdm(his_triplets.items()):
            sentences = self._build_sentences_from_triplets(his_triplet)
            embeddings[t] = self._embed_sentences_in_batches(sentences)
        
        return embeddings


    def _build_sentences_from_triplets(self, his_triplet):
        """
        Helper function to create sentences from historical triplets.
        """
        if self.model_type in ['llama']:
            return ['. '.join(triplet) for triplet in his_triplet]
        return []


    def _migrate_pt_to_shards(self, pt_path: str, base: str, tag: str):
        print(f"[migrate] {pt_path} → {base}_shard*.safetensors")
        data = torch.load(pt_path, map_location="cpu")
        self._save_his_embeddings(data, base, tag)
        del data; gc.collect()

        print('====================')                                       # ③ 强制回收
        print("[migrate] done.")

    def _has_cached_shards(self, base: str) -> bool:
        """
        若 *.safetensors 分片文件已存在（比如 _shard0.safetensors），返回 True
        """
        return os.path.exists(self._shard_path(base, 0))

    def encode(self, ent_ent_his_triplets: dict, ent_rel_his_triplets: dict):
        print("执行 encode() – checking cached *.safetensors …")


        
        # ★ ② 判断“base 或任意分片”是否存在
        ent_ent_ready = self._has_cached_shards(self.ent_ent_his_emb_path)
        ent_rel_ready = self._has_cached_shards(self.ent_rel_his_emb_path)


        if ent_ent_ready and ent_rel_ready:
            print("发现缓存文件，使用 memory‑map 方式加载 …")
            ent_ent_his_embeddings = self._load_his_embeddings(self.ent_ent_his_emb_path)
            ent_rel_his_embeddings = self._load_his_embeddings(self.ent_rel_his_emb_path)
            return ent_ent_his_embeddings, ent_rel_his_embeddings

        print("未找到缓存 – 开始重新编码历史三元组 …")


        if os.path.exists(self.ent_ent_legacy_pt) and os.path.exists(self.ent_rel_legacy_pt):
            self._migrate_pt_to_shards(self.ent_ent_legacy_pt, self.ent_ent_his_emb_path, "ent_ent")
            self._migrate_pt_to_shards(self.ent_rel_legacy_pt, self.ent_rel_his_emb_path, "ent_rel")
            return (
                self._load_his_embeddings(self.ent_ent_his_emb_path),
                self._load_his_embeddings(self.ent_rel_his_emb_path),
            )
        
        # ent_ent_his_embeddings = self.encode_his_triplets(ent_ent_his_triplets)
        # ent_rel_his_embeddings = self.encode_his_triplets(ent_rel_his_triplets)

        if self.save:
            self._save_his_embeddings(ent_ent_his_embeddings, self.ent_ent_his_emb_path, 'ent_ent_his_embeddings')
            self._save_his_embeddings(ent_rel_his_embeddings, self.ent_rel_his_emb_path, 'ent_rel_his_embeddings')
        
        return ent_ent_his_embeddings, ent_rel_his_embeddings
    
    def _load_his_embeddings(self, base: str) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        shard = 0
        while os.path.exists(p := self._shard_path(base, shard)):
            with safe_open(p, framework="pt", device="cpu") as f:
                for k in f.keys():
                    out[int(k[1:])] = f.get_tensor(k)  # mmap
            shard += 1
        return out

    def _shard_path(self, base: str, idx: int) -> str:
        return base.replace(".safetensors", f"_shard{idx}.safetensors")

    def _save_his_embeddings(self, embeddings: Dict[int, torch.Tensor], base: str, name: str):
        os.makedirs(os.path.dirname(base), exist_ok=True)
        shard, buff = 0, {}
        for t, e in sorted(embeddings.items()):
            buff[f"t{t}"] = e.cpu().contiguous()
            if len(buff) == self._SHARD_SIZE:
                save_file(buff, self._shard_path(base, shard), metadata={"type": name})
                shard, buff = shard + 1, {}
        if buff:  # 最后一片
            save_file(buff, self._shard_path(base, shard), metadata={"type": name})



def get_historical_embeddings(dataset, num_k, plm='llama', model_type='llama', batch_size=32, gpu=-1, save=False):
    
    preprocessor = DataPreprocessor(dataset, num_k, plm)
    
    # Generate file paths and historical triplets, 现在包含 pair_list
    paths = preprocessor.get_paths()
    ent_ent_his_triplets, ent_rel_his_triplets, \
    ent_ent_his_triplets_id, ent_rel_his_triplets_id, \
    ent_ent_pair_list, ent_rel_pair_list, \
    new_entity_ids = preprocessor.generate_historical_triplets()

    id2ent, id2rel = preprocessor.get_id2ent_id2rel()
    entities, relations = list(id2ent.values()), list(id2rel.values())

    encoder = DSEncoder(paths, plm, model_type, batch_size, gpu, save)
    
    # Generate initial embeddings (these remain on GPU or CPU per torch.load / new)
    entities_embedding, relations_embedding = encoder.initial_word_embedding(entities, relations)
    
    # Generate historical embeddings (each time step moves to CPU immediately)
    ent_ent_his_embeddings, ent_rel_his_embeddings = encoder.encode(ent_ent_his_triplets, ent_rel_his_triplets)
    
    return (
        entities_embedding,
        relations_embedding,
        ent_ent_his_embeddings,
        ent_rel_his_embeddings,
        ent_ent_his_triplets_id,
        ent_rel_his_triplets_id,
        ent_ent_pair_list,
        ent_rel_pair_list,
        new_entity_ids
    )
