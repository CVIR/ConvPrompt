import json
import os
from sentence_transformers import SentenceTransformer, util
import torch
import math
import numpy as np
import pickle as pkl
from transformers import RobertaTokenizer, RobertaModel

def get_dataset_name(args=None):
    if args.dataset == "Split-Imagenet-R":
        return "imr"
    elif args.dataset == "Split-CIFAR100":
        return "cifar100"
    elif args.dataset == "Split-CUB200":
        return "cub"

def getUnion(list1, list2):
    return list(set(list1) | set(list2))

def getSimilarity_RoBERTa(desc_list, desc_list2, getEmbeddings=False):
    
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    
        # print(len(desc_list), len(desc_list2))
        encodings1 = tokenizer(desc_list, return_tensors='pt', padding=True, truncation=True)
        encodings2 = tokenizer(desc_list2, return_tensors='pt', padding=True, truncation=True)
    
        with torch.no_grad():
            output1 = model(**encodings1)
            output2 = model(**encodings2)
        

        embeddings1 = output1[0][:,0,:]
        embeddings2 = output2[0][:,0,:]
    
        similarity = util.cos_sim(embeddings1, embeddings2)
        similarity, _ = torch.max(similarity, dim=1)
    
        return torch.mean(similarity)

def getSimilarity(desc_list, desc_list2, getEmbeddings=False):

    model = SentenceTransformer('whaleloops/phrase-bert')
    encodings1 = model.encode(desc_list, convert_to_tensor=False)
    encodings2 = model.encode(desc_list2, convert_to_tensor=False)

    similarity = util.cos_sim(encodings1, encodings2)
    similarity, _ = torch.max(similarity, dim=1)

    return torch.mean(similarity)

def image_embedding_similarity(class_mask, task_id, args=None):
    k = args.num_prompts_per_task
    if task_id>0 and args.variable_num_prompts:
        curr_mask = class_mask[task_id]
        print(curr_mask)

        dataset_name = args.dataset
        filename = os.getcwd()+"/"+dataset_name+"_class_prototypes.pkl"
        with open(filename, 'rb') as f:
            x = pkl.load(f)
        prev_task_embeddings = []
        for i in range(task_id):
            for class_num in class_mask[i]:
                prev_task_embeddings.append(torch.tensor(x[class_num]))
        prev_task_embeddings = torch.stack(prev_task_embeddings)

        curr_task_embeddings = []
        for class_num in class_mask[task_id]:
            curr_task_embeddings.append(torch.tensor(x[class_num]))
        curr_task_embeddings = torch.stack(curr_task_embeddings)
        
        similarity = util.cos_sim(curr_task_embeddings, prev_task_embeddings)
        similarity, _ = torch.max(similarity, dim=1)
        similarity = np.mean(np.array(similarity), axis=0)

        print('Image-based Similarity: ', similarity, ' Task: ', task_id)
        k = math.ceil((1-similarity)*k)

    return k

def class_label_similarity(class_mask, task_id, args=None):
    k = args.num_prompts_per_task
    if task_id>0 and args.variable_num_prompts:
        curr_mask = class_mask[task_id]
        print(curr_mask)

        dataset_name = get_dataset_name(args)
        path = os.getcwd()+"/descriptors/descriptors_"+dataset_name+".json"

        if not os.path.exists(path):
            print('Class List not found for ', args.dataset)
            raise NotImplementedError
        
        f = open(path)
        desc = json.load(f)

        class_labels = list(desc.keys())
        names_list1 = []

        for id in range(task_id):
            for item in class_mask[id]:
                names_list1.append(class_labels[item])

        names_list2 = []
        for item in class_mask[task_id]:
            names_list2.append(class_labels[item])

        similarity = getSimilarity(desc_list=names_list1, desc_list2=names_list2)
        print('Similarity: ', similarity, ' Task: ', task_id)

        k = math.ceil((1-similarity)*k)
    
    return k

def num_new_prompts(class_mask, task_id, args=None):
    k = args.num_prompts_per_task
    if task_id>0 and args.variable_num_prompts:
        curr_mask = class_mask[task_id]

        dataset_name = get_dataset_name(args)
        path = os.getcwd()+"/descriptors/descriptors_"+dataset_name+".json"

        if not os.path.exists(path):
            print('Descriptor List not found for ', args.dataset)
            raise NotImplementedError
        
        f = open(path)
        desc = json.load(f)

        desc_list = []

        for id in range(task_id):
            for item in class_mask[id]:
                attributes = list(desc.items())[item][1]
                # attr_list.append(attributes)
                desc_list = getUnion(desc_list, attributes)


        desc_list2 = []
        for item in class_mask[task_id]:
            attribs = list(desc.items())[item][1]
            desc_list2 = getUnion(desc_list2, attribs)


        similarity = getSimilarity(desc_list=desc_list, desc_list2=desc_list2)
        print('Similarity: ', similarity, ' Task: ', task_id)

        k = math.ceil((1-similarity)*k)
    
        
    return k