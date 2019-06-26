#!/usr/bin/env python

# How to run:
# CUDA_VISIBLE_DEVICES=2  python inference_api_v1.py  --bert_type_abb uS   --model_file /home/aerin/Desktop/sqlova/model_best.pt   --bert_model_file  /home/aerin/Desktop/sqlova/model_bert_best.pt  --bert_path /home/aerin/Desktop/sqlova/data/wikisql_tok2/

# BERT Files you need:
#    - bert_config_uncased_*.json
#    - vocab_uncased_*.txt

import argparse, os
from sqlnet.dbengine import DBEngine
from sqlova.utils.utils_wikisql import *
from train import construct_hyper_param, get_models



def predict(data, model, model_bert, bert_config, tokenizer, max_seq_length, num_target_layers ):
    model.eval()
    model_bert.eval()

    #t = [data['rows']]
    g_wvi = [data['data_ix']]
    hds = [data['header']]
    types = [data['types']]
    nlu = [data['question']]
    nlu_t = [nlu[0].rstrip().replace('?', '').split(' ')] 
    n = data['n']
    #tb = 
    
    wemb_n, wemb_h, l_n, l_hpu, l_hs, nlu_tt, t_to_tt_idx, tt_to_t_idx = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length, num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

    # No Execution guided decoding
    s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs, g_wvi=g_wvi)
    
    # normalize the score
    s_sc = F.softmax(s_sc, dim=1)
    s_wc = F.softmax(s_sc, dim=1)

    pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se_api_topN(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, n)

    pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

    pr_sql_i = generate_sql_i_api_topN(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu, n)
    #pr_sql_q = generate_sql_q(pr_sql_i, tb)

    return pr_sql_i






## Set up hyper parameters and paths
parser = argparse.ArgumentParser()
parser.add_argument("--model_file", required=True, help='model file to use (e.g. model_best.pt)')
parser.add_argument("--bert_model_file", required=True, help='bert model file to use (e.g. model_bert_best.pt)')
parser.add_argument("--bert_path", required=True, help='path to bert files (bert_config*.json etc)')
args = construct_hyper_param(parser)

# Load pre-trained models
BERT_PT_PATH = args.bert_path
path_model_bert = args.bert_model_file
path_model = args.model_file
args.no_pretraining = True  # counterintuitive, but avoids loading unused models
model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)


# Input format
"""
data = {"question": "What is terrence ross' nationality", "header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"], "types": ["text", "text", "text", "text", "text", "text"], "data_ix":[[2,3]], "n": 3}
data ={"question": "Give me a list of account name under Renee Lo.", "header": ["account name, account", "created on", "account id", "activate state", "revenue", "country/region", "state/povince", "city", "phone, telephone", "email address", "secondary email address", "industry", "number of employees", "description/detail", "web site, url", "contact name", "owner, owning user, account manager"], "types": ["text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "number", "text", "text", "text", "text"], "data_ix":[[8,9]], "n": 3}

"""



data = {"question": "List out the number of Accounts owned by Microsoft in USA?", "header": ["account name, account", "created on", "account id", "activate state", "revenue", "country/region", "state/povince", "city", "phone, telephone", "email address", "secondary email address", "industry", "number of employees", "description/detail", "web site, url", "contact name", "owner, owning user, account manager"], "types": ["text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "number", "text", "text", "text", "text"], "data_ix":[[8,8],[10,10]], "n": 3}

# Prediction
with torch.no_grad():
    results = predict(data, model, model_bert, bert_config, tokenizer, args.max_seq_length, args.num_target_layers) 

print ('\n\n\n Question : ', data['question'], '\n Result : ', results)












"""
# When we pass the table

data = {"question": "What is terrence ross' nationality", "header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"], "types": ["text", "text", "text", "text", "text", "text"], "rows": [["Aleksandar Radojevi\u0107", "25", "Serbia", "Center", "1999-2000", "Barton CC (KS)"], ["Shawn Respert", "31", "United States", "Guard", "1997-98", "Michigan State"], ["Quentin Richardson", "N/A", "United States", "Forward", "2013-present", "DePaul"], ["Alvin Robertson", "7, 21", "United States", "Guard", "1995-96", "Arkansas"], ["Carlos Rogers", "33, 34", "United States", "Forward-Center", "1995-98", "Tennessee State"], ["Roy Rogers", "9", "United States", "Forward", "1998", "Alabama"], ["Jalen Rose", "5", "United States", "Guard-Forward", "2003-06", "Michigan"], ["Terrence Ross", "31", "United States", "Guard", "2012-present", "Washington"]], "data_ix":[[2,3]]}
"""
