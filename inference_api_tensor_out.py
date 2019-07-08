#!/usr/bin/env python

# How to run:
# CUDA_VISIBLE_DEVICES=2  python inference_api_tensor_out.py  --bert_type_abb uS   --model_file /home/aerin/Desktop/sqlova/model_best.pt   --bert_model_file  /home/aerin/Desktop/sqlova/model_bert_best.pt  --bert_path /home/aerin/Desktop/sqlova/data/wikisql_tok2/

# BERT Files you need:
#    - bert_config_uncased_*.json
#    - vocab_uncased_*.txt

import argparse, os
from sqlova.utils.utils_wikisql import *
from train import construct_hyper_param, get_models_v2

def is_empty(any_structure):
    if any_structure:
        return False
    else:
        return True


def predict_wrapper(data):
    ## Set up hyper parameters and paths
    parser = argparse.ArgumentParser()

    if is_empty(data):
        parser.add_argument("--question", required=True, help='query')
        parser.add_argument("--header", required=True, help='column names')
        parser.add_argument("--types", required=True, help='column types')
        parser.add_argument("--data_ix", required=True, help='data index')
    args = construct_hyper_param(parser)

    args.model_file = '/home/aerin/Desktop/sqlova/model_best.pt'
    args.bert_model_file = '/home/aerin/Desktop/sqlova/model_bert_best.pt'
    args.bert_path = '/home/aerin/Desktop/sqlova/data/wikisql_tok2/'
    args.max_seq_length = 512

    # Load pre-trained models
    BERT_PT_PATH = args.bert_path
    path_model_bert = args.bert_model_file
    path_model = args.model_file
    args.no_pretraining = True  # counterintuitive, but avoids loading unused models
    model, model_bert, tokenizer, bert_config = get_models_v2(args, BERT_PT_PATH, trained=True,
                                                           path_model_bert=path_model_bert, path_model=path_model)

    # Input format
    if is_empty(data):
        data["question"] = args.question
        data["header"] = json.loads(args.header)
        data["types"] = json.loads(args.types)
        data["data_ix"] = json.loads(args.data_ix)

    # Prediction
    with torch.no_grad():
        return predict(data, model, model_bert, bert_config, tokenizer, args.max_seq_length, args.num_target_layers)


def predict(data, model, model_bert, bert_config, tokenizer, max_seq_length, num_target_layers ):
    model.eval()
    model_bert.eval()

    #t = [data['rows']]
    g_wvi = [data['data_ix']]
    hds = [data['header']]
    types = [data['types']]
    nlu = [data['question']]
    nlu_t = [nlu[0].rstrip().replace('?', '').split(' ')] 
    tb = None
    engine = None
    beam_size = 4

    wemb_n, wemb_h, l_n, l_hpu, l_hs, nlu_tt, t_to_tt_idx, tt_to_t_idx = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length, num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

    select, wcn, decoded_where = model.beam_forward_sqlmax(wemb_n, l_n, wemb_h, l_hpu, l_hs, nlu_t, nlu_tt, tt_to_t_idx, nlu, beam_size=beam_size)

    print ("select:", select[0])
    print ("\n")
    print ("wcn:", wcn[0])
    print ("\n")
    print ("decoded_where:", decoded_where) 
    return select[0], wcn[0], decoded_where

if __name__ == '__main__':
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
    model, model_bert, tokenizer, bert_config = get_models_v2(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)


    # Input format Examples
    """
    data = {"question": "What is terrence ross' nationality", "header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"], "types": ["text", "text", "text", "text", "text", "text"], "data_ix":[[2,3]], "n": 3}
    data ={"question": "Give me a list of account name under Renee Lo.", "header": ["account name, account", "created on", "account id", "activate state", "revenue", "country/region", "state/povince", "city", "phone, telephone", "email address", "secondary email address", "industry", "number of employees", "description/detail", "web site, url", "contact name", "owner, owning user, account manager"], "types": ["text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "number", "text", "text", "text", "text"], "data_ix":[[8,9]]}
    """

    data = {   
        "question": "Show the Accounts owned by Molly clark created on April 2019?",
        "header": ["account name, account", "created on", "account id", "activate state", "revenue", "country/region", "state/povince", "city", "phone, telephone", "email address", "secondary email address", "industry", "number of employees", "description/detail", "web site, url", "contact name", "owner, owning user, account manager"],
        "types": ["text", "text", "text", "text", "text", "text"],
        "data_ix":[[5,6],[9,10]]}

    # Prediction
    with torch.no_grad():
        results = predict(data, model, model_bert, bert_config, tokenizer, args.max_seq_length, args.num_target_layers) 

    print ('\n\n\n Question : ', data['question'], '\n Result : ', results)

