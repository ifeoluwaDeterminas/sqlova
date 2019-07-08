#!/usr/bin/env python

# Use existing model to predict sql from tables and questions.
#
# For example, you can get a pretrained model from https://github.com/naver/sqlova/releases:
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_bert_best.pt
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_best.pt
#
# Make sure you also have the following support files (see README for where to get them):
#    - bert_config_uncased_*.json
#    - vocab_uncased_*.txt
#
# Finally, you need some data - some files called:
#    - <split>.db
#    - <split>.jsonl
#    - <split>.tables.jsonl
#    - <split>_tok.jsonl         # derived using annotate_ws.py
# You can play with the existing train/dev/test splits, or make your own with
# the add_csv.py and add_question.py utilities.
#
# Once you have all that, you are ready to predict, using:
#   python predict.py \
#     --bert_type_abb uL \       # need to match the architecture of the model you are using
#     --model_file <path to models>/model_best.pt            \
#     --bert_model_file <path to models>/model_bert_best.pt  \
#     --bert_path <path to bert_config/vocab>  \
#     --result_path <where to place results>                 \
#     --data_path <path to db/jsonl/tables.jsonl>            \
#     --split <split>
#
# Results will be in a file called results_<split>.jsonl in the result_path.

import argparse, os
from sqlnet.dbengine import DBEngine
from sqlova.utils.utils_wikisql import *
from train import construct_hyper_param, get_models

# This is a stripped down version of the test() method in train.py - identical, except:
#   - does not attempt to measure accuracy and indeed does not expect the data to be labelled.
#   - saves plain text sql queries.
#
def predict(data_loader, data_table, model, model_bert, bert_config, tokenizer,
            max_seq_length,
            num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
            path_db=None, dset_name='test'):

    model.eval()
    model_bert.eval()

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):
        """
        (Pdb) pr_sql_i
        [{'agg': 0, 'sel': 2, 'conds': [[0, 0, 'terrence ross']]}, {'agg': 0, 'sel': 2, 'conds': [[4, 0, '1995-96']]}]

        (Pdb) pr_sql_q
        ['SELECT (Nationality) FROM 1-10015132-16 WHERE Player = terrence ross', 'SELECT (Nationality) FROM 1-10015132-16 WHERE Years in Toronto = 1995-96']
    
        (Pdb) nlu_t
        [['What', 'is', 'terrence', 'ross', "'", 'nationality'], ['What', 'clu', 'was', 'in', 'toronto', '1995-96']]

        (Pdb) hds
        [['Player', 'No.', 'Nationality', 'Position', 'Years in Toronto', 'School/Club Team'], ['Player', 'No.', 'Nationality', 'Position', 'Years in Toronto', 'School/Club Team']]

        (Pdb) nlu
        ["What is terrence ross' nationality", 'What clu was in toronto 1995-96']

        (Pdb) t[0]
        {'table_id': '1-10015132-16', 'phase': 1, 'question': "What is terrence ross' nationality", 'question_tok': ['What', 'is', 'terrence', 'ross', "'", 'nationality'], 'sql': {'sel': 2, 'conds': [[0, 0, 'Terrence Ross']], 'agg': 0}, 'query': {'sel': 2, 'conds': [[0, 0, 'Terrence Ross']], 'agg': 0}, 'wvi_corenlp': [[2, 3]]}

        (Pdb) tb[0]
        {'header': ['Player', 'No.', 'Nationality', 'Position', 'Years in Toronto', 'School/Club Team'], 'page_title': 'Toronto Raptors all-time roster', 'types': ['text', 'text', 'text', 'text', 'text', 'text'], 'id': '1-10015132-16', 'section_title': 'R', 'caption': 'R', 'rows': [['Aleksandar Radojević', '25', 'Serbia', 'Center', '1999-2000', 'Barton CC (KS)'], ['Shawn Respert', '31', 'United States', 'Guard', '1997-98', 'Michigan State'], ['Quentin Richardson', 'N/A', 'United States', 'Forward', '2013-present', 'DePaul'], ['Alvin Robertson', '7, 21', 'United States', 'Guard', '1995-96', 'Arkansas'], ['Carlos Rogers', '33, 34', 'United States', 'Forward-Center', '1995-98', 'Tennessee State'], ['Roy Rogers', '9', 'United States', 'Forward', '1998', 'Alabama'], ['Jalen Rose', '5', 'United States', 'Guard-Forward', '2003-06', 'Michigan'], ['Terrence Ross', '31', 'United States', 'Guard', '2012-present', 'Washington']], 'name': 'table_10015132_16'}
      
        """
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)
        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs)
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            #prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward_sqlmax(wemb_n, l_n, wemb_h, l_hpu, l_hs, engine, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu, beam_size=beam_size)
            select, wcn, decoded_where = model.beam_forward_sqlmax(wemb_n, l_n, wemb_h, l_hpu, l_hs, engine, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu, beam_size=beam_size)
            # sort and generate
            #pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
            print ("select:", select[0])
            print ("\n")
            print ("wcn:", wcn[0])
            print ("\n")
            print ("decoded_where:", decoded_where) 
            return select[0], wcn[0], decoded_where

        pr_sql_q = generate_sql_q(pr_sql_i, tb)

        for b, (pr_sql_i1, pr_sql_q1) in enumerate(zip(pr_sql_i, pr_sql_q)):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results1["sql"] = pr_sql_q1
            results.append(results1)
        #import pdb;pdb.set_trace()
    return results



# Use for later. 
def decode(tokenizer, token, after, where, beam_decode=False):
    """ decode actual values in condition logits """
    gloss = token
    decoder = gloss_decoder(tokenizer, gloss, after)
    query_conds = where
    decoded_conds = []
    if beam_decode:
        decoded_conds = copy.copy(query_conds)
        for i, cols in enumerate(query_conds):
            for j, ops in enumerate(cols):
                for k, values in enumerate(ops):
                    decoded_conds[i][j][k] = values[:-2] + [decoder.decode_span(values[-2:])]
    else:
        for col,op,sub_span in query_conds:
            decoded_conds.append([col, op, decoder.decode_span(sub_span)])
    return decoded_conds

class gloss_decoder:
    def __init__(self, tokenizer, gloss, after):
        self.tokenizer = tokenizer
        self.gloss = gloss
        self.after = after
        pos_map={}
        start_index=1
        sub_tokens = []
        for j,t in enumerate(gloss):
            tok = tokenizer.tokenize(t)
            sub_tokens.extend(tok)
            l = len(tok)
            for k in range(start_index, start_index + l):
                pos_map[k]=j
            start_index += l
        self.pos_map = pos_map

    def decode_span(self, sub_span):
        if any(s < 0 for s in sub_span) or any (s > len(self.pos_map) for s in sub_span):
            return None
        span = [self.pos_map[int(s)] for s in sub_span]
        value = self.gloss[span[0]]
        for i in range(span[0]+1, span[1]+1):
            value += self.after[i-1] + self.gloss[i]
        return value


## Set up hyper parameters and paths
parser = argparse.ArgumentParser()
parser.add_argument("--model_file", required=True, help='model file to use (e.g. model_best.pt)')
parser.add_argument("--bert_model_file", required=True, help='bert model file to use (e.g. model_bert_best.pt)')
parser.add_argument("--bert_path", required=True, help='path to bert files (bert_config*.json etc)')
parser.add_argument("--data_path", required=True, help='path to *.jsonl and *.db files')
parser.add_argument("--split", required=True, help='prefix of jsonl and db files (e.g. dev)')
parser.add_argument("--result_path", required=True, help='directory in which to place results')
args = construct_hyper_param(parser)

BERT_PT_PATH = args.bert_path
path_save_for_evaluation = args.result_path

# Load pre-trained models
path_model_bert = args.bert_model_file
path_model = args.model_file
args.no_pretraining = True  # counterintuitive, but avoids loading unused models
model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)

# Load data
dev_data, dev_table = load_wikisql_data(args.data_path, mode=args.split, toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
dev_loader = torch.utils.data.DataLoader(
    batch_size=args.bS,
    dataset=dev_data,
    shuffle=False,
    num_workers=1,
    collate_fn=lambda x: x  # now dictionary values are not merged!
)

# Run prediction
with torch.no_grad():
    results = predict(dev_loader,
                      dev_table,
                      model,
                      model_bert,
                      bert_config,
                      tokenizer,
                      args.max_seq_length,
                      args.num_target_layers,
                      detail=False,
                      path_db=args.data_path,
                      st_pos=0,
                      dset_name=args.split, EG=args.EG)

# Save results
#save_for_evaluation(path_save_for_evaluation, results, args.split)


