# coding: utf-8

"""
Tokenize query sentences and save information to recovery tokens to sentences.
"""
from stanfordcorenlp import StanfordCoreNLP
import ujson
from tqdm import tqdm
import os

#TODO Move to configuration
try:
  coreNlpHost = os.environ['CORENLPHOST'] if 'CORENLPHOST' in os.environ else 'http://bizqa-nl-corenlp.westus2.cloudapp.azure.com'
  coreNlpPort = int(os.environ['CORENLPPORT']) if 'CORENLPPORT' in os.environ else 80
  client = StanfordCoreNLP(coreNlpHost, port=coreNlpPort)
except:
  print("An exception occurred")

print("created StanfordCoreNLP client")

def annotate_str(text):
  annotation=client.annotate(text[0], properties={'annotators': 'ssplit,tokenize',
    'tokenize.options':'invertible,americanize=false,untokenizable=allKeep'})

  annotation=ujson.loads(annotation)
  words,gloss,after,start,end=[],[],[],[],[]
  last_end = None
  for s in annotation['sentences']:
    for t in s['tokens']:
        words.append(t['word'])
        gloss.append(t['originalText'])
        if 'characterOffsetEnd' in t:
          start.append(t['characterOffsetBegin'])
          end.append(t['characterOffsetEnd'])
        if 'after' in t:
          after.append(t['after'])
        else:
          after.append(None)
  for i,(a,s,e) in enumerate(zip(after,start,end)):
    if a is None:
      a = text[end[i]:start[i+1]] if i<len(start) else text[end[i]]
      after[i]=a
  return (words, gloss, after)

def exact_gloss_match(gloss, after, value_gloss, value_after):
  gloss = [g.lower() for g in gloss]
  v_len = len(value_gloss)
  g_len = len(gloss)
  for i in range(len(gloss)):
    if i+v_len<=len(gloss) and value_gloss==gloss[i:i+v_len] and value_after[0:v_len-1]==after[i:i+v_len-1]:
      return i,i+v_len-1
  return None,None

def gloss_match(sq_gloss, value_gloss):
  for i in range(len(sq_gloss)):
    if value_gloss[0]==sq_gloss[i].lower():
      matched = True
      for j in range(len(value_gloss)):
        if i+j>=len(sq_gloss) or value_gloss[j] != sq_gloss[i+j].lower():
          matched = False
          break
      if matched:
        return i,i+len(value_gloss)-1
  return None,None

def fuzzy_match(query, gloss, after, value):
  start_idx = query.lower().index(value)
  end_idx = start_idx + len(value) - 1
  offset = 0
  start_span = 0
  end_span = 0
  for i, (g,a) in enumerate(zip(gloss, after)):
    span = len(g)
    if start_idx>=offset and start_idx<offset+span:
      start_span = i
    if end_idx>=offset and end_idx<offset+span:
      end_span = i
    offset += span + len(a)
  return start_span, end_span

def annotate_data(input_data, output, name=None):
  name = os.path.basename(input_data) if name is None else name
  print("annotate_data:" + name)

  with open(input_data) as fs:
    data = [ujson.loads(s.strip()) for s in fs.readlines()]
    data = [data[0], data[1]]
  for line, d in tqdm(list(enumerate(data)), ncols=90, desc='Annotating {}'.format(name)):
    print(line)
  #for d in data:
    query = d['question'].strip()
    _,gloss,after=annotate_str(query)
    d['wvi_corenlp'] = None
    d['question_tok'] = gloss
    d['question_after']=after
    wvi = []
    for c in d['sql']['conds']:
      value = str(c[2]).lower().strip()
      _,value_gloss, value_after = annotate_str(value)
      start_span,end_span = exact_gloss_match(gloss, after, value_gloss, value_after)
      if start_span is None:
        start_span,end_span = gloss_match(gloss, value_gloss)
      if start_span is None:
        start_span, end_span = fuzzy_match(query, gloss, after, value)
      wvi.append([start_span, end_span])
    d['wvi_corenlp'] = wvi
    for cond,span in zip(d['sql']['conds'], d['wvi_corenlp']):
      value=gloss[span[0]]
      for i in range(span[0]+1, span[1]+1):
        value += after[i-1] + gloss[i]
      if value.lower()!=str(cond[2]).lower():
        print('\nL{}:Mismatched: {} vs {}\n'.format(line, value, cond[2]))

  with open(output, 'w') as fs:
    for d in data:
      fs.write('{}\n'.format(ujson.dumps(d)))
  return data

# input_data='./data/test.jsonl'
# output_data='test_tok_2.jsonl'

# _,gloss,after=annotate_str("What is terrence ross' nationality")
# print(_)
# print(gloss)
# print(after)

# annotated = annotate_data(input_data, output_data)
