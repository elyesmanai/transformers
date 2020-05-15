from transformers import pipeline
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
from tqdm import tqdm
import pandas as pd
from datetime import datetime


fill_mask = pipeline(
    "fill-mask",
    model="google/electra-small-generator",
    tokenizer="google/electra-small-generator",
    device=-1
)
mask = fill_mask.tokenizer.mask_token


def generate(i,chunk):
  propositions = []
  for sentence in tqdm(chunk,position=0, leave=True):
    for token in sentence.split(' '):
      #print('')
      #now = datetime.now()
      masked = sentence[:sentence.index(token)] + '[MASK]' + sentence[sentence.index(token)+len(token):]
      #print(datetime.now() - now)
      now = datetime.now()
      joumal = fill_mask(masked)
      print(datetime.now() - now)
      #now = datetime.now()
      [propositions.append(x.replace('[CLS] ','').replace(' [SEP]','')) for x in joumal]
      #print(datetime.now() - now)
  print(propositions)

  


chunks = [['elyes rides his bike']]

for i,chunk in enumerate(chunks):
  generate(i,chunk)