import os
import math
import numpy as np
import pandas as pd
import requests
import re

#extract information from variants in the format [a-z][09]*[a-z]
train = pd.read_csv("./training_variants/training_variants")
test = pd.read_csv("./test_variants/test_variants")
df_joint = pd.concat([train,test])
genes = set(df_joint["Gene"])
variations = set(df_joint["Variation"])
AA_VALID = 'ACDEFGHIKLMNPQRSTVWY'
df_joint["simple_variation_pattern"] = df_joint.Variation.str.contains(r'^[A-Z]\d{1,7}[A-Z]',case=False)
df_joint["rare_variation_pattern"] = df_joint.Variation.str.contains(r'^[A-Z]\d{1,7}[*]+',case=False)
df_joint['location_number'] = df_joint.Variation.str.extract('(\d+)')
df_joint['variant_letter_first'] = df_joint.apply(lambda row: row.Variation[0] if row.Variation[0] in (AA_VALID) else np.NaN,axis=1)
df_joint['variant_letter_last'] = df_joint.apply(lambda row: row.Variation.split()[0][-1] if (row.Variation.split()[0][-1] in (AA_VALID)) else np.NaN ,axis=1)
df_joint.loc[df_joint.simple_variation_pattern==False,['variant_letter_last',"variant_letter_first"]] = np.NaN

#get some extra information for the genes used here such as the official name and number of aminoacids
read_extra=False
try:
  read_extra=True
  genes_extra=pd.read_csv("./genes.txt", sep="\|", engine="python", names=["Gene", "Name","Size1","Size2","Size3","Size4"])
except: 
  read_extra=False
pd.set_option('display.max_colwidth', -1)
with open("genes.txt", "a") as myfile:
 for x in genes:
  if(read_extra==False or genes_extra[genes_extra.Gene==x].empty):
   print(x)
   r = requests.get('http://www.genecards.org/cgi-bin/carddisp.pl?gene='+str(x), headers={"User-Agent":"Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36"})
   index=r.text.find("aliasMainName")+15
   index_end=index+r.text[index:index+150].find("</span>")
   name=r.text[index:index_end].replace("&#39;", "")
   myfile.write(str(x)+"|"+name)
   last=0
   count=0
   for a in re.finditer("Protein attributes",r.text):
    index_start=r.text.find("Protein attributes",a.start())
    index=r.text.find("Size:",index_start)
    index=r.text.find("<dd>",index)+4
    index_end=index+r.text[index:index+100].find(" amino acids</dd>")
    size=r.text[index:index_end]
    if(last!=int(size)):
     last=int(size)
     myfile.write("|"+size)
     count=count+1
   myfile.write(("|0"*(4-count))+"\n")
   myfile.flush()
train = df_joint.loc[~df_joint.Class.isnull()]
test = df_joint.loc[df_joint.Class.isnull()]
#save the new information to an enriched variants file
with open("train_variants_extra", "a") as myfile:
 for index, row in train.iterrows():
  if(row['simple_variation_pattern']==True or row['rare_variation_pattern']==True):
   location=int(5*float(row['location_number'])/max(float(genes_extra[genes_extra.Gene==row['Gene']]['Size1'].to_string()[4:]),float(genes_extra[genes_extra.Gene==row['Gene']]['Size2'].to_string()[4:]),float(genes_extra[genes_extra.Gene==row['Gene']]['Size3'].to_string()[4:]),float(genes_extra[genes_extra.Gene==row['Gene']]['Size4'].to_string()[4:])))
   if(row['rare_variation_pattern']==True):
    letter_first=row['Variation'][0].lower()
    letter_last="na"
   else:
    nanstring=""+str(row['variant_letter_first'])
    letter_first="na" if nanstring=="nan" else row['variant_letter_first'].lower()
    nanstring=""+str(row['variant_letter_last'])
    letter_last="na" if nanstring=="nan" else row['variant_letter_last'].lower()
   myfile.write(str(int(row['Class']))+"|"+row['Gene']+" "+re.sub(r"[\(\)<>\+\.,-_:;]", "",genes_extra[genes_extra.Gene==row['Gene']]['Name'].to_string()[4:].strip().replace("/"," ").lower())+" location"+str(location)+" missense "+letter_first+"amino "+letter_last+"amino "+row['location_number']+"index\n")
  else:
   myfile.write(str(int(row['Class']))+"|"+row['Gene']+" "+re.sub(r"[\(\)<>\+\.,-_:;]", "",genes_extra[genes_extra.Gene==row['Gene']]['Name'].to_string()[4:].strip().replace("/"," ").lower())+" "+row['Variation']+"\n")
with open("test_variants_extra", "a") as myfile:
 for index, row in test.iterrows():
  if(row['simple_variation_pattern']==True or row['rare_variation_pattern']==True):
   location=int(5*float(row['location_number'])/max(float(genes_extra[genes_extra.Gene==row['Gene']]['Size1'].to_string()[4:]),float(genes_extra[genes_extra.Gene==row['Gene']]['Size2'].to_string()[4:]),float(genes_extra[genes_extra.Gene==row['Gene']]['Size3'].to_string()[4:]),float(genes_extra[genes_extra.Gene==row['Gene']]['Size4'].to_string()[4:])))
   if(row['rare_variation_pattern']==True):
    letter_first=row['Variation'][0].lower()
    letter_last="na"
   else:
    nanstring=""+str(row['variant_letter_first'])
    letter_first="na" if nanstring=="nan" else row['variant_letter_first'].lower()
    nanstring=""+str(row['variant_letter_last'])
    letter_last="na" if nanstring=="nan" else row['variant_letter_last'].lower()
   myfile.write(row['Gene'].lower()+" "+re.sub(r"[\(\)<>\+\.,-_:;]", "",genes_extra[genes_extra.Gene==row['Gene']]['Name'].to_string()[4:].strip().replace("/"," ").lower())+" location"+str(location)+" missense "+letter_first+"amino "+letter_last+"amino "+row['location_number']+"index\n")
  else:
   myfile.write(row['Gene'].lower()+" "+re.sub(r"[\(\)<>\+\.,-_:;]", "",genes_extra[genes_extra.Gene==row['Gene']]['Name'].to_string()[4:].strip().replace("/"," ").lower())+" "+row['Variation']+"\n")