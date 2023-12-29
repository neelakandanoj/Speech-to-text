# impory deprndent libarary
from evaluate import load
import json
from main import *


# Open the txt file
with open('trans.txt', 'r', encoding='utf-8') as file2:
    txt_lines = file2.readlines()
Empty_data = {}

# Converting to Json
for line in txt_lines:
    line = line.strip().split('\t')
    if len(line) == 2:
        filename, content = line
        Empty_data[filename] = content

json_data = json.dumps(Empty_data, ensure_ascii=False, indent=2)

# Creating  the JSON data to a file
with open('output2.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

with open('output2.json','rb') as file3:
   data2 = json.load(file3)

if sample[0]=='/':
  s=sample.split("/")[-1]
  json_file=data2[s]
  predictions=result['text'].split()
  references=json_file.split()
  wer = load("wer")
  wer_score = wer.compute(predictions=predictions, references=references)
  print("WER Score:", wer_score)
else:
  json_file=data2[sample]
  predictions=result['text'].split()
  references=json_file.split()
  wer = load("wer")
  wer_score = wer.compute(predictions=predictions, references=references)
  print("WER Score:", wer_score)


