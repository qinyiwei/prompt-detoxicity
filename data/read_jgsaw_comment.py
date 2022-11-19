import json
import csv

data_dir = "/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/datasets/jigsaw-unintended-bias-in-toxicity-classification/"
data_json = data_dir + "all_data.csv"
#data_all = json.load(open(data_json, 'r'))
#data_all = []
# for line in open(data_json, 'r'):
#    data_all.append(json.loads(line))

data_all = []
with open(data_json) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data_all.append(row)

output_safe = "safe_whole.txt"
output_toxicity = "toxicity_whole.txt"

head = data_all[0]
data_safe = []
data_toxicity = []
toxicity_index = head.index('toxicity')

for example in data_all[1:]:
    toxicity_score = float(example[toxicity_index])
    text = example[1].replace("\n", " ")
    if toxicity_score == 0:
        data_safe.append(text)
    else:
        data_toxicity.append(text)

with open(data_dir+output_safe, 'w') as fo:
    for i in data_safe:
        fo.write(i)
        fo.write("\n")

with open(data_dir+output_toxicity, 'w') as fo:
    for i in data_toxicity:
        fo.write(i)
        fo.write("\n")
