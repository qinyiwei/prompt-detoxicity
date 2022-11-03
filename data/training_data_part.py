import random

dir = '/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/datasets/openwebtext/'
input_files = ['toxicity_gte99.txt', 'toxicity_lte2.txt']

#NUM = 10000
NUM = 5000
#NUM = 2000
#NUM = 1000

for input_file in input_files:
    f = open(dir+input_file,'r')
    DATA_ALL = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    random.seed(0)
    mini_indices = random.sample(list(range(len(DATA_ALL))), NUM)
    mini_indices.sort()
    DATA_part = [DATA_ALL[index] for index in mini_indices]
    output_file = input_file.replace('.txt','_num_{}.txt'.format(NUM))
    with open(dir+output_file,'w') as fo:
        for line in DATA_part:
            fo.write(line+'\n')
