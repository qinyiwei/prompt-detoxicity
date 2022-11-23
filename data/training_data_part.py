import random

dir = '/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/datasets/openwebtext/'
#input_files = ['toxicity_gte99.txt', 'toxicity_lte2.txt']
#input_files = ['positivity_gte99.txt', 'negativity_gte99.txt']
#dir = '/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/datasets/jigsaw-unintended-bias-in-toxicity-classification/'
#input_files = ['toxicity_gte0.5.txt']
#input_files = ['safe_all.txt']
input_files = ['negativity_gte99.txt']

rate = 5
#NUM = 10000
#NUM = 5000
#NUM = 2000
#NUM = 1000
#NUM = 597754

for input_file in input_files:
    f = open(dir+input_file,'r')
    DATA_ALL = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    print(len(DATA_ALL))
    NUM = int(rate/100*len(DATA_ALL))

    random.seed(0)
    #mini_indices = random.sample(list(range(len(DATA_ALL))), NUM)
    #mini_indices.sort()
    #DATA_part = [DATA_ALL[index] for index in mini_indices]
    DATA_part = DATA_ALL
    #output_file = input_file.replace('.txt','_num_{}.txt'.format(NUM))
    #output_file = "safe_whole_all.txt"
    #output_file ='safe_num_{}.txt'.format(rate)
    output_file = "negativity_gte99_whole.txt"
    with open(dir+output_file,'w') as fo:
        for line in DATA_part:
            fo.write(line+'\n')
    