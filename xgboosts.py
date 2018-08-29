# -*-coding=UTF-8-*-  
import jieba
import jieba.posseg as pseg
import numpy as np
import xgboost as xgb
import sys

def readtrain(path, posd):
    labels = []
    contents = []
    ids = []
    weights = []
    with open(path, 'r') as f:
        for line in f:
            line = line[1:-2]
            tagwithv = line.split('\", \"')
            label = bytes(tagwithv[0].split('\": \"')[1].rstrip(' \"'), 'latin1').decode('unicode-escape')
            content = bytes(tagwithv[1].split('\": \"')[1].rstrip(' \"\\'), 'latin1').decode('unicode-escape')
            id = int(tagwithv[2].split(':')[1].strip())
            labels.append(label)
            contents.append(content)
            ids.append(id)
            weights.append(getweights(content, posd))
            if id % 1000 == 999:
            	print('Now %d articles have been processed!'%id)
            	#break
        print('Totally there are %d articles in training set' % len(labels))
    trains =(labels,contents,ids, weights)
    return trains

def getweights(content, posd):
	weights = []
	weights.append(sys.getsizeof(content)) # 0. number of bytes
	weights.append(len(content)) # 1. number of characters
	words = pseg.cut(content)
	wnum = 0 # 2. number of words
	engnum = 0 # 3. number of english words
	digitnum = 0 # 4. number of digital words
	chnum = 0 # 5. number of chinese words
	posn = [0 for i in range(0,50)]
	for w in words:
		wnum += 1
		#print(w.word + " " + w.flag)
		if is_chinese(w.word[0]):
			chnum += 1
		if is_alphabet(w.word[0]):
			engnum += 1
		if is_number(w.word[0]):
			digitnum += 1
		if w.flag.lower() in posd:
			posn[posd[w.flag.lower()]] += 1
		elif w.flag.lower()[0] in posd:
			posn[posd[w.flag.lower()[0]]] += 1
	weights.append(wnum)
	weights.append(engnum)
	weights.append(digitnum)
	weights.append(chnum)
	weights = weights + posn # 6-*: number of pos
	return weights

# 判断一个unicode是否是汉字
def is_chinese(uchar):         
    if '\u4e00' <= uchar<='\u9fff':
        return True
    else:
        return False
 
# 判断一个unicode是否是数字
def is_number(uchar):
    if '\u0030' <= uchar<='\u0039':
        return True
    else:
        return False
 
# 判断一个unicode是否是英文字母
def is_alphabet(uchar):         
    if ('\u0041' <= uchar<='\u005a') or ('\u0061' <= uchar<='\u007a'):
        return True
    else:
        return False

def readvalidation(path, posd):
    contents = []
    ids = []
    weights = []
    with open(path, 'r') as f:
        for line in f:
            line = line[1:-2]
            tagwithv = line.split(', \"')
            #print(tagwithv[1].split('\": \"')[1].rstrip(' \"\\'))
            content = bytes(tagwithv[1].split('\": \"')[1].rstrip(' \"\\'), 'latin1').decode('unicode-escape')
            id = int(tagwithv[0].split('\": ')[1].strip())
            contents.append(content)
            ids.append(id)
            weights.append(getweights(content, posd))
            if id % 1000 == 999:
            	print('Now validation number %d articles have been processed!'%id)
            	#break
    validations =(ids,contents, weights)
    print('Totally there are %d articles in validation set' % len(ids))
    return validations

def getutferror(contents):
    utf = []
    with open('D:\\src\\git\\text_classification_AI100\\train.txt', 'w', encoding='utf-8') as f:
        for x in contents:
            try:
                f.write(x)
                utf.append(0)
            except UnicodeEncodeError:
                #f.write(x.replace('\ud83d', ''))
                utf.append(1)
    return utf

def outputweights(path, weights):
	with open(path, 'w', encoding='utf-8') as f:
		for xweights in weights:
			for yweight in xweights:
				f.write(str(yweight))
				f.write(" ")
			f.write("\n")

def outputlabels(path, labels):
	with open(path, 'w', encoding='utf-8') as f:
		for label in labels:
			f.write(str(label))
			f.write("\n")

def readweights(path):
    weights = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip() # ignore last space
            #print(line)
            weigts = line.split(' ')
            weight = []
            for w in weigts:
            	weight.append(int(w))
            if weight[1] == 0:
            	print(str(len(weights)))
            weight.append(0 if weight[1] == 0 else weight[0]*1.0/weight[1]) # average character length
            weight.append(0 if weight[2] == 0 else weight[1]*1.0/weight[2]) # average word length
            weight.append(0 if weight[2] == 0 else weight[3]*1.0/weight[2]) # english word percentage
            weight.append(0 if weight[2] == 0 else weight[4]*1.0/weight[2]) # digital word percentage
            weight.append(0 if weight[2] == 0 else weight[5]*1.0/weight[2]) # chinese word percentage
            weight.append(0 if weight[2] == 0 else (weight[6+17]+weight[24]+weight[25]+weight[26]+weight[27]+weight[28]+weight[49]+weight[51])*1.0/weight[2])# percentage of n*
            weight.append(0 if weight[2] == 0 else (weight[6+31]+weight[38]+weight[39]+weight[40])*1.0/weight[2])# percentage of v*
            weight.append(0 if weight[2] == 0 else (weight[6+0]+weight[7]+weight[8]+weight[9])*1.0/weight[2])# percentage of a*
            weights.append(weight)
        print('Totally there are %d articles in weights set' % len(weights))
    return weights

def readlabels(path):
	labels = []
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			labels.append(int(line))
	return labels
'''
# class label
dict = {u'人类作者': 0, u'机器作者': 1, u'自动摘要': 2, u'机器翻译': 3}
# jieba fenci pos tag
posd = {'ag':0, 'a': 1, 'ad':2, 'an':3, 'b':4, 'c':5, 'dg':6, 'd':7, 'e':8, 'f':9, 'g':10, 'h':11, 'i':12, 'j':13, 'k':14, 'l':15, 'm':16, 'ng':17, 'n':18, 'nr':19, 'ns':20, 'nt':21, 'nz':22, 'o':23, 'p':24, 'q':25, 'r':26, 's':27, 'tg': 28, 't':29, 'u':30, 'vg':31, 'v':32, 'vd':33, 'vn':34, 'w':35, 'x':36, 'y':37, 'z':38, 'un':39, 'uj':40, 'eng':41, 'ul':42, 'nrfg':43, 'zg':44, 'nrt':45, 'uv':46, 'ud':47, 'mq':48, 'df':49}
# read train data and extract the label and weights
train = readtrain('D:\\src\\git\\text_classification_AI100\\data\\training.txt', posd)
outputweights('D:\\src\\git\\text_classification_AI100\\data\\train_weights.txt', train[3])
outputlabels('D:\\src\\git\\text_classification_AI100\\data\\train_labels.txt', list(map(lambda x: dict[x], train[0])))
print('train data load finished %d' % len(train[0]))
# read validation data and extract weights
test = readvalidation('D:\\src\\git\\text_classification_AI100\\data\\validation.txt', posd)
outputweights('D:\\src\\git\\text_classification_AI100\\data\\validation_weights.txt', test[2])
print('validation data load finished')
'''
# read weights and labels
train_opinion = np.array(readlabels('D:\\src\\git\\text_classification_AI100\\data\\train_labels.txt'))
train_weights = np.array(readweights('D:\\src\\git\\text_classification_AI100\\data\\train_weights.txt'))
test_weights = np.array(readweights('D:\\src\\git\\text_classification_AI100\\data\\validation_weights.txt'))
# load train and validation weights
dtrain = xgb.DMatrix(train_weights, label=train_opinion)
dtest = xgb.DMatrix(test_weights)  # label可以不要，此处需要是为了测试效果
param = {'max_depth':7, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':4}  # 参数
evallist  = [(dtrain,'train')]  # 这步可以不要，用于测试效果
num_round = 300  # 循环次数
bst = xgb.train(param, dtrain, num_round, evallist)
preds = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
with open('D:\\src\\git\\text_classification_AI100\\XGBOOST_Toutiao_OUTPUT.csv', 'w', encoding='utf-8') as f:
    for i, pre in enumerate(preds):
        f.write(str(i + 146421))
        f.write(',')
        f.write(list(dict.keys())[int(pre)])
        f.write('\n')
