import numpy as np
import pandas as pd
from tqdm import tqdm
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer, load_vocab
from keras.layers import *
from keras import backend as K
from bert4keras.snippets import parallel_apply
from keras.optimizers import Adam
import keras
import math

def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


class DataGenerator(keras.utils.Sequence):

    # 对于所有数据输入，每个 epoch 取 dataSize 个数据
    # data 为 pandas iterator
    def __init__(self, data_path  ,batch_size=8):
        print("init")
        self.data_path = data_path
        data = pd.read_csv(data_path,
                           sep = '\t',
                           header=None,
                           )
        self.batch_size = batch_size
        self.dataItor = data
        self.data = data.dropna().sample(frac=1)

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.floor(len(self.data) / (self.batch_size))-1

    def __getitem__(self, index):
        # 生成每个batch数据
        batch = self.data[index*self.batch_size:(index+1)*self.batch_size]

        # 生成数据
        x,y = self.data_generation(batch,index,len(self.data))
        return [x,y], None

    def on_epoch_end(self):
        #在每一次epoch结束进行一次随机
        self.data = self.data.sample(frac=1)

    def data_generation(self, batch,index,lenth):
        batch_x = []
        batch_y = []
        for a,b in batch.iterrows():
            content_len = len(b[1])
            title_len = len(b[0])
            if(content_len + title_len > max_input_len):
                content = b[1][:max_input_len - title_len]
            else:
                content = b[1]
            x, s = tokenizer.encode(content, b[0])
            batch_x.append(x)
            batch_y.append(s)
        return padding(batch_x),padding(batch_y)



def get_model(config_path,checkpoint_path,keep_words,albert=False,lr = 1e-5):
    model = build_bert_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        application='seq2seq',
        keep_words=keep_words,
        albert=albert
    )


    y_in = model.input[0][:, 1:] # 目标tokens
    y_mask = model.input[1][:, 1:]
    y = model.output[:, :-1] # 预测tokens，预测与目标错开一位

    # 交叉熵作为loss，并mask掉输入部分的预测
    cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    model.add_loss(cross_entropy)
    model.compile(optimizer=Adam(lr))
    return model


def just_show():
    s1 = u'为了避免现有讽刺识别方法的性能会受训练数据缺乏的影响,在使用有限标注数据训练的注意力卷积神经网络基础上,提出一种对抗学习框架,该框架包含两种互补的对抗学习方法。首先,提出一种基于对抗样本的学习方法,应用对抗生成的样本参与模型训练,以期提高分类器的鲁棒性和泛化能力。进而,研究基于领域迁移的对抗学习方法,以期利用跨领域讽刺表达数据,改善模型在目标领域上的识别性能。在3个讽刺数据集上的实验结果表明,两种对抗学习方法都能提高讽刺识别的性能,其中基于领域迁移方法的性能提升更显著;同时结合两种对抗学习方法能够进一步提高讽刺识别性能。'
    s2 = u'针对现有跨领域情感分类方法中文本表示特征忽略了重要单词与句子的情感信息,且在迁移过程中存在负面迁移的问题,提出一种将文本表示学习与迁移学习算法相结合的跨领域情感分类方法。首先,利用低维稠密的词向量对文本进行初始化,通过分层注意力网络,对文本中重要单词与句子的情感信息进行建模,从而学习源领域与目标领域的文档级分布式表示。随后,采用类噪声估计方法,对源领域中的迁移数据进行检测,剔除负面迁移样例,挑选高质量样例来扩充目标领域的训练集。最后,训练支持向量机对目标领域文本进行情感分类。在大规模公开数据集上进行的两个实验结果表明,与基准方法相比,所提方法的均方根误差分别降低1.5%和1.0%,说明该方法可以有效地提高跨领域情感分类性能。'
    s3 = u'结合排序学习方法,对电影排名预测任务进行研究。通过挖掘和分析电影媒体网站数据,完成对排名预测相关特征的抽取与扩展及排名标注的对齐和划分等,并提出面向电影媒体网站的排名预测模型。实验结果显示,该模型能有效地提高电影排名预测任务的性能,在为影视院线合理规划同期电影的上映时间及排片比例、为观影者提供优质热门的电影推荐等方面具有一定的应用价值。'
    s4 = u'回顾了我国微生物浸出技术发展的历史进程,总结了我国开展生物浸铜技术的探索与应用进程,介绍了紫金山铜矿、德兴铜矿两个典型的生物浸铜案例;探讨了浸矿细菌分离、鉴定与富集,生物浸出机理与界面反应,浸出体系多级渗流行为,孔隙结构重构与定量化,浸出体系多场耦合与过程模拟,电子废弃物中的铜金属回收领域的主要进展.最后,结合生物浸铜技术的当前进展,阐述了生物浸铜技术面临的环保、安全等方面的挑战与未来发展趋势,为今后该领域的研究提供良好借鉴.'
    s5 = u'首先介绍了激光增材制造中残余应力的产生和危害,指出高的温度梯度和不均匀相变是高残余应力的原因,列举了残余应力造成的热裂纹、翘曲和疲劳失效等危害。然后,从残余应力的试验测定、数值模拟以及调控消减三个方面总结了相关研究现状。残余应力试验测定部分包括表面和内部残余应力的测试,方法有X射线衍射法、中子衍射法和压痕法等。数值模拟部分主要评述了工艺参数和扫描策略对应力场的影响。残余应力调控指的是在成形过程中,通过工艺控制减少应力的产生,主要介绍了采用热处理、超声冲击降低已成形构件残余应力的相关研究。最后提出应开展微观残余应力到大型构件宏观残余应力的多尺度表征,表面残余应力和内部残余应力相结合的多手段定量测定等专题研究,为开展残余应力与工件失效的关联性研究打下基础。'
    s6 = u'目的:定量测定与评价急性过量酒精灌胃对大鼠氧化应激的影响。方法:将成年雄性SD大鼠随机分为两组,每组10只。对照组给予大鼠灌胃纯净水;实验组按7.6 ml/kg给大鼠灌胃50%酒精,每天2次,连续3d,制作酒精急性氧化损伤模型。测定大鼠血液中丙二醛(malondialdehyde,MDA)的含量,总抗氧化力(total antioxidant force,T-AOC)的改变和超氧化物歧化酶(superoxide dismutase,T-SOD)、过氧化氢酶(catalase,CAT)、谷胱甘肽硫转移酶(glutathione sulfur transferase,GST)等活性的变化。结果:与对照组相比,实验组大鼠血清中MDA含量明显升高(P&lt;0.05);而T-SOD活性无明显变化(P&gt;0.05),但CAT、GPx、GST活性和T-AOC活力均明显升高(P&lt;0.05)。结论:过量酒精的摄入可以引起机体氧化应激,表现为脂质过氧化产物增多;抗氧化酶活性诱导性升高,提示只要不损伤机体抗氧化代偿机制,机体仍能够维持氧化-抗氧化体系的平衡。'
    for s in [s1, s2, s3, s4, s5, s6]:
        print(u'生成标题:', gen_sent(s))


def gen_sent(s, topk=2):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    token_ids, segment_ids = tokenizer.encode(s[:max_input_len])
    target_ids = [[] for _ in range(topk)]  # 候选答案id
    target_scores = [0] * topk  # 候选答案分数
    for i in range(max_output_len):  # 强制要求输出不超过max_output_len字
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        _probas = model.predict([_target_ids, _segment_ids
                                 ])[:, -1, 3:]  # 直接忽略[PAD], [UNK], [CLS]
        _log_probas = np.log(_probas + 1e-6)  # 取对数，方便计算
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            # 预测第一个字的时候，输入的topk事实上都是同一个，
            # 所以只需要看第一个，不需要遍历后面的。
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]  # 从中选出新的topk
        target_ids = [_candidate_ids[k] for k in _topk_arg]
        target_scores = [_candidate_scores[k] for k in _topk_arg]
        best_one = np.argmax(target_scores)
        if target_ids[best_one][-1] == 3:
            return tokenizer.decode(target_ids[best_one])
    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])

class Evaluate(keras.callbacks.Callback):
    def __init__(self, val_data_path, topk):
        self.topk = topk

    def on_epoch_end(self, epoch, logs=None):
        just_show(self.topk)



config_path = 'bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'bert/chinese_L-12_H-768_A-12/vocab.txt'


min_count = 0
max_input_len = 256
max_output_len = 32
batch_size = 8
epochs = 10
topk = 2

train_data_path = 'train.tsv'

_token_dict = load_vocab(dict_path)  # 读取词典
_tokenizer = Tokenizer(_token_dict, do_lower_case=True)  # 建立临时分词器


def read_texts():
    txts = [train_data_path]
    for txt in txts:
        lines = open(txt).readlines()
        for line in lines:
            d = line.split('\t')
            yield d[1][:max_input_len], d[0]

def _batch_texts():
    texts = []
    for text in read_texts():
        texts.extend(text)
        if len(texts) >= 1000:
            yield texts
            texts = []
    if texts:
        yield texts

def _tokenize_and_count(texts):
    _tokens = {}
    for text in texts:
        for token in _tokenizer.tokenize(text):
            _tokens[token] = _tokens.get(token, 0) + 1
    return _tokens

tokens = {}

def _total_count(result):
    for k, v in result.items():
        tokens[k] = tokens.get(k, 0) + v

# 词频统计
parallel_apply(
    func=_tokenize_and_count,
    iterable=tqdm(_batch_texts(), desc=u'构建词汇表中'),
    workers=10,
    max_queue_size=500,
    callback=_total_count,
)

tokens = [(i, j) for i, j in tokens.items() if j >= min_count]
tokens = sorted(tokens, key=lambda t: -t[1])
tokens = [t[0] for t in tokens]

token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t in tokens:
    if t in _token_dict and t not in token_dict:
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict, do_lower_case=True)  # 建立分词器

model = get_model(config_path, checkpoint_path, keep_words, args.albert, args.lr)

evaluator = Evaluate(topk)

model.fit_generator(
    DataGenerator(train_data_path,batch_size),
    epochs=epochs,
    callbacks=[evaluator]
)


