import os


class Config:
    root_path = os.path.abspath(os.path.dirname(__file__))

    with open(root_path + '/data/stopwords.txt', "r", encoding='utf-8') as f:
        stopwords = [word.strip() for word in f.readlines()]

    config_path = root_path + '/model/rbtl3_private/config.json'
    # bert_path = root_path + '/model/mac_bert/'
    model_name = root_path + '/model/rbtl3_private/'
    bert_path = root_path + '/model/wobert/'
    tnews_train_path = root_path + '/data/TNEWS/train.csv'
    tnews_dev_path = root_path + '/data/TNEWS/dev.csv'
    tnews_test_path = root_path + '/data/TNEWS/dev.csv'
    # tnews_train_path = root_path + '/datasets/TNEWS/train.json'
    # tnews_dev_path = root_path + '/dataseta/TNEWS/dev.json'
    # tnews_test_path = root_path + '/datasets/TNEWS/test.json'

    ocnli_train_path = root_path + '/data/OCNLI/train.csv'
    ocnli_dev_path = root_path + '/data/OCNLI/dev.csv'
    ocnli_test_path = root_path + '/data/OCNLI/dev.csv'

    # ocnli_train_path = root_path + '/datasets/OCNLI/train.json'
    # ocnli_dev_path = root_path + '/datasets/OCNLI/dev.json'
    # ocnli_test_path = root_path + '/datasets/OCNLI/test.json'

    emotion_train_path = root_path + '/data/OCEMOTION/train.csv'
    emotion_dev_path = root_path + '/data/OCEMOTION/dev.csv'
    emotion_test_path = root_path + '/data/OCEMOTION/dev.csv'

    # emotion_train_path = root_path + '/datasets/OCEMOTION/train.json'
    # emotion_dev_path = root_path + '/datasets/OCEMOTION/dev.json'
    # emotion_test_path = root_path + '/datasets/OCEMOTION/test.json'
    num_attention_heads = 12
    attention_probs_dropout_prob = 0.3
    hidden_size = 768

    news_classes = 15
    ocnli_classes = 3
    emotion_classes = 7
    num_labels = 2

    lr = 1e-4
    warmup_proportion = 0.1
    # lstm
    num_layers = 2
    bidiractional = True
    dropout_prob = 0.3

    # 字典的向量维度
    vocab_size = 50000
