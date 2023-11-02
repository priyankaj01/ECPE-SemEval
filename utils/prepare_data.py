# encoding:utf-8

import codecs
import random
import json
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb, time

def print_time():
    print '\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend( [emotion] + clause.split())
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) # 每个词及词的位置
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) # 每个词及词的位置
    word_idx
    w2v = {}
    inputFile2 = open(embedding_path, 'r')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos

def token_seq(text):
    return text.split()

def load_w2v_semeval(embedding_dim, embedding_dim_pos, data_file_path, embedding_path):
    print('\nload embedding...')
    words = []
    speakers = []
    speaker_dict = {}
    with open(data_file_path, 'r') as file:
        data = json.load(file)
    for conversation in data:
        for utteranceConv in conversation['conversation']:
            speaker = utteranceConv['speaker']
            emotion = utteranceConv['emotion']
            utterance = utteranceConv['text']
            # print(speaker)
            # print(emotion)
            # print(utterance)            
            if speaker in speaker_dict:
                speaker_dict[speaker] += 1
            else:
                speaker_dict[speaker] = 1
            speakers.append(speaker)
   
            words.extend([emotion] + token_seq(utterance))

    words = set(words)
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) 
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) 

    speaker_dict = sorted(speaker_dict.items(), key=lambda x: x[1], reverse=True)
    speakers = [item[0] for item in speaker_dict]
    spe_idx = dict((c, k + 1) for k, c in enumerate(speakers)) 
    spe_idx_rev = dict((k + 1, c) for k, c in enumerate(speakers))

    # main_speakers = ['Monica', 'Ross', 'Chandler', 'Rachel', 'Phoebe', 'Joey']
    # spe_idx = dict((c, k + 1) for k, c in enumerate(main_speakers))
    # spe_idx_rev = dict((k + 1, c) for k, c in enumerate(main_speakers))
    # print('all_speakers: {}'.format(len(spe_idx)))

    w2v = {}
    inputFile = open(embedding_path, 'r')
    emb_cnt = int(inputFile.readline().split()[0])
    for line in inputFile.readlines():
        line = line.strip().split()
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    seed = 150
    np.random.seed(seed)
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)
    print('data_file: {}\nw2v_file: {}\nall_words_emb {} all_words_file: {} hit_words: {}'.format(data_file_path, embedding_path, emb_cnt, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, spe_idx_rev, spe_idx, embedding, embedding_pos


def load_data(input_file, word_idx, max_doc_len = 75, max_sen_len = 45):
    print('load data_file: {}'.format(input_file))
    y_position, y_cause, y_pairs, x, sen_len, doc_len = [], [], [], [], [], []
    doc_id = []
    
    n_cut = 0
    inputFile = open(input_file, 'r')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        pos, cause = zip(*pairs)
        y_po, y_ca, sen_len_tmp, x_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        for i in range(d_len):
            y_po[i][int(i+1 in pos)]=1
            y_ca[i][int(i+1 in cause)]=1
            words = inputFile.readline().strip().split(',')[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])
        
        y_position.append(y_po)
        y_cause.append(y_ca)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
    
    y_position, y_cause, x, sen_len, doc_len = map(np.array, [y_position, y_cause, x, sen_len, doc_len])
    for var in ['y_position', 'y_cause', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return doc_id, y_position, y_cause, y_pairs, x, sen_len, doc_len

def load_data_2nd_step(input_file, word_idx, max_doc_len = 75, max_sen_len = 45):
    print('load data_file: {}'.format(input_file))
    pair_id_all, pair_id, y, x, sen_len, distance = [], [], [], [], [], []
    
    n_cut = 0
    inputFile = open(input_file, 'r')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        line = inputFile.readline().strip()

        # Split the line into two values
        pair_strings = line.strip(')').split('(')
        for pair in pair_strings:
            pair1 = pair.strip(',').strip(')')
            values = pair.split(',')
            if len(values) == 2:
                p0 = int(values[0])
                p1 = int(values[1])
                pair_id_all.append(doc_id * 10000 + p0 * 100 + p1)

        # pairs = eval(inputFile.readline().strip())
        # pair_id_all.extend([doc_id*10000+p[0]*100+p[1] for p in pairs])

        sen_len_tmp, x_tmp = np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        pos_list, cause_list = [], []
        for i in range(d_len):
            line = inputFile.readline().strip().split(',')
            if line[1].strip() != 'null':
                pos_list.append(i+1)
            if line[2].strip() != 'null':
                cause_list.append(i+1)
            words = line[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])
        for i in pos_list:
            for j in cause_list:
                pair_id_cur = doc_id*10000+i*100+j
                pair_id.append(pair_id_cur)
                y.append([0,1] if pair_id_cur in pair_id_all else [1,0])
                x.append([x_tmp[i-1],x_tmp[j-1]])
                sen_len.append([sen_len_tmp[i-1], sen_len_tmp[j-1]])
                distance.append(j-i+100)
    y, x, sen_len, distance = map(np.array, [y, x, sen_len, distance])
    for var in ['y', 'x', 'sen_len', 'distance']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}, (y-negative, y-positive): {}'.format(n_cut, y.sum(axis=0)))
    print('load data done!\n')
    return pair_id_all, pair_id, y, x, sen_len, distance


def load_data_semeval(input_file, word_idx, max_doc_len = 35, max_sen_len = 35):
    # print('load data_file: {}'.format(input_file))
    with open(input_file, 'r') as file:
        data = json.load(file)
      
    pair_id_all, pair_id, y, x, sen_len, distance = [], [], [], [], [], []
    sen_len_tmp, x_tmp = np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
    n_cut = 0
    num = 0
    for conversation in data:
        num += 1
        conversationID = conversation['conversation_ID']
        # extract pair_strings from emotion_cause_pairs in conversation['emotion_cause_pairs']
        for pair in conversation['emotion-cause_pairs']:
            # print(pair[0][0])
            pairid1 = int(pair[0].split('_')[0])
            pairid2 = int(pair[1].split('_')[0])
            pair_id_all.append(conversationID * 10000 + pairid1 * 100 + pairid2)
        
        if num==100:
            break
        pos_list, cause_list = [], []
        d_len = len(conversation['conversation'])
        # print(d_len)
        for conv in conversation['conversation']:
            emotion = conv['emotion']
            uttId = conv['utterance_ID']-1
            if emotion != 'neutral':
                pos_list.append(uttId)
                cause_list.append(uttId)
            # if line[2].strip() != 'null':
                # cause_list.append(i+1)
            words = conv['text']
            # print(words)
            sen_len_tmp[uttId] = min(len(words.split()), max_sen_len)
            # print(sen_len_tmp[i])
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[uttId][j] = int(word_idx[word])
        for i in pos_list:
            for j in cause_list:
                pair_id_cur = conversationID*10000+i*100+j
                pair_id.append(pair_id_cur)
                y.append([0,1] if pair_id_cur in pair_id_all else [1,0])
                x.append([x_tmp[i-1],x_tmp[j-1]])
                sen_len.append([sen_len_tmp[i-1], sen_len_tmp[j-1]])
                distance.append(j-i+100)
    
    y, x, sen_len, distance = map(np.array, [y, x, sen_len, distance])
    for var in ['y', 'x', 'sen_len', 'distance']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}, (y-negative, y-positive): {}'.format(n_cut, y.sum(axis=0)))
    print('load data done!\n')
    return pair_id_all, pair_id, y, x, sen_len, distance

def load_data_semeval_train_test_split(input_file, word_idx, max_doc_len = 35, max_sen_len = 35):
    # print('load data_file: {}'.format(input_file))
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    pair_id_all_train = []
    pair_id_all_test = []
    pair_id_train = []
    pair_id_test = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    sen_len_train = []
    sen_len_test = []
    distance_train = []
    distance_test = []
    
    
    # pair_id_all, pair_id, y, x, sen_len, distance = [], [], [], [], [], []
    sen_len_tmp, x_tmp = np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
    n_cut = 0
    num = 0
    for conversation in data:
        num += 1
    
    train_num = int(num*0.8)
    test_num = num - train_num

    num = 0
    for conversation in data:
        num += 1
        if(num==train_num):
            break
        conversationID = conversation['conversation_ID']
        # extract pair_strings from emotion_cause_pairs in conversation['emotion_cause_pairs']
        for pair in conversation['emotion-cause_pairs']:
            # print(pair[0][0])
            pairid1 = int(pair[0].split('_')[0])
            pairid2 = int(pair[1].split('_')[0])
            pair_id_all_train.append(conversationID * 10000 + pairid1 * 100 + pairid2)
        
        # if num==100:
        #     break
        pos_list, cause_list = [], []
        d_len = len(conversation['conversation'])
        # print(d_len)
        for conv in conversation['conversation']:
            emotion = conv['emotion']
            uttId = conv['utterance_ID']-1
            if emotion != 'neutral':
                pos_list.append(uttId)
                cause_list.append(uttId)
            # if line[2].strip() != 'null':
                # cause_list.append(i+1)
            words = conv['text']
            # print(words)
            sen_len_tmp[uttId] = min(len(words.split()), max_sen_len)
            # print(sen_len_tmp[i])
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[uttId][j] = int(word_idx[word])
        for i in pos_list:
            for j in cause_list:
                pair_id_cur = conversationID*10000+i*100+j
                pair_id_train.append(pair_id_cur)
                y_train.append([0,1] if pair_id_cur in pair_id_all_train else [1,0])
                x_train.append([x_tmp[i-1],x_tmp[j-1]])
                sen_len_train.append([sen_len_tmp[i-1], sen_len_tmp[j-1]])
                distance_train.append(j-i+100)

    num = 0
    sen_len_tmp, x_tmp = np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
    for conversation in data:
        num += 1
        if(num==test_num):
            break
        conversationID = conversation['conversation_ID']
        # extract pair_strings from emotion_cause_pairs in conversation['emotion_cause_pairs']
        for pair in conversation['emotion-cause_pairs']:
            # print(pair[0][0])
            pairid1 = int(pair[0].split('_')[0])
            pairid2 = int(pair[1].split('_')[0])
            pair_id_all_test.append(conversationID * 10000 + pairid1 * 100 + pairid2)
        
        # if num==100:
        #     break
        pos_list, cause_list = [], []
        d_len = len(conversation['conversation'])
        # print(d_len)
        for conv in conversation['conversation']:
            emotion = conv['emotion']
            uttId = conv['utterance_ID']-1
            if emotion != 'neutral':
                pos_list.append(uttId)
                cause_list.append(uttId)
            # if line[2].strip() != 'null':
                # cause_list.append(i+1)
            words = conv['text']
            # print(words)
            sen_len_tmp[uttId] = min(len(words.split()), max_sen_len)
            # print(sen_len_tmp[i])
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[uttId][j] = int(word_idx[word])
        for i in pos_list:
            for j in cause_list:
                pair_id_cur = conversationID*10000+i*100+j
                pair_id_test.append(pair_id_cur)
                y_test.append([0,1] if pair_id_cur in pair_id_all_test else [1,0])
                x_test.append([x_tmp[i-1],x_tmp[j-1]])
                sen_len_test.append([sen_len_tmp[i-1], sen_len_tmp[j-1]])
                distance_test.append(j-i+100)
    

    y_train , y_test, x_train, x_test, sen_len_train , sen_len_test, distance_train, distance_test = map(np.array, [ y_train , y_test, x_train, x_test, sen_len_train , sen_len_test, distance_train, distance_test])
    #split x , y , sen_len , distance into 80 : 20 train validation percentage
    # also split for pair_id   
    # pair_id_train = pair_id[:int(len(pair_id)*0.8)]
    # pair_id_test = pair_id[int(len(pair_id)*0.8):]
    # x_train = x[:int(len(x)*0.8)]
    # x_test = x[int(len(x)*0.8):]
    # y_train = y[:int(len(y)*0.8)]
    # y_test = y[int(len(y)*0.8):]
    # sen_len_train = sen_len[:int(len(sen_len)*0.8)]
    # sen_len_test = sen_len[int(len(sen_len)*0.8):]
    # distance_train = distance[:int(len(distance)*0.8)]
    # distance_test = distance[int(len(distance)*0.8):]

    for var in ['y_train', 'x_train', 'sen_len_train', 'distance_train', 'y_test', 'x_test', 'sen_len_test', 'distance_test' ]:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}, (y-negative, y-positive): {}'.format(n_cut, y_train.sum(axis=0)))
    print('load data done!\n')
    return pair_id_all_train,  pair_id_all_test , pair_id_train ,pair_id_test, x_train, x_test, y_train, y_test, sen_len_train, sen_len_test, distance_train, distance_test
    

def acc_prf(pred_y, true_y, doc_len, average='binary'): 
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1

def prf_2nd_step(pair_id_all, pair_id, pred_y, fold = 0, save_dir = ''):
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])
    def write_log():
        pair_to_y = dict(zip(pair_id, pred_y))
        g = open(save_dir+'pair_log_fold{}.txt'.format(fold), 'w')
        doc_id_b, doc_id_e = pair_id_all[0]/10000, pair_id_all[-1]/10000
        idx_1, idx_2 = 0, 0
        for doc_id in range(doc_id_b, doc_id_e+1):
            true_pair, pred_pair, pair_y = [], [], []
            line = str(doc_id) + ' '
            while True:
                p_id = pair_id_all[idx_1]
                d, p1, p2 = p_id/10000, p_id%10000/100, p_id%100
                if d != doc_id: break
                true_pair.append((p1, p2))
                line += '({}, {}) '.format(p1,p2)
                idx_1 += 1
                if idx_1 == len(pair_id_all): break
            line += '|| '
            while True:
                p_id = pair_id[idx_2]
                d, p1, p2 = p_id/10000, p_id%10000/100, p_id%100
                if d != doc_id: break
                if pred_y[idx_2]:
                    pred_pair.append((p1, p2))
                pair_y.append(pred_y[idx_2])
                line += '({}, {}) {} '.format(p1, p2, pred_y[idx_2])
                idx_2 += 1
                if idx_2 == len(pair_id): break
            if len(true_pair)>1:
                line += 'multipair '
                if true_pair == pred_pair:
                    line += 'good '
            line += '\n'
            g.write(line)
    if fold:
        write_log()
    keep_rate = len(pair_id_filtered)/(len(pair_id)+1e-8)
    s1, s2, s3 = set(pair_id_all), set(pair_id), set(pair_id_filtered)
    o_acc_num = len(s1 & s2)
    acc_num = len(s1 & s3)
    o_p, o_r = o_acc_num/(len(s2)+1e-8), o_acc_num/(len(s1)+1e-8)
    p, r = acc_num/(len(s3)+1e-8), acc_num/(len(s1)+1e-8)
    f1, o_f1 = 2*p*r/(p+r+1e-8), 2*o_p*o_r/(o_p+o_r+1e-8)
    
    return p, r, f1, o_p, o_r, o_f1, keep_rate
    
