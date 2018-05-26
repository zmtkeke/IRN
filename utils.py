import os
import math
import random
import numpy as np
import tensorflow as tf
from sklearn import cross_validation, metrics

def norm(matrix):
    n = tf.sqrt(tf.reduce_sum(matrix*matrix,1))
    return tf.reshape(n,[-1,1])

def MatrixCos(inputdata,key):
    #inputdata = [batch,embed]
    #key = [slot,embed]
    #return most similar key_id for each inputdata
    addressing = tf.matmul(inputdata, key, transpose_b = True) #(b,e)*(e,slots) -> (b,s)
    norm1 = norm(inputdata) #(b,1)
    norm2 = norm(key) #(s,1)
    n = tf.matmul(norm1,norm2,transpose_b = True) + 1e-8 #(b,s)
    addressing = tf.div(addressing,n)
    index = tf.reshape(tf.argmax(addressing,1),[-1,1]) #(b,1)
    return tf.to_int32(index)

def SimpleMatrixCos(inputdata,key):
    inputdata = tf.nn.l2_normalize(inputdata,1)
    key = tf.nn.l2_normalize(key,1)
    addressing = tf.matmul(inputdata, key, transpose_b = True) #(b,4)*(4,5) -> (b,5)
    index = tf.reshape(tf.argmax(addressing,1),[-1,1]) #(b,1)
    return tf.to_int32(index)

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    m_i = sum_j l_ij*A*x_ij /J/d
    l_ij = Jd-jd-iJ+2ij  = ij-Ji/2-jd/2+Jd/4
    return l-matrix-transpose (fixed)
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2) 
    encoding = (1 + 4 * encoding / embedding_size / sentence_size) / 2
    return np.transpose(encoding)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise",[t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.
    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot",[t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s])) #tf.zeros([1,s])
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def MultiAcc_C(labels,preds):
    #labels = [[[1,2,3],[4,5,3]],  []
    Acc=[]
    batch_size=preds.shape[0]
    correct = 0.0
    pred_len = preds.shape[1]
    for j in range(batch_size):
        if(labels[j,0,-1]==preds[j,-1]):
            correct += 1.0
    for i in range(3):
        Acc.append(round(metrics.accuracy_score(labels[:,0,i],preds[:,i]),3))
    for i in range(3):
        Acc.append(round(metrics.accuracy_score(labels[:,1,i],preds[:,pred_len/2+i]),3))
    Acc.append(round( correct/batch_size ,3))
    return Acc


def MultiAcc(labels,preds,length):
    #length = path = 2 * hop + 1   (hop == path_l + cons_l + final == path_l * 2 + 1 )
    #compare path and final answer accuracy
    Acc = []

    for i in range(length):
        Acc.append(round(metrics.accuracy_score(labels[:,i],preds[:,i]),3))

    batch_size = preds.shape[0]
    correct = 0.0
    for j in range(batch_size):
        k = length - 1
        while(labels[j,k]==0):
            k -= 2
        if(labels[j,k]==preds[j,k]):
            correct += 1.0   #final answer accuracy 
    Acc.append(round( correct/batch_size ,3))
    return Acc

def RealAnswer(labels,pathpreds):
    #find answer-list from path-list
    batch_size = preds.shape[0]
    anspreds = np.zeros(batch_size,dtype=int)
    for j in range(batch_size):
        k = len(labels[0]) - 1
        while(labels[j,k]==0):
            k -= 2
        anspreds[j] = pathpreds[j,k]
    return anspreds


def ScoreRank(label,scores):
    indexrank = np.argsort(-scores)
    rank = 0.0
    for i in range(len(label)):
        row_rank= np.where(indexrank[i]==label[i])[0][0] #([0], )
        if row_rank < 3:
            rank += 1
    return round(rank/len(label), 3)

def InSet(labels,anset,preds):
    #get accuracy(whether in answer set or not)
    #labels does not matter
    #preds is path-list
    #labels is path-labels
    right = 0.0
    for i in xrange(len(anset)):
        if type(preds[i]) is np.int64:
            ans_pred = preds[i]
        else:
            ans_pred = preds[i,-1]
            '''
            k = len(labels[0]) - 1
            while(labels[i,k]==0):
                k -= 2
            ans_pred = preds[i,k]
            '''
        if ans_pred in anset[i]:
            right += 1
    return round(right/len(anset), 3)

def InnerRight(preds, KBs):
    Acc = []
    pl = len(preds[0])-2
    batch = len(preds)
    flags = np.ones(batch)
    for l in range(0,pl,2):
        right = 0.0
        for j in range(batch):
            if flags[j]==0:
                continue
            key = preds[j,l]*7+preds[j,l+1]
            if preds[j,l+2] in KBs[key]:
                right += 1
            else:
                flags[j]=0
        Acc.append(round(right/batch ,3))
    return Acc
