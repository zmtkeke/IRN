import os
import tensorflow as tf
import numpy as np
import time

from data_process import process_data, process_data_c
from utils import MultiAcc, MultiAcc_C, RealAnswer, ScoreRank, InSet, InnerRight
from sklearn import cross_validation, metrics
from model import IRN, IRN_C

flags = tf.app.flags


flags.DEFINE_integer("edim", 50, "words vector dimension [50]")
flags.DEFINE_integer("nhop", 3, "number of hops [2/3+1]")
flags.DEFINE_integer("batch_size", 50, "batch size to use during training [50]")
flags.DEFINE_integer("nepoch", 5000, "number of epoch to use during training [1000]")
flags.DEFINE_integer("inner_nepoch",3, "PRN inner loop [5]")
flags.DEFINE_float("init_lr", 0.001, "initial learning rate")
flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
#flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
#flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 20, "clip gradients to this norm [20]")
flags.DEFINE_string("dataset", "pq", "pq2h/pq3h/pql2h/pql3h/wc/")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "checkpoint directory")
flags.DEFINE_boolean("unseen",False,"True to hide 3 relations when training [False]")
FLAGS = flags.FLAGS

FLAGS.data_dir = "data/WC2014"
FLAGS.KB_file = "WC2014"
if FLAGS.dataset == 'wc1h':
    FLAGS.data_file = "WC-P1" #"WC-C/P1/P2/P"
elif FLAGS.dataset == 'wc2h':
    FLAGS.data_file = "WC-P2" #"WC-C/P1/P2/P"
elif FLAGS.dataset == 'wcm':
    FLAGS.data_file = "WC-P" #"WC-C/P1/P2/P"
elif FLAGS.dataset == 'wcc':
    FLAGS.data_file = "WC-C" #"WC-C/P1/P2/P"
elif FLAGS.dataset == 'pql2h':
    FLAGS.data_dir = "PathQuestion"
    FLAGS.data_file = 'PQL-2H'
    FLAGS.KB_file = 'PQL2-KB'
elif FLAGS.dataset == 'pql3h':
    FLAGS.data_dir = "PathQuestion"
    FLAGS.data_file = 'PQL-3H'
    FLAGS.KB_file = 'PQL3-KB'
elif FLAGS.dataset == 'pq2h':
    FLAGS.data_dir = "PathQuestion"
    FLAGS.data_file = 'PQ-2H'
    FLAGS.KB_file = '2H-kb'
elif FLAGS.dataset == 'pq3h':
    FLAGS.data_dir = "PathQuestion"
    FLAGS.data_file = 'PQ-3H'
    FLAGS.KB_file = '3H-kb'


def main(_):
    word2id = {}
    ent2id = {}
    rel2id = {}
    words = set()
    relations = set()
    entities = set()

    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir,FLAGS.data_file)
    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir,FLAGS.KB_file)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    KB_file = '%s/%s.txt' % (FLAGS.data_dir, FLAGS.KB_file)
    data_file = '%s/%s.txt' % (FLAGS.data_dir, FLAGS.data_file)


    start = time.time()
    if FLAGS.data_file == "WC-C":
        Q,A,P,S,Triples,FLAGS.query_size = process_data_c(KB_file, data_file, word2id, rel2id, ent2id, words, relations, entities)
        FLAGS.path_size = len(P[0][0]) #5
    else:
        Q,A,P,S,Triples,FLAGS.query_size = process_data(KB_file, data_file, word2id, rel2id, ent2id, words, relations, entities)
        FLAGS.path_size = len(P[0]) #5 or 7 or 

    FLAGS.nhop = FLAGS.path_size / 2

    print ("read data cost %f seconds" %(time.time()-start))
    FLAGS.nwords = len(word2id) 
    FLAGS.nrels = len(rel2id) 
    FLAGS.nents = len(ent2id)
    
    trainQ, testQ, trainA, testA, trainP, testP, trainS, testS = cross_validation.train_test_split(Q, A, P, S, test_size=.1, random_state=123)
    
    # for UNSEEN relations (incomplete kb setting, change data_utils.py)
    if FLAGS.unseen:
        id_c=[]
        for idx in range(trainQ.shape[0]):
            if trainP[idx][-4] == 1 or trainP[idx][-4]==2 or trainP[idx][-4]==3:
                id_c.append(idx)
        trainQ = np.delete(trainQ,id_c,axis=0)
        trainA = np.delete(trainA,id_c,axis=0) 
        trainP = np.delete(trainP,id_c,axis=0)
        trainS = np.delete(trainS,id_c,axis=0) 
    

    #
    #other data and some flags
    #
    id2word = dict(zip(word2id.values(), word2id.keys()))
    id2rel = dict(zip(rel2id.values(), rel2id.keys())) #{0: '<end>', 1: 'cause_of_death', 2: 'gender', 3: 'profession', 4: 'institution', 5: 'religion', 6: 'parents', 7: 'location', 8: 'place_of_birth', 9: 'nationality', 10: 'place_of_death', 11: 'spouse', 12: 'children', 13: 'ethnicity'} 

    test_labels = np.argmax(testA, axis=1)


    print(flags.FLAGS.__flags)

    with tf.Session() as sess:
        if not FLAGS.data_file == "WC-C":
            model = IRN(FLAGS,sess)
        elif FLAGS.data_file == "WC-C":
            model = IRN_C(FLAGS,sess)
        
        model.load()

        test_preds = model.predict(Triples,testQ, testP)

        if not FLAGS.data_file == "WC-C":
            test_acc = MultiAcc(testP,test_preds,FLAGS.path_size)
        elif FLAGS.data_file == "WC-C":
            test_acc = MultiAcc_C(testP,test_preds)

        test_true_acc = InSet(testP,testS,test_preds)


        print('-----------------------')
        print('Test Data',data_file)
        print('Test Accuracy:', test_true_acc)
        print('Test Accuracy for whole Path:', test_acc)                    
        print('-----------------------')




if __name__ == '__main__':
    tf.app.run()