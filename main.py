import os
import tensorflow as tf
import numpy as np
import time

from data_utils import process_data, process_data_c
from utils import MultiAcc, MultiAcc_C, RealAnswer, ScoreRank, InSet, InnerRight
from sklearn import cross_validation, metrics
from baseline import MemN2N, KVMemN2N, Embed, SubgraphEmbed, Seq2Seq
from model import IRN, IRN_C

flags = tf.app.flags

flags.DEFINE_string("model","IRN", "model name [IRN/KVMemN2N/MemN2N/Embed/SubgraphEmbed]")

flags.DEFINE_integer("feature", 20, "internal state dimension [20]")
flags.DEFINE_integer("edim", 50, "words vector dimension [50]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 3, "number of hops [2/3]")
flags.DEFINE_integer("mem_size", 100, "memory size [100]")
flags.DEFINE_integer("batch_size", 50, "batch size to use during training [50]")
flags.DEFINE_integer("nepoch", 1000, "number of epoch to use during training [1000]")
flags.DEFINE_integer("inner_nepoch",3, "PRN inner loop [5]")
flags.DEFINE_float("init_lr", 0.001, "initial learning rate")
flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 20, "clip gradients to this norm [20]")
flags.DEFINE_string("dataset", "wc", "pq/pql/wc/")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "checkpoint directory")
flags.DEFINE_boolean("encoder", False, "True for use LSTM to encode question [False]")
flags.DEFINE_boolean("unseen",False,"True to hide 3 relations when training [False]")
FLAGS = flags.FLAGS
if FLAGS.dataset == 'wc':
    FLAGS.data_dir = "data/WC2014"
    FLAGS.data_file = "WC-P2" #"WC-C/P1/P2/P"
    FLAGS.KB_file = "WC2014"
elif FLAGS.dataset == 'pql':
    FLAGS.data_dir = "data/PQL"
    df_list = ['2hop','3hop']
    kb_list = ['exact_2mkb','exact_2mkb3']
    FLAGS.data_file =df_list[1] 
    FLAGS.KB_file =kb_list[1]
elif FLAGS.dataset == 'pq':
    FLAGS.data_dir = "data/PQ"
    df_list = ['2H','3H']
    kb_list = ['2H-kb','3H-kb']
    FLAGS.data_file =df_list[1] 
    FLAGS.KB_file =kb_list[1] 


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
        Q,A,P,D,S,_,_,_,_,_,Triples,KBs,FLAGS.query_size,FLAGS.mem_size,FLAGS.tails_size= process_data_c(KB_file, data_file, word2id, rel2id, ent2id, words, relations, entities)
        FLAGS.path_size = len(P[0][0]) #5
    else:
        Q,A,P,D,S,_,_,_,_,_,Triples,KBs,FLAGS.query_size,FLAGS.mem_size,FLAGS.tails_size= process_data(KB_file, data_file, word2id, rel2id, ent2id, words, relations, entities)
        FLAGS.path_size = len(P[0]) #5 or 7 or 


    print ("read data cost %f seconds" %(time.time()-start))
    FLAGS.nwords = len(word2id) 
    FLAGS.nrels = len(rel2id) 
    FLAGS.nents = len(ent2id)
    if FLAGS.data_file == "CPQ-2hop":
        FLAGS.nhop = (FLAGS.path_size-1)/4
    else:
        FLAGS.nhop = (FLAGS.path_size-1)/2
    
    trainQ, testQ, trainA, testA, trainP, testP, trainD, testD, trainS, testS = cross_validation.train_test_split(Q, A, P, D, S, test_size=.1, random_state=123)
    trainQ, validQ, trainA, validA, trainP, validP, trainD, validD, trainS, validS = cross_validation.train_test_split(trainQ, trainA, trainP, trainD, trainS, test_size=.11, random_state=0)
    
    # for UNSEEN relations (incomplete kb setting, change data_utils.py)
    if FLAGS.unseen:
        id_c=[]
        for idx in range(trainQ.shape[0]):
            if trainP[idx][-4] == 1 or trainP[idx][-4]==2 or trainP[idx][-4]==3:
                id_c.append(idx)
        trainQ = np.delete(trainQ,id_c,axis=0)
        trainA = np.delete(trainA,id_c,axis=0) 
        trainP = np.delete(trainP,id_c,axis=0)
        trainD = np.delete(trainD,id_c,axis=0)
        trainS = np.delete(trainS,id_c,axis=0) 
    
    n_train = trainQ.shape[0]     
    n_test = testQ.shape[0]
    n_val = validQ.shape[0]
    print("Training Size", n_train) 
    print("Validation Size", n_val) 
    print("Testing Size", n_test) 
    

    #
    #other data and some flags
    #
    id2word = dict(zip(word2id.values(), word2id.keys()))
    id2rel = dict(zip(rel2id.values(), rel2id.keys())) #{0: '<end>', 1: 'cause_of_death', 2: 'gender', 3: 'profession', 4: 'institution', 5: 'religion', 6: 'parents', 7: 'location', 8: 'place_of_birth', 9: 'nationality', 10: 'place_of_death', 11: 'spouse', 12: 'children', 13: 'ethnicity'} 
    
    train_labels = np.argmax(trainA, axis=1)
    test_labels = np.argmax(testA, axis=1)
    valid_labels = np.argmax(validA, axis=1)


    print(flags.FLAGS.__flags)

    #batch_id
    #batches = [(start, end) for start, end in batches] abandom last few examples
    batches = zip(range(0, n_train-FLAGS.batch_size, FLAGS.batch_size), range(FLAGS.batch_size, n_train, FLAGS.batch_size))

    r = np.arange(n_train) # instance idx to be shuffled
    l = n_train / FLAGS.batch_size * FLAGS.batch_size #total instances used in training 

    with tf.Session() as sess:
        if FLAGS.model == 'IRN' and not FLAGS.data_file == "WC-C":
            if FLAGS.model == 'IRN':
                model = IRN(FLAGS,sess)
            #model.load()

            print("KBs Size", Triples.shape[0]) #144
            pre_batches = zip(range(0, Triples.shape[0]-FLAGS.batch_size, FLAGS.batch_size), range(FLAGS.batch_size, Triples.shape[0], FLAGS.batch_size))

            pre_val_preds = model.predict(Triples, validQ, validP)
            pre_test_preds = model.predict(Triples, testQ, testP)
            best_val_epoch = -1
            best_val_acc = MultiAcc(validP,pre_val_preds,FLAGS.path_size)
            best_val_true_acc = InSet(validP,validS,pre_val_preds)
            best_test_true_acc = InSet(testP,testS,pre_test_preds)
            best_test_path_acc = best_val_acc
            #best_val_path_acc = InnerRight(pre_val_preds, KBs)
             
            for t in range(1,FLAGS.nepoch + 1):
                start = time.time()
                np.random.shuffle(batches)
                for i in range(FLAGS.inner_nepoch):
                    np.random.shuffle(pre_batches)
                    pre_total_cost = 0.0
                    for s,e in pre_batches:
                        pre_total_cost += model.batch_pretrain(Triples[s:e],trainQ[0:FLAGS.batch_size],trainA[0:FLAGS.batch_size],np.argmax(trainA[0:FLAGS.batch_size], axis=1),trainP[0:FLAGS.batch_size])

                total_cost = 0.0                
                for s,e in batches:
                    total_cost += model.batch_fit(Triples[s:e],trainQ[s:e],trainA[s:e],np.argmax(trainA[s:e], axis=1),trainP[s:e])

                if t % 1 == 0:
                    train_preds = model.predict(Triples,trainQ,trainP)
                    train_acc = MultiAcc(trainP,train_preds,FLAGS.path_size)
                    train_true_acc = InSet(trainP,trainS,train_preds)

                    val_preds = model.predict(Triples,validQ, validP) # (n_val,1)  each is answer id
                    val_acc = MultiAcc(validP,val_preds,FLAGS.path_size)
                    val_true_acc = InSet(validP,validS,val_preds)

                    test_preds = model.predict(Triples,testQ, testP)
                    test_acc = MultiAcc(testP,test_preds,FLAGS.path_size)
                    test_true_acc = InSet(testP,testS,test_preds)

                    if val_true_acc > best_val_true_acc:
                        best_val_epoch = t
                        best_val_true_acc = val_true_acc
                        best_test_true_acc = test_true_acc
                        best_test_path_acc = test_acc
                        #model.store()

                    print('-----------------------')
                    print('Epoch', t)
                    print('timing', (time.time()-start))
                    print('Total Cost:', total_cost)
                    print('Train Accuracy:', train_true_acc)
                    print('Validation Accuracy:', val_true_acc)
                    print('Best Validation epoch & Acc:', best_val_epoch, best_val_true_acc)
                    print('Test Accuracy:', best_test_true_acc)
                    print('Test Accuracy for whole Path:', best_test_path_acc)                    
                    print('-----------------------')

                    '''
                    if not t % 100 == 0:
                        continue
                    idx = model.match()
                    for i in range(1,14):
                        print "relation: ",id2word[i]
                        print "similar words are: "
                        for iid in idx[i]:
                            print id2word[iid]
                        print('-----------------------')
                        print('-----------------------')
                    '''

        elif FLAGS.model == 'IRN' and FLAGS.data_file == "WC-C":
            if FLAGS.model == 'IRN':
                model = IRN_C(FLAGS,sess)
            #model.load()

            KBs = Triples
            print("KBs Size", KBs.shape[0]) #144
            pre_batches = zip(range(0, KBs.shape[0]-FLAGS.batch_size, FLAGS.batch_size), range(FLAGS.batch_size, KBs.shape[0], FLAGS.batch_size))

            pre_val_preds = model.predict(KBs, validQ, validP)
            pre_test_preds = model.predict(Triples, testQ, testP)
            best_val_epoch = -1
            best_val_true_acc = InSet(validP,validS,pre_val_preds)
            best_test_true_acc = InSet(testP,testS,pre_test_preds)
            best_test_path_acc = best_val_acc
             
            for t in range(1,FLAGS.nepoch + 1):
                start = time.time()
                np.random.shuffle(batches)
                for i in range(FLAGS.inner_nepoch):
                    np.random.shuffle(pre_batches)
                    pre_total_cost = 0.0
                    for s,e in pre_batches:
                        pre_total_cost += model.batch_pretrain(KBs[s:e],trainQ[0:FLAGS.batch_size],trainA[0:FLAGS.batch_size],np.argmax(trainA[0:FLAGS.batch_size], axis=1),trainP[0:FLAGS.batch_size])

                total_cost = 0.0                
                for s,e in batches:
                    total_cost += model.batch_fit(KBs[s:e],trainQ[s:e],trainA[s:e],np.argmax(trainA[s:e], axis=1),trainP[s:e])

                if t % 1 == 0:
                    train_preds = model.predict(KBs,trainQ,trainP)
                    train_acc = MultiAcc_C(trainP,train_preds)
                    train_true_acc = InSet(trainP,trainS,train_preds)
                    val_preds = model.predict(KBs,validQ, validP) # (n_val,1)  each is answer id
                    val_acc = MultiAcc_C(validP,val_preds)
                    val_true_acc = InSet(validP,validS,val_preds)
                    test_preds = model.predict(KBs,testQ, testP)
                    test_acc = MultiAcc_C(testP,test_preds)
                    test_true_acc = InSet(testP,testS,test_preds)

                    if val_true_acc > best_val_true_acc:
                        best_val_epoch = t
                        best_val_true_acc = val_true_acc
                        best_test_true_acc = test_true_acc
                        best_test_path_acc = test_acc
                        #model.store()

                    print('-----------------------')
                    print('Epoch', t)
                    print('timing', (time.time()-start))
                    print('Total Cost:', total_cost)
                    print('Train Accuracy:', train_true_acc)
                    print('Validation Accuracy:', val_true_acc)
                    print('Best Validation epoch & Acc:', best_val_epoch, best_val_true_acc)
                    print('Test Accuracy:', best_test_true_acc)
                    print('Test Accuracy for whole Path:', best_test_path_acc)                    
                    print('-----------------------')

        elif FLAGS.model == 'Embed' or FLAGS.model == 'SubgraphEmbed':
            if FLAGS.model == 'Embed':
                model = Embed(FLAGS,sess)
            elif FLAGS.model == 'SubgraphEmbed':
                model = SubgraphEmbed(FLAGS,sess)
            #model.load()

            if not FLAGS.data_file == 'WC-C':
                pre_val_preds,_ = model.predict(KBs, validQ, validP)
                best_val_epoch = -1
                best_val_true_acc = InSet(validP,validS,pre_val_preds)
                best_test_acc = best_val_true_acc
             
                for t in range(1,FLAGS.nepoch + 1):
                    start = time.time()
                    np.random.shuffle(r)
                    batches = np.split(r[:l], n_train / FLAGS.batch_size)

                    total_cost = 0.0
                    for batch in batches:
                        #batches is list hase n_train/batch_size elements
                        #batch is list has FLAGS.batch_size numbers
                        total_cost += model.batch_fit(KBs,trainQ[batch],trainA[batch],np.argmax(trainA[batch], axis=1),trainP[batch])

                    if t % 1 == 0:
                        train_preds,train_rank = model.predict(KBs,trainQ,trainP)
                        train_true_acc = InSet(trainP,trainS,train_preds)
                        val_preds,val_rank = model.predict(KBs,validQ, validP) # (n_val,1)  each is answer id
                        val_true_acc = InSet(validP,validS,val_preds)
                        test_preds,test_rank = model.predict(KBs,testQ, testP)
                        test_true_acc = InSet(testP,testS,test_preds)


                        if val_true_acc > best_val_true_acc:
                            best_val_true_acc = val_true_acc
                            best_val_epoch = t
                            best_test_acc = test_true_acc
                            #model.store()

                        print('-----------------------')
                        print('Epoch', t)
                        print('timing', (time.time()-start))
                        print('Total Cost:', total_cost)
                        print('train Accuracy:', train_true_acc)
                        print('Validation Accuracy:', val_true_acc)
                        print('Best Validation epoch & Acc:', best_val_epoch, best_val_true_acc)
                        print('Test Accuracy:', best_test_acc)
                        print('-----------------------')
            else:
                pre_val_preds,_ = model.predict(KBs, validQ, validP)
                best_val_epoch = -1
                best_val_true_acc = InSet(validP,validS,pre_val_preds)
                best_test_acc = best_val_true_acc
             
                for t in range(1,FLAGS.nepoch + 1):
                    start = time.time()
                    np.random.shuffle(r)
                    batches = np.split(r[:l], n_train / FLAGS.batch_size)

                    total_cost = 0.0
                    for batch in batches:
                        #batches is list hase n_train/batch_size elements
                        #batch is list has FLAGS.batch_size numbers
                        total_cost += model.batch_fit(KBs,trainQ[batch],trainA[batch],np.argmax(trainA[batch], axis=1),trainP[batch])

                    if t % 1 == 0:
                        train_preds,train_rank = model.predict(KBs,trainQ,trainP)
                        train_true_acc = InSet(trainP,trainS,train_preds)
                        val_preds,val_rank = model.predict(KBs,validQ, validP) # (n_val,1)  each is answer id
                        val_true_acc = InSet(validP,validS,val_preds)
                        test_preds,test_rank = model.predict(KBs,testQ, testP)
                        test_true_acc = InSet(testP,testS,test_preds)

                        if val_true_acc > best_val_true_acc:
                            best_val_true_acc = val_true_acc
                            best_val_epoch = t
                            best_test_acc = test_true_acc
                            #model.store()

                        print('-----------------------')
                        print('Epoch', t)
                        print('timing', (time.time()-start))
                        print('Total Cost:', total_cost)
                        print('train Accuracy:', train_true_acc)
                        print('Validation Accuracy:', val_true_acc)
                        print('Best Validation epoch & Acc:', best_val_epoch, best_val_true_acc)
                        print('Test Accuracy:', best_test_acc)
                        print('-----------------------')
       

        elif FLAGS.model == 'MemN2N' or FLAGS.model =='KVMemN2N':
            if FLAGS.model == "MemN2N":
                model = MemN2N(FLAGS,sess)
            elif FLAGS.model == "KVMemN2N":
                model = KVMemN2N(FLAGS,sess)

            #model.load()

            pre_val_preds, pre_val_lists = model.predict(validD, validQ, validP) #(batch,)
            best_val_epoch = -1
            best_val_acc = MultiAcc(validP,pre_val_lists,FLAGS.path_size)
            best_val_true_acc = InSet(validP,validS,pre_val_preds)
            best_test_true_acc = best_val_true_acc
            best_test_path_acc = best_val_acc

            for t in range(1,FLAGS.nepoch + 1):
                start = time.time()
                np.random.shuffle(r)
                batches = np.split(r[:l], n_train / FLAGS.batch_size)

                total_cost = 0.0
                for batch in batches:
                    total_cost += model.batch_fit(trainD[batch],trainQ[batch],trainA[batch],np.argmax(trainA[batch], axis=1),trainP[batch])
                    pred = model.predict(trainD[batch],trainQ[batch],trainP[batch])

                if t % 1 == 0:
                    train_preds, train_lists = model.predict(trainD,trainQ,trainP)                  
                    val_preds, val_lists = model.predict(validD, validQ, validP)             
                    test_preds, test_lists = model.predict(testD, testQ, testP)
                    
                    train_true_acc = InSet(trainP,trainS,train_preds)
                    val_true_acc = InSet(validP,validS,val_preds)
                    test_true_acc = InSet(testP,testS,test_preds)

                    '''
                    train_acc = metrics.accuracy_score(train_labels,train_preds)
                    val_acc = metrics.accuracy_score(valid_labels,val_preds)
                    test_acc = metrics.accuracy_score(test_labels,test_preds)
                    '''

                    train_acc = MultiAcc(trainP,train_lists,FLAGS.path_size)
                    val_acc = MultiAcc(validP,val_lists,FLAGS.path_size)
                    test_acc = MultiAcc(testP,test_lists,FLAGS.path_size)


                    if val_true_acc > best_val_true_acc:
                        best_val_true_acc = val_true_acc
                        best_val_acc = val_acc
                        best_val_epoch = t
                        best_test_true_acc = test_true_acc
                        best_test_path_acc = test_acc
                        #model.store()


                    print('-----------------------')
                    print('Epoch', t)
                    print('timing', (time.time()-start))
                    print('Total Cost:', total_cost)
                    print('train Accuracy:', train_true_acc)
                    print('Validation Accuracy:', val_true_acc)
                    print('Best Validation epoch:', best_val_epoch)
                    print('Best Validation Accuracy:', best_val_true_acc)
                    print('Test Accuracy:', test_true_acc)
                    print('Test Accuracy for whole path:', test_acc)
                    print('-----------------------')

        elif FLAGS.model == 'Seq2Seq':
            if FLAGS.model == "Seq2Seq":
                model = Seq2Seq(FLAGS,sess)

            #model.load()

            pre_val_preds = model.predict(validQ, validP) #(batch,)
            best_val_epoch = -1
            if FLAGS.data_file == "WC-C":
                best_val_acc = MultiAcc_C(validP,pre_val_preds)
            else:
                best_val_acc = MultiAcc(validP,pre_val_preds,FLAGS.path_size)
            best_val_true_acc = InSet(validP,validS,pre_val_preds)
            best_test_path_acc = best_val_acc
            best_test_true_acc = best_val_true_acc

            for t in range(1,FLAGS.nepoch + 1):
                start = time.time()
                np.random.shuffle(r)
                batches = np.split(r[:l], n_train / FLAGS.batch_size)

                total_cost = 0.0
                for batch in batches:
                    #batches is list hase n_train/batch_size elements
                    #batch is list has FLAGS.batch_size numbers 
                    total_cost += model.batch_fit(trainQ[batch],trainP[batch])
                    pred = model.predict(trainQ[batch],trainP[batch])

                if t % 1 == 0:                 
                    train_preds = model.predict(trainQ,trainP)                  
                    val_preds = model.predict(validQ, validP)             
                    test_preds = model.predict(testQ, testP)
                    
                    train_true_acc = InSet(trainP,trainS,train_preds)
                    val_true_acc = InSet(validP,validS,val_preds)
                    test_true_acc = InSet(testP,testS,test_preds)

                    if FLAGS.data_file == "WC-C":
                        train_acc = MultiAcc_C(trainP,train_preds)
                        val_acc = MultiAcc_C(validP,val_preds)
                        test_acc = MultiAcc_C(testP,test_preds)
                    else:
                        train_acc = MultiAcc(trainP,train_preds,FLAGS.path_size)
                        val_acc = MultiAcc(validP,val_preds,FLAGS.path_size)
                        test_acc = MultiAcc(testP,test_preds,FLAGS.path_size)



                    if val_true_acc > best_val_true_acc:
                        best_val_true_acc = val_true_acc
                        best_val_acc = val_acc
                        best_val_epoch = t
                        best_test_true_acc = test_true_acc
                        best_test_path_acc = test_acc
                        #model.store()


                    print('-----------------------')
                    print('Epoch', t)
                    print('timing', (time.time()-start))
                    print('Total Cost:', total_cost)
                    print('train Accuracy:', train_true_acc)
                    print('Validation Accuracy:', val_true_acc)
                    print('Best Validation epoch:', best_val_epoch)
                    print('Best Validation Accuracy:', best_val_true_acc)
                    print('Test Accuracy:', test_true_acc)
                    print('Test Accuracy for whole path:', test_acc)
                    print('-----------------------')


if __name__ == '__main__':
    tf.app.run()