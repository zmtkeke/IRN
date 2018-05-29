import os
import math
import random
import numpy as np
import tensorflow as tf
from utils import add_gradient_noise,zero_nil_slot,position_encoding
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense



class MemN2N(object):
    """End-To-End Memory Network. reference memn2n_qa"""
    def __init__(self, config, sess):
        self._data_file = config.data_file

        self._path_size = config.path_size

        self._batch_size = config.batch_size
        self._vocab_size = config.nwords
        self._rel_size = config.nrels
        self._ent_size = config.nents
        self._sentence_size = config.query_size
        self._memory_size = config.mem_size
        self._embedding_size = config.edim
        self._hops = config.nhop
        self._max_grad_norm = config.max_grad_norm
        self._init = tf.contrib.layers.xavier_initializer()
    #    self._nonlin = nonlin
    #    self._init = tf.random_normal_initializer(stddev=config.init_std)
        self._opt = tf.train.AdamOptimizer()
        #self._opt = tf.train.GradientDescentOptimizer(learning_rate=config.init_lr)
        self._name = "MemN2N"
        self._checkpoint_dir = config.checkpoint_dir+'/'+self._name

        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
        self._encoder = config.encoder

        self._build_inputs()
        self._build_vars()
        self._saver = tf.train.Saver(max_to_keep=10)
        #encoding_shape = _sentence_size * _embedding_size

        #self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy as loss
        inner_loss, ans_list, logits = self._inference() # (batch_size, ent_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self._answers, tf.float32), name="cross_entropy")
        loss_op = tf.reduce_sum(cross_entropy, name="loss_op") + inner_loss


        # gradient pipeline, seem not affect much
        #grads_and_vars = self._opt.compute_gradients(loss_op,[self.A,self.B,self.C,self.R,self.TA,self.TC])
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        
        
        nil_grads_and_vars = []
        
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        
        for g,v in nil_grads_and_vars:
            print g, v.name
        
        #grads_and_vars = [(tf.Print(g, [v.name,str(g.get_shape()),g], summarize=1e1/2), v) for g, v in nil_grads_and_vars]
        #train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")
        
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op") #(b,)
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_list_op = ans_list
        self.predict_proba_op = predict_proba_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = sess
        self._sess.run(init_op)


    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, 3], name="stories")
        #self._stories = tf.placeholder(tf.int32, [None, self._memory_size, 1], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._ent_size], name="answers")
        if self._data_file == "WC-C":
            self._paths = tf.placeholder(tf.int32, [None, 2, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        else:
            self._paths = tf.placeholder(tf.int32, [None, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        self._answers_id = tf.placeholder(tf.int32, [None], name="answers_id") #id for answer
        self._paddings = tf.placeholder(tf.int64, [None], name="paddings") #for id_padding
        self._ones = tf.placeholder(tf.float32, [None], name="paddings") #for multiple
        self._zeros = tf.placeholder(tf.float32, [None], name="paddings") #for add
        
    def _build_vars(self):
        with tf.variable_scope(self._name):
            
            nil_word_slot = tf.zeros([1, self._embedding_size])
            nil_rel_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._ent_size-1, self._embedding_size]) ])
            B = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            R = tf.concat(axis=0, values=[ nil_rel_slot, self._init([self._rel_size-1, self._embedding_size]) ])
            self.A = tf.Variable(A, name="A") # encode entity to vector to calculate weight
            self.B = tf.Variable(B, name="B") # encode question-words to vector
            self.C = tf.Variable(A, name="C") # encode entity to vector 
            self.R = tf.Variable(R, name="R") # encode relation to vector
            
            
            
            #self.A = tf.Variable(self._init([self._ent_size, self._embedding_size]), name="A") # encode entity to vector to calculate weight
            #self.B = tf.Variable(self._init([self._vocab_size, self._embedding_size]), name="B") # encode question-words to vector
            #self.C = tf.Variable(self._init([self._ent_size, self._embedding_size]), name="C") # encode entity to vector 
            #self.R = tf.Variable(self._init([self._rel_size, self._embedding_size]), name="R") # encode relation to vector
            

            #self.TA = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TA')
            #self.TC = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TC')
            
            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            self.W = tf.Variable(self._init([self._embedding_size, self._ent_size]), name="W")
        self._nil_vars = set([self.A.name, self.C.name, self.B.name, self.R.name]) #need to keep first line 0

    def _inference(self):
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self.B, self._queries) #Ax_ij shape is (batch, sentence_size ,embedding_size)
            u_0 = tf.reduce_sum(q_emb, 1) #shape is (batch,embed)

            q = tf.transpose(q_emb,[1,0,2]) #(s,b,e)
            q = tf.reshape(q,[-1,self._embedding_size]) #(s*b,e)
            q = tf.split(axis=0,num_or_size_splits=self._sentence_size,value=q)  #a list of sentence_size tensors of shape [batch,embed]

            '''
            # Define a lstm cell with tensorflow
            if self._encoder:
                lstm_cell = rnn_cell.BasicLSTMCell(self._embedding_size, forget_bias=1.0)
                # Get lstm cell output
                outputs, states = rnn.rnn(lstm_cell, q, dtype=tf.float32) #s * (b,e) list
                u_0 = outputs[-1] #(b,e)

            '''

            u=[u_0]
            a_index = tf.argmax(tf.matmul(u_0, self.W),1)
            al = tf.reshape(tf.cast(a_index,tf.int32),[-1,1])

            
            #d1 = stories.get_shape().as_list()[0] #b = None
            d2 = self._stories.get_shape().as_list()[1] #memory 
            d3 = self._stories.get_shape().as_list()[2] #triple_size = 3


            e1 = tf.reshape(self._stories[:,:,0],[-1,d2,1]) #(batch,memory,1)
            r = tf.reshape(self._stories[:,:,1],[-1,d2,1])
            e2 = tf.reshape(self._stories[:,:,2],[-1,d2,1])
            
            inner_loss = 0
            for hop in range(self._hops):
                
                
                #m is attention
                m_1 = tf.nn.embedding_lookup(self.A, e1) #shape is (batch,memory,1,embedding)
                m_2 = tf.nn.embedding_lookup(self.R, r) #shape is (batch,memory,1,embedding)
                m_3 = tf.nn.embedding_lookup(self.A, e2) #shape is (batch,memory,1,embedding)
                m_emb = tf.concat(axis=2,values=[m_1,m_2,m_3]) #shape is (batch,memory,3,embedding)
                m = tf.reduce_sum(m_emb, 2) #+ self.TA #(batch,memory,embed)


               # mm = tf.reduce_sum(tf.nn.embedding_lookup(self.C,stories),2) + self.TC #(b,m,s,e)->(b,m,e)
                mm_1 = tf.nn.embedding_lookup(self.C, e1) #shape is (batch,memory,1,embedding)
                mm_2 = tf.nn.embedding_lookup(self.R, r) #shape is (batch,memory,1,embedding)
                mm_3 = tf.nn.embedding_lookup(self.C, e2) #shape is (batch,memory,1,embedding)
                mm_emb = tf.concat(axis=2,values=[mm_1,mm_2,mm_3]) #shape is (batch,memory,3,embedding)
                mm = tf.reduce_sum(mm_emb, 2) #+ self.TC #(batch,memory,embed)
                '''
                m = tf.squeeze(tf.nn.embedding_lookup(self.A, self._stories)) #(b,m,1,e)->(b,m,e)
                mm = tf.squeeze(tf.nn.embedding_lookup(self.C, self._stories)) #(b,m,1,e)->(b,m,e)
                '''
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1]) #(b,e,1)->(b,1,e)
                dotted = tf.reduce_sum(m * u_temp, 2) #(b,m,e)->(b,m)
                # Calculate probabilities/ weights over slots
                probs = tf.nn.softmax(dotted)


                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1]) #(batch,m,1)->(batch,1,m)
                c_temp = tf.transpose(mm, [0, 2, 1]) #(b,e,m)
                o_k = tf.reduce_sum(c_temp * probs_temp, 2) #(b,e,m)->(b,e) sum_over_memoryslots
                

                u_k = tf.matmul(u[-1], self.H) + o_k #(batch, embed)
                # nonlinearity
                #if self._nonlin:
                #    u_k = nonlin(u_k)

                a_index = tf.argmax(tf.matmul(u_k, self.W),1)

                u.append(u_k)
                al = tf.concat(axis=1,values=[al,tf.reshape(tf.cast(tf.zeros_like(a_index),tf.int32),[-1,1])])
                al = tf.concat(axis=1,values=[al,tf.reshape(tf.cast(a_index,tf.int32),[-1,1])])

                #additional supervision, wc-c is not applicable
                #logits = tf.matmul(u_k, self.W)
                #real_ans_onehot = tf.one_hot(self._paths[:,2 * hop+2], self._ent_size, on_value=1.0, off_value=0.0, axis=-1) #(b,rel_size)
                #inner_loss = inner_loss + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=real_ans_onehot)) #(b,1)
                
            #u_k = tf.matmul(u[-1], self.H)

            return inner_loss, al, tf.matmul(u_k, self.W) #(b,e)*(e,ent) -> (b,ent)

    def batch_fit(self, stories, queries, answers, answers_id, paths):
    #def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, ent_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        nexample = queries.shape[0]
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._answers_id: answers_id, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        #feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, queries, paths):
    #def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: id (None, 1)  ,predict_op = max(1, [None,ent_size])
        """
        nexample = queries.shape[0]
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._stories: stories, self._queries: queries, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        #feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run([self.predict_op,self.predict_list_op], feed_dict=feed_dict)

    def predict_proba(self, stories, queries, paths):
    #def predict_proba(self, stories):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, ent_size)
        """
        nexample = queries.shape[0]
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._stories: stories, self._queries: queries, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        #feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)


    def store(self):
        file = os.path.join(self._checkpoint_dir, self._name)
        print(" [*] save current parameters to %s." % file )
        self._saver.save(self._sess, file)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir,latest_filename = 'checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            print ("[*] Read from %s" % ckpt.model_checkpoint_path)
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            print (" [!] Test mode but no checkpoint found")
            #raise Exception(" [!] Trest mode but no checkpoint found")

class KVMemN2N(object):
    def __init__(self, config, sess):
        self._data_file = config.data_file

        self._path_size = config.path_size

        self._batch_size = config.batch_size
        self._vocab_size = config.nwords
        self._rel_size = config.nrels
        self._ent_size = config.nents
        self._sentence_size = config.query_size

        #|key| = |value| 
        self._memory_key_size = config.mem_size
        self._memory_value_size = config.mem_size

        self._embedding_size = config.edim
        self._feature_size = config.feature
        self._hops = config.nhop
        self._max_grad_norm = config.max_grad_norm
        self._init = tf.contrib.layers.xavier_initializer()
      #  self._init = tf.random_normal_initializer(stddev=config.init_std)
        self._opt = tf.train.AdamOptimizer()
        self._name = "KVMemN2N"
        self._checkpoint_dir = config.checkpoint_dir+'/'+self._name
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        self._build_inputs()
        self._build_vars()
        self._saver = tf.train.Saver()

        #self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy as loss
        '''
        out = self._inference(self._KBcandidates, self._queries) # (f,b)
        out = tf.transpose(out) #(b,f)
        y_tmp = tf.matmul(self.A, self.V, transpose_b=True)  # (feature ,embedding/hidden) * (embedding ,ent_size) -> (f,v)
        logits = tf.matmul(out, y_tmp) # (b,f)*(f,ent) -> (b,ent)
        '''
        out, logits, ans_list, inner_loss = self._inference(self._KBcandidates, self._queries)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self._answers, tf.float32), name="cross_entropy")
        loss_op = tf.reduce_sum(cross_entropy, name="loss_op") + inner_loss


        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars if g is not None]
        #grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        #nil_grads_and_vars = []
        #for g, v in grads_and_vars:
        #    if v.name in self._nil_vars:
        #        nil_grads_and_vars.append((zero_nil_slot(g), v))
        #    else:
        #        nil_grads_and_vars.append((g, v))
        #grads_and_vars = [(tf.Print(g, [v.name,g], summarize=1e0), v) if g is not None else None for g, v in grads_and_vars]
        train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_list_op = ans_list
        self.predict_proba_op = predict_proba_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = sess
        self._sess.run(init_op)


    def _build_inputs(self):
        #self._keys = tf.placeholder(tf.int32, [None, self._memory_key_size, 2], name="memory_key")
        #self._values = tf.placeholder(tf.int32, [None, self._memory_value_size,1],name="memory_value")
        self._KBcandidates = tf.placeholder(tf.int32, [None, self._memory_key_size,3],name="KBcandidates")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._ent_size], name="answers")
        if self._data_file == "WC-C":
            self._paths = tf.placeholder(tf.int32, [None, 2, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        else:
            self._paths = tf.placeholder(tf.int32, [None, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        self._answers_id = tf.placeholder(tf.int32, [None], name="answers_id") #id for answer
        self._paddings = tf.placeholder(tf.int64, [None], name="paddings") #for id_padding
        self._ones = tf.placeholder(tf.float32, [None], name="paddings") #for multiple
        self._zeros = tf.placeholder(tf.float32, [None], name="paddings") #for add

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            nil_rel_slot = tf.zeros([1, self._embedding_size])
            Erep = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._ent_size-1, self._embedding_size]) ])
            Wrep = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            Rrep = tf.concat(axis=0, values=[ nil_rel_slot, self._init([self._rel_size-1, self._embedding_size]) ])
            self.K = tf.Variable(Erep, name="K") # encode key-entity to vector to calculate weight
            self.B = tf.Variable(Wrep, name="B") # encode question-words to vector
            self.V = tf.Variable(Erep, name="V") # encode value-entity to vector 
            self.R = tf.Variable(Rrep, name="R") # encode relation to vector

            self.TK = tf.Variable(self._init([self._memory_key_size, self._embedding_size]), name='TK')
            self.TV = tf.Variable(self._init([self._memory_value_size, self._embedding_size]), name='TV')
            
            self.A = tf.Variable(self._init([self._feature_size, self._embedding_size]), name ='A')
            self.H_list=[]
            for _ in range(self._hops):
                # define R for variables
                H = tf.get_variable('H{}'.format(_), shape=[self._feature_size, self._feature_size],
                                initializer=tf.contrib.layers.xavier_initializer())
                self.H_list.append(H)

        self._nil_vars = set([self.K.name, self.V.name, self.B.name, self.R.name])

    def _inference(self, KBcandidates, queries):
        with tf.variable_scope(self._name):
            q_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.B, queries), 1) #\sum Bx_ij shape is (batch, (sentence_size ),embedding_size)
            u_0 = tf.matmul(self.A, q_emb, transpose_b = True) #shape is (feature * embed) * (batch,embed)^T = (feature, batch)
            u = [u_0] #feature * batch

            out = tf.transpose(u[-1]) #(b,f)
            y_tmp = tf.matmul(self.A, self.V, transpose_b=True)  # (feature ,embedding/hidden) * (embedding ,ent_size) -> (f,v)
            logits = tf.matmul(out, y_tmp) # (b,f)*(f,ent) -> (b,ent)
            a_index = tf.argmax(logits,1)
            al = tf.reshape(tf.cast(a_index,tf.int32),[-1,1])

            #d1 = KBcandidates.get_shape().as_list()[0] #b = None
            d2 = KBcandidates.get_shape().as_list()[1] # memory_key/value_size
            d3 = KBcandidates.get_shape().as_list()[2] #triple_size = 3

            e1 = tf.reshape(KBcandidates[:,:,0],[-1,d2,1]) #(batch,memory,1)
            r = tf.reshape(KBcandidates[:,:,1],[-1,d2,1])
            e2 = tf.reshape(KBcandidates[:,:,2],[-1,d2])

            m_1 = tf.nn.embedding_lookup(self.K, e1) #shape is (batch,memory,1,embedding)
            m_2 = tf.nn.embedding_lookup(self.R, r) #shape is (batch,memory,1,embedding)
            mvalues = tf.nn.embedding_lookup(self.V, e2) + self.TV #shape is (batch,memory,embedding)
            key_emb = tf.concat(axis=2,values=[m_1,m_2]) #shape is (batch,memory,3,embedding)
            mkeys = tf.reduce_sum(key_emb, 2) + self.TK #(batch,memory,embed)

            inner_loss = 0

            for h in range(self._hops):
                H = self.H_list[h]  #(f,f)

                k_tmp = tf.reshape(tf.transpose(mkeys, [2, 0, 1]), [self._embedding_size, -1]) # [embedding_size, batch_size x memory_key_size]
                a_k_tmp = tf.matmul(self.A, k_tmp) # [feature_size, batch_size x memory_key_size]
                a_k = tf.reshape(tf.transpose(a_k_tmp), [-1, self._memory_key_size, self._feature_size]) # [batch_size, memory_key_size, feature_size]

                v_tmp = tf.reshape(tf.transpose(mvalues, [2, 0, 1]), [self._embedding_size, -1]) # [embedding_size, batch_size x memory_value_size]
                a_v_tmp = tf.matmul(self.A, v_tmp) # [feature_size, batch_size x memory_key_size]
                a_v = tf.reshape(tf.transpose(a_v_tmp), [-1, self._memory_value_size, self._feature_size]) # [batch_size, memory_value_size, feature_size]

                # hack to get around no reduce_dot
                u_expanded = tf.expand_dims(tf.transpose(u[-1]), [1]) #(b,f)->(b,1,f)
                dotted = tf.reduce_sum(a_k * u_expanded, 2) # (b,mk,f) * (b,1,f) -> (b,mk)
                # Calculate probabilities/ weights
                probs = tf.nn.softmax(dotted) 

                probs_temp = tf.expand_dims(probs, -1) #(b,m) -> (batch,m,1)
                o_k = tf.transpose( tf.reduce_sum(probs_temp * a_v, 1) ) #(b,m,f)->(b,f)->(f,b) sum_over_memoryslots

                u_k = tf.matmul(H, u[-1]+o_k) #(f,f)*(f,b) -> (f,b)
                # nonlinearity
                #if self._nonlin:
                #    u_k = nonlin(u_k)

                #out = self._inference(self._KBcandidates, self._queries) # (f,b)
                out = tf.transpose(u_k) #(b,f)
                y_tmp = tf.matmul(self.A, self.V, transpose_b=True)  # (feature ,embedding/hidden) * (embedding ,ent_size) -> (f,v)
                logits = tf.matmul(out, y_tmp) # (b,f)*(f,ent) -> (b,ent)
                a_index = tf.argmax(logits,1)

                al = tf.concat(axis=1,values=[al,tf.reshape(tf.cast(tf.zeros_like(a_index),tf.int32),[-1,1])])
                al = tf.concat(axis=1,values=[al,tf.reshape(tf.cast(a_index,tf.int32),[-1,1])])

                #additional supervision, wc-c is not applicable
                #real_ans_onehot = tf.one_hot(self._paths[:,2 * h+2], self._ent_size, on_value=1.0, off_value=0.0, axis=-1) #(b,rel_size)
                #inner_loss = inner_loss + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=real_ans_onehot)) #(b,1)

                u.append(u_k)

            return u[-1], logits, al, inner_loss

    #def batch_fit(self, KBcandidates, queries, answers):
    def batch_fit(self, KBcandidates, queries, answers, answers_id, paths):
        """Runs the training algorithm over the passed batch

        Args:
            KBcandidates: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, ent_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        nexample = queries.shape[0]
        pad = np.arange(nexample)


        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._KBcandidates: KBcandidates, self._queries: queries, self._answers: answers, self._answers_id: answers_id, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        #feed_dict = {self._KBcandidates: KBcandidates, self._queries: queries, self._answers: answers}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    #def predict(self, KBcandidates, queries):
    def predict(self, KBcandidates, queries, paths):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: id (None, 1)  ,predict_op = max(1, [None,ent_size])
        """
        nexample = queries.shape[0]
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._KBcandidates: KBcandidates, self._queries: queries, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        #feed_dict = {self._KBcandidates: KBcandidates, self._queries: queries}
        return self._sess.run([self.predict_op,self.predict_list_op], feed_dict=feed_dict)

    #def predict_proba(self, KBcandidates, queries):
    def predict_proba(self, KBcandidates, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, ent_size)
        """
        nexample = queries.shape[0]
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._KBcandidates: KBcandidates, self._queries: queries, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        #feed_dict = {self._KBcandidates: KBcandidates, self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)


    def store(self):
        file = os.path.join(self._checkpoint_dir, self._name)
        print(" [*] save current parameters to %s." % file )
        self._saver.save(self._sess, file)


    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            print(" [!] Test mode but no checkpoint found")

class Embed(object):
    def __init__(self, config, sess):
        self._data_file = config.data_file

        self._batch_size = config.batch_size
        self._vocab_size = config.nwords #also entity_size
        self._rel_size = config.nrels
        self._ent_size = config.nents
        self._tail_size = config.tails_size  #3rd-dim of KB-matrix
        self._sentence_size = config.query_size
        self._embedding_size = config.edim
        self._path_size = config.path_size
        self._memory_key_size = config.mem_size

        self._encoder = config.encoder
        self._margin = 1
        self._hops = config.nhop
        self._max_grad_norm = config.max_grad_norm
        self._init = tf.contrib.layers.xavier_initializer()
        #self._init = tf.random_normal_initializer(stddev=config.init_std)

        #self._opt = tf.train.GradientDescentOptimizer(learning_rate=config.init_lr)
        #self._opt = tf.train.AdadeltaOptimizer(learning_rate=config.init_lr)
        self._opt = tf.train.AdamOptimizer()
        self._name = "Embed"
        self._checkpoint_dir = config.checkpoint_dir+'/'+self._name

        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        self._build_inputs()
        self._build_vars()
        self._saver = tf.train.Saver(max_to_keep=10)

        self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="encoding")


        batch_loss, p, self.score_op = self._inference() # (b,1), (batch_size, 5)
        loss_op = tf.reduce_sum(batch_loss, name="loss_op")# + 0.000005 * 1.7 * tf.reduce_sum(tf.square(tf.abs(self.M)))
        E_norm_op = tf.nn.l2_normalize(self.EE,1)
        Q_norm_op = tf.nn.l2_normalize(self.QE,1)


        # gradient pipeline, seem not affect much
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))    
        print "nil_grads_and_vars"
        for g,v in nil_grads_and_vars:
            print g, v.name
        #grads_and_vars = [(tf.Print(g, [v.name,str(g.get_shape()),g], summarize=1e1/2), v) for g, v in grads_and_vars]

        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        
        '''
        predict_op = tf.argmax(logits, 1, name="predict_op") #(b,)
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        '''
        predict_op = p

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        #self.predict_proba_op = predict_proba_op
        self.train_op = train_op
        self.Q_norm_op = Q_norm_op
        self.E_norm_op = E_norm_op

        init_op = tf.global_variables_initializer()
        self._sess = sess
        self._sess.run(init_op)


    def _build_inputs(self):
        self._KBs = tf.placeholder(tf.int32, [self._ent_size * self._rel_size, self._tail_size], name="KBs") #_KBs[i*14+j]=[k1,k2,k3] stand for (i,j,k1)(i,j,k2)
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        if self._data_file == "WC-C":
            self._paths = tf.placeholder(tf.int32, [None, 2, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        else:
            self._paths = tf.placeholder(tf.int32, [None, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        #self._paths = tf.placeholder(tf.int32, [None, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        self._answers = tf.placeholder(tf.int32, [None, self._ent_size], name="answers") #id-hot for answer
        self._answers_id = tf.placeholder(tf.int32, [None], name="answers_id") #id for answer
        self._paddings = tf.placeholder(tf.int64, [None], name="paddings") #for id_padding
        self._ones = tf.placeholder(tf.float32, [None], name="paddings") #for multiple
        self._zeros = tf.placeholder(tf.float32, [None], name="paddings") #for add

    def _build_vars(self):
        with tf.variable_scope(self._name):
            
            nil_word_slot = tf.zeros([1, self._embedding_size])
            nil_rel_slot = tf.zeros([1, self._embedding_size])
            E = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._ent_size-1, self._embedding_size]) ])
            Q = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            R = tf.concat(axis=0, values=[ nil_rel_slot, self._init([self._rel_size-1, self._embedding_size]) ])
            self.EE = tf.Variable(E, name="EE") # encode entity to vector to calculate weight
            self.QE = tf.Variable(Q, name="QE") # encode question-words to vector

            #self.M = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="M") #eembed->nwords

        self._nil_vars = set([self.EE.name, self.QE.name]) #need to keep first line 0

    def _inference(self):
        with tf.variable_scope(self._name):
            #initial
            loss = tf.reshape(self._zeros,[-1,1],name='loss')  #(none,1)
            if not self._data_file == 'WC-C':
                s_index = tf.reshape(self._paths[:,0],[-1,1]) #(none,1)

            q_emb = tf.nn.embedding_lookup(self.QE, self._queries) #Ax_ij shape is (batch, sentence_size ,embedding_size)

            '''
            if self._encoder:
                q = tf.transpose(q_emb,[1,0,2]) #(s,b,e)
                q = tf.reshape(q,[-1,self._embedding_size]) #(s*b,e)
                q = tf.split(0,self._sentence_size,q)  #a list of sentence_size tensors of shape [batch,embed]

                # Define a lstm cell with tensorflow
                fw_lstm_cell = rnn_cell.BasicLSTMCell(self._embedding_size/2, forget_bias=1.0)
                bw_lstm_cell = rnn_cell.BasicLSTMCell(self._embedding_size/2, forget_bias=1.0)
                # Get lstm cell output
                #outputs, states = rnn.rnn(lstm_cell, q, dtype=tf.float32) #s * (b,e) list
                outputs,_,_ = rnn.bidirectional_rnn(fw_lstm_cell,bw_lstm_cell,q,dtype=tf.float32) # s * (b,2e) list
                
                q_emb = tf.transpose(tf.pack(outputs,0),[1,0,2]) #(s,b,2e)->(b,s,2e)

            '''
            #q = outputs[-1] #(b,e)
            q = tf.reduce_sum(q_emb, 1) #shape is (batch,embed) V^T*bag_of_words(q)

            t = tf.nn.embedding_lookup(self.EE, self._answers_id) #(batch,embed)
            tt = tf.nn.embedding_lookup(self.EE, self._paddings)

            s = tf.reduce_sum(q*t, 1) #gold score
            ss = tf.reduce_sum(q*tt, 1) #wrong score
            '''
            tmp = tf.matmul(q,self.M)
            s = tf.reduce_sum(tmp * t, 1)
            ss = tf.reduce_sum(tmp * tt, 1)
            '''
            loss = self._margin + ss - s
            loss = tf.maximum(self._zeros,loss) #(b,1)

            logits = tf.matmul(q, self.EE, transpose_b = True) #(b,e)*(v,e) =(b,v)
            #logits = tf.matmul(tmp, self.EE, transpose_b = True) #(b,e)*(v,e) =(b,v)
            if self._data_file == 'WC-C':
                p = tf.reshape(self._paths[:,0,0:2],[-1,2])
                p = tf.concat(axis=1,values=[p,tf.reshape(tf.cast(tf.argmax(logits,1),tf.int32),[-1,1])]) 
                return loss, p, logits

            p = s_index
            #p = tf.concat(1,[p,tf.reshape(self._paths[:,1],[-1,1])])
            #p = tf.concat(1,[p,tf.reshape(tf.cast(tf.argmax(logits,1),tf.int32),[-1,1])])  
            #p = tf.concat(1,[p,tf.reshape(self._paths[:,3],[-1,1])])
            #p = tf.concat(1,[p,tf.reshape(tf.cast(tf.argmax(logits,1),tf.int32),[-1,1])]) 
            for i in range(0,self._hops*2,2):
                p = tf.concat(axis=1,values=[p,tf.reshape(self._paths[:,i+1],[-1,1])])
                p = tf.concat(axis=1,values=[p,tf.reshape(tf.cast(tf.argmax(logits,1),tf.int32),[-1,1])])


            return loss, p, logits

    def batch_fit(self, KBs, queries, answers, answers_id, paths):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, ent_size)
            paths: Tensor

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        nexample = queries.shape[0]
        pad = np.random.randint(self._ent_size,size=nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._KBs: KBs, self._queries: queries, self._answers: answers, self._answers_id: answers_id, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        self._istrain = True
        loss, _ , _, _= self._sess.run([self.loss_op, self.train_op, self.Q_norm_op, self.E_norm_op], feed_dict=feed_dict)
        return loss

    def predict(self, KBs, queries, paths):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: id (None, 1)  ,predict_op = max(1, [None,ent_size])
        """
        nexample = queries.shape[0]
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._KBs: KBs, self._queries: queries, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        self._istrain = False
        return self._sess.run([self.predict_op,self.score_op], feed_dict=feed_dict)

    def store(self):
        file = os.path.join(self._checkpoint_dir, self._name)
        #print(" [*] save current parameters to %s." % file )
        self._saver.save(self._sess, file)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir,latest_filename = 'checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            print ("[*] Read from %s" % ckpt.model_checkpoint_path)
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            print (" [!] Test mode but no checkpoint found")
            #raise Exception(" [!] Trest mode but no checkpoint found")

class SubgraphEmbed(object):
    def __init__(self, config, sess):
        self._data_file = config.data_file

        self._batch_size = config.batch_size
        self._vocab_size = config.nwords #also entity_size
        self._rel_size = config.nrels
        self._ent_size = config.nents
        self._tail_size = config.tails_size  #3rd-dim of KB-matrix
        self._sentence_size = config.query_size
        self._embedding_size = config.edim
        self._path_size = config.path_size
        self._memory_key_size = config.mem_size

        self._encoder = config.encoder
        self._margin = 1
        self._hops = config.nhop
        self._max_grad_norm = config.max_grad_norm
        self._init = tf.contrib.layers.xavier_initializer()
        #self._init = tf.random_normal_initializer(stddev=config.init_std)

        #self._opt = tf.train.GradientDescentOptimizer(learning_rate=config.init_lr)
        #self._opt = tf.train.AdadeltaOptimizer(learning_rate=config.init_lr)
        self._opt = tf.train.AdamOptimizer()
        self._name = "SubgraphEmbed"
        self._checkpoint_dir = config.checkpoint_dir+'/'+self._name

        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        self._build_inputs()
        self._build_vars()
        self._saver = tf.train.Saver(max_to_keep=10)

        self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="encoding")


        batch_loss, p, self.score_op = self._inference() # (b,1), (batch_size, 5)
        loss_op = tf.reduce_sum(batch_loss, name="loss_op")# + 0.000005 * 1.7 * tf.reduce_sum(tf.square(tf.abs(self.M)))
        E_norm_op = tf.nn.l2_normalize(self.W,1)
        Q_norm_op = tf.nn.l2_normalize(self.V,1)


        # gradient pipeline, seem not affect much
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))    
        print "nil_grads_and_vars"
        for g,v in nil_grads_and_vars:
            print g, v.name
        #grads_and_vars = [(tf.Print(g, [v.name,str(g.get_shape()),g], summarize=1e1/2), v) for g, v in grads_and_vars]

        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        predict_op = p

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.train_op = train_op
        self.E_norm_op = E_norm_op
        self.Q_norm_op = Q_norm_op
        init_op = tf.global_variables_initializer()
        self._sess = sess
        self._sess.run(init_op)


    def _build_inputs(self):
        self._KBs = tf.placeholder(tf.int32, [self._ent_size * self._rel_size, self._tail_size], name="KBs") #_KBs[i*14+j]=[k1,k2,k3] stand for (i,j,k1)(i,j,k2)
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        if self._data_file == "WC-C":
            self._paths = tf.placeholder(tf.int32, [None, 2, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        else:
            self._paths = tf.placeholder(tf.int32, [None, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        #self._paths = tf.placeholder(tf.int32, [None, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        self._answers = tf.placeholder(tf.int32, [None, self._ent_size], name="answers") #id-hot for answer
        self._answers_id = tf.placeholder(tf.int32, [None], name="answers_id") #id for answer
        self._paddings = tf.placeholder(tf.int64, [None], name="paddings") #for id_padding
        self._ones = tf.placeholder(tf.float32, [None], name="paddings") #for multiple
        self._zeros = tf.placeholder(tf.float32, [None], name="paddings") #for add

    def _build_vars(self):
        with tf.variable_scope(self._name):
            
            nil_word_slot = tf.zeros([1, self._embedding_size])
            E = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._ent_size-1, self._embedding_size]) ])
            Q = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ]) 
            self.Q = tf.Variable(Q, name="Q") # encode question
            self.W = tf.Variable(E, name="W") # encode entity and path
            self.V = tf.Variable(E, name="v") # encode subgraph


        self._nil_vars = set([self.W.name, self.V.name, self.Q.name]) #need to keep first line 0

    def _inference(self):
        with tf.variable_scope(self._name):
            #initial
            loss = tf.reshape(self._zeros,[-1,1],name='loss')  #(none,1)

            if not self._data_file == 'WC-C':
                s_index = tf.reshape(self._paths[:,0],[-1,1]) #(none,1)

            q_emb = tf.nn.embedding_lookup(self.Q, self._queries) #Ax_ij shape is (batch, sentence_size ,embedding_size)
            q = tf.reduce_sum(q_emb, 1) #shape is (batch,embed) V^T*bag_of_words(q)

            #single entity
            t = tf.nn.embedding_lookup(self.W, self._answers_id) #(batch,embed)
            tt = tf.nn.embedding_lookup(self.W, self._paddings)
            ent_embedding = self.W #(ent,embed)

            #path representation
            wrong_path = self._paths
            tf.random_shuffle(wrong_path)
            if not self._data_file == 'WC-C':
                t += tf.reduce_sum(tf.nn.embedding_lookup(self.W, self._paths),1) - tf.nn.embedding_lookup(self.W, self._paths[:,2])
                tt += tf.reduce_sum(tf.nn.embedding_lookup(self.W, wrong_path),1) - tf.nn.embedding_lookup(self.W, wrong_path[:,2])
            else:
                t += tf.reduce_sum(tf.nn.embedding_lookup(self.W, self._paths[:,0,:]),1) - tf.nn.embedding_lookup(self.W, self._paths[:,0,2])
                tt += tf.reduce_sum(tf.nn.embedding_lookup(self.W, wrong_path[:,0,:]),1) - tf.nn.embedding_lookup(self.W, wrong_path[:,0,2])
                t += tf.reduce_sum(tf.nn.embedding_lookup(self.W, self._paths[:,1,:]),1) - tf.nn.embedding_lookup(self.W, self._paths[:,1,2])
                tt += tf.reduce_sum(tf.nn.embedding_lookup(self.W, wrong_path[:,1,:]),1) - tf.nn.embedding_lookup(self.W, wrong_path[:,1,2])

            #subgraph representation
            '''
            KBs = tf.reshape(self._KBs,[self._ent_size,-1]) #(ent, rel * tail)
            subgraph = tf.nn.embedding_lookup(KBs,self._answers_id) #(batch, rel*tail)
            wrong_graph = tf.nn.embedding_lookup(KBs, self._paddings) #(batch, rel*tail)
            t += tf.reduce_sum(tf.nn.embedding_lookup(self.V, subgraph),1) #(b,r*t,e)->(b,e)
            tt += tf.reduce_sum(tf.nn.embedding_lookup(self.V, wrong_graph),1)
            ent_embedding += tf.reduce_sum(tf.nn.embedding_lookup(self.V, KBs),1) #(v,r*t,e)_>(v,e)
            '''

            s = tf.reduce_sum(q * t, 1) #gold score, dot product
            ss = tf.reduce_sum(q * tt, 1) #wrong score

            loss = self._margin + ss - s
            loss = tf.maximum(self._zeros,loss) #(b,1)

            logits = tf.matmul(q, ent_embedding, transpose_b = True) #(b,e)*(v,e) =(b,v)
            #logits = tf.matmul(q, self.EE, transpose_b = True) #(b,e)*(v,e) =(b,v)

            if self._data_file == 'WC-C':
                p = tf.reshape(self._paths[:,0,0:2],[-1,2])
                p = tf.concat(axis=1,values=[p,tf.reshape(tf.cast(tf.argmax(logits,1),tf.int32),[-1,1])]) 
                return loss, p, logits


            p = s_index
            for i in range(0,self._hops*2,2):
                p = tf.concat(axis=1,values=[p,tf.reshape(self._paths[:,i+1],[-1,1])])
                p = tf.concat(axis=1,values=[p,tf.reshape(tf.cast(tf.argmax(logits,1),tf.int32),[-1,1])])         

            return loss, p, logits

    def batch_fit(self, KBs, queries, answers, answers_id, paths):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, ent_size)
            paths: Tensor

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        nexample = queries.shape[0]
        pad = np.random.randint(self._ent_size,size=nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._KBs: KBs, self._queries: queries, self._answers: answers, self._answers_id: answers_id, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        self._istrain = True
        loss, _ , _, _= self._sess.run([self.loss_op, self.train_op, self.Q_norm_op, self.E_norm_op], feed_dict=feed_dict)
        return loss

    def predict(self, KBs, queries, paths):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: id (None, 1)  ,predict_op = max(1, [None,ent_size])
        """
        nexample = queries.shape[0]
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._KBs: KBs, self._queries: queries, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros}
        self._istrain = False
        return self._sess.run([self.predict_op,self.score_op], feed_dict=feed_dict)

    def store(self):
        file = os.path.join(self._checkpoint_dir, self._name)
        #print(" [*] save current parameters to %s." % file )
        self._saver.save(self._sess, file)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self._checkpoint_dir,latest_filename = 'checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            print ("[*] Read from %s" % ckpt.model_checkpoint_path)
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            print (" [!] Test mode but no checkpoint found")
            #raise Exception(" [!] Trest mode but no checkpoint found")

class Seq2Seq(object):
    def __init__(self, config, sess):
        self._data_file = config.data_file

        self._path_size = config.path_size

        self._batch_size = config.batch_size
        self._vocab_size = config.nwords
        self._rel_size = config.nrels
        self._ent_size = config.nents
        self._kb_size = config.nrels + config.nents
        self._sentence_size = config.query_size
        self._embedding_size = config.edim
        
        self._hops = config.nhop

        self._max_grad_norm = config.max_grad_norm
        self._init = tf.contrib.layers.xavier_initializer()
    #    self._nonlin = nonlin
    #    self._init = tf.random_normal_initializer(stddev=config.init_std)
        self._opt = tf.train.AdamOptimizer(learning_rate=config.init_lr)
        #self._opt = tf.train.GradientDescentOptimizer(learning_rate=config.init_lr)
        self._name = "Seq2Seq"
        self._checkpoint_dir = config.checkpoint_dir+'/'+self._name

        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
        self._encoder = config.encoder

        self._build_inputs()
        self._build_vars()
        self._saver = tf.train.Saver(max_to_keep=10)
        #encoding_shape = _sentence_size * _embedding_size

        #self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy as loss
        stepwise_cross_entropy, self._decoder_prediction = self._inference() # (batch_size, ent_size)
        #print "predict",self._decoder_prediction #(b,7)
        loss_op = tf.reduce_mean(stepwise_cross_entropy,name="loss_op")


        # gradient pipeline, seem not affect much
        #grads_and_vars = self._opt.compute_gradients(loss_op,[self.A,self.B,self.C,self.R,self.TA,self.TC])
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]   
        nil_grads_and_vars = []    
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        '''
        for g,v in nil_grads_and_vars:
            print g, v.name
        '''
        
        #grads_and_vars = [(tf.Print(g, [v.name,str(g.get_shape()),g], summarize=1e1/2), v) for g, v in nil_grads_and_vars]
        #train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")
        
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # assign ops
        self.loss_op = loss_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = sess
        self._sess.run(init_op)


    def _build_inputs(self):
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        if self._data_file == "WC-C":
            self._paths = tf.placeholder(tf.int32, [None, 2, self._path_size], name="paths") 
        else:
            self._paths = tf.placeholder(tf.int32, [None, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        self._zeros = tf.placeholder(tf.int32, [None], name="paddings") #for add

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            E = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._ent_size-1, self._embedding_size]) ])
            Q = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            self.EE = tf.Variable(E, name="EE") # encode entity to vector to calculate weight
            self.QE = tf.Variable(Q, name="QE")# encode question-words to vector
        self._nil_vars = set([self.EE.name, self.QE.name]) #need to keep first line 0

    def _inference(self):
        if not self._data_file == 'WC-C':
            _paths = self._paths
        else:
            _paths = tf.reshape(self._paths,[-1,10])
        encoder_inputs_embedded = tf.nn.embedding_lookup(self.QE, self._queries) #(b,s,e)
        eos = tf.nn.embedding_lookup(self.EE, self._zeros) #(b,e)
        state = tf.nn.embedding_lookup(self.EE, _paths[:,0]) #(b,e)
        decoder_inputs_embedded = tf.expand_dims(eos,1)
        decoder_targets_embedded = tf.expand_dims(state,1)
        decoder_targets = tf.reshape(_paths[:,0],[-1,1])
        for hop in range(self._hops):
            if not self._data_file == 'WC-C':
                e_p = tf.nn.embedding_lookup(self.EE, self._paths[:,2*hop]) #(b,e)
                r = tf.nn.embedding_lookup(self.EE, self._paths[:,2*hop+1]) #(b,e)
                e = tf.nn.embedding_lookup(self.EE, self._paths[:,2*hop+2]) #(b,e)
                decoder_targets = tf.concat(axis=1,values=[decoder_targets, tf.reshape(self._paths[:,2*hop+1],[-1,1]),tf.reshape(self._paths[:,2*hop+2],[-1,1])])
            else:
                e_p = tf.nn.embedding_lookup(self.EE, self._paths[:,hop,0]) #(b,e)
                r = tf.nn.embedding_lookup(self.EE, self._paths[:,hop,1]) #(b,e)
                e = tf.nn.embedding_lookup(self.EE, self._paths[:,hop,2]) #(b,e)
                decoder_targets = tf.concat(axis=1,values=[decoder_targets, tf.reshape(self._paths[:,hop,1],[-1,1]),tf.reshape(self._paths[:,hop,2],[-1,1])])
            decoder_inputs_embedded = tf.concat(axis=1,values=[decoder_inputs_embedded, tf.expand_dims(e_p,1), tf.expand_dims(r,1)])
            decoder_targets_embedded = tf.concat(axis=1,values=[decoder_targets_embedded, tf.expand_dims(r,1), tf.expand_dims(e,1)])
            #print "decoder_targets",decoder_targets,self._paths[:,2*hop+1]+self._ent_size,self._paths[:,2*hop+2]
            
        
        encoder_cell = tf.contrib.rnn.LSTMCell(32)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, dtype=tf.float32, time_major=False)
        decoder_cell = tf.contrib.rnn.LSTMCell(32)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded,initial_state=encoder_final_state,dtype=tf.float32, time_major=False, scope="plain_decoder")

        decoder_logits = tf.contrib.layers.linear(decoder_outputs,self._kb_size)
        decoder_prediction = tf.argmax(decoder_logits, 2)
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets, depth=self._kb_size, dtype=tf.float32),logits=decoder_logits)

        return stepwise_cross_entropy, decoder_prediction

    def batch_fit(self, queries, paths):
        nexample = queries.shape[0]
        zeros = np.zeros(nexample)
        feed_dict = {self._queries: queries, self._paths: paths, self._zeros: zeros}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, queries, paths):
        nexample = queries.shape[0]
        zeros = np.zeros(nexample)
        feed_dict = {self._queries: queries, self._paths: paths, self._zeros: zeros}
        predict_ = self._sess.run(self._decoder_prediction, feed_dict)
        return predict_

