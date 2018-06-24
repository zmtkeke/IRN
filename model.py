import os
import math
import random
import numpy as np
import tensorflow as tf
from utils import add_gradient_noise,zero_nil_slot,MatrixCos,position_encoding, ScoreRank

class IRN(object):
    def __init__(self, config, sess):
        self._data_file = config.data_file
        self._margin = 4
        self._batch_size = config.batch_size
        self._vocab_size = config.nwords 
        self._rel_size = config.nrels
        self._ent_size = config.nents
        self._sentence_size = config.query_size
        self._embedding_size = config.edim
        self._path_size = config.path_size
        self._memory_size = config.nrels

        self._hops = config.nhop
        self._max_grad_norm = config.max_grad_norm
        self._init = tf.contrib.layers.xavier_initializer()
        #self._init = tf.random_normal_initializer(stddev=config.init_std)

        self._opt = tf.train.AdamOptimizer()
        self._name = "IRN"
        self._checkpoint_dir = config.checkpoint_dir+'/'+self._name

        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        self._build_inputs()
        self._build_vars()
        self._saver = tf.train.Saver(max_to_keep=1)


        self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="encoding")

        KB_batch_loss = self._pretranse()
        KB_loss_op = tf.reduce_sum(KB_batch_loss, name="KB_loss_op")
        KB_grads_and_vars = self._opt.compute_gradients(KB_loss_op,[self.EE,self.RE,self.Mse])
        KB_nil_grads_and_vars = []
        for g, v in KB_grads_and_vars:
            if v.name in self._nil_vars:
                KB_nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                KB_nil_grads_and_vars.append((g, v))
        print "KB_grads_and_vars"
        for g,v in KB_nil_grads_and_vars:
            print g, v.name   
        KB_train_op = self._opt.apply_gradients(KB_grads_and_vars, name="KB_train_op")


        #cross entropy as loss for QA:
        batch_loss, p = self._inference() # (b,1), (batch_size, 5)
        QA_loss_op = tf.reduce_sum(batch_loss, name="QA_loss_op")


        QA_params = [self.QE,self.Mrq,self.Mrs]
        QA_grads_and_vars = self._opt.compute_gradients(QA_loss_op,QA_params)
        
        QA_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in QA_grads_and_vars if g is not None ]
        QA_grads_and_vars = [(add_gradient_noise(g), v) for g,v in QA_grads_and_vars]
        QA_nil_grads_and_vars = []
        for g, v in QA_grads_and_vars:
            if v.name in self._nil_vars:
                QA_nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                QA_nil_grads_and_vars.append((g, v))
               
        print "QA_grads_and_vars"
        for g,v in QA_nil_grads_and_vars:
            print g, v.name
        #grads_and_vars = [(tf.Print(g, [v.name,str(g.get_shape()),g], summarize=1e1/2), v) for g, v in grads_and_vars]

        QA_train_op = self._opt.apply_gradients(QA_nil_grads_and_vars, name="QA_train_op")

        # predict ops
        QA_predict_op = p

        # assign ops
        self.KB_loss_op = KB_loss_op
        self.KB_train_op = KB_train_op
        self.QA_loss_op = QA_loss_op
        self.QA_predict_op = QA_predict_op
        self.QA_train_op = QA_train_op


        init_op = tf.global_variables_initializer()
        self._sess = sess
        self._sess.run(init_op)


    def _build_inputs(self):
        self._KBs = tf.placeholder(tf.int32, [None,3], name="KBs") #_KB
        self._keys = tf.placeholder(tf.int32, [None, self._memory_size],name="keys")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._paths = tf.placeholder(tf.int32, [None, self._path_size], name="paths") #id for e1,r1,e2,r2,a
        self._answers = tf.placeholder(tf.int32, [None, self._ent_size], name="answers") #id-hot for answer
        self._answers_id = tf.placeholder(tf.int32, [None], name="answers_id") #id for answer
        self._paddings = tf.placeholder(tf.int64, [None], name="paddings") #for id_padding
        self._ones = tf.placeholder(tf.float32, [None], name="ones") #for multiple
        self._zeros = tf.placeholder(tf.float32, [None], name="zeros") #for add

        self._istrain = tf.placeholder(tf.int32,name="ground_truth")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            
            nil_word_slot = tf.zeros([1, self._embedding_size])
            nil_rel_slot = tf.zeros([1, self._embedding_size])
            E = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._ent_size-1, self._embedding_size]) ])
            Q = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            R = tf.concat(axis=0, values=[ nil_rel_slot, self._init([self._rel_size-1, self._embedding_size]) ])
            self.EE = tf.Variable(E, name="EE") # encode entity to vector to calculate weight
            self.QE = tf.Variable(Q, name="QE")# encode question-words to vector
            self.RE = tf.Variable(R, name="RE") # encode relation to vector
            self.Mrq = tf.Variable(self._init([self._embedding_size,self._embedding_size]), name="Mrq")
            self.Mrs = tf.Variable(self._init([self._embedding_size,self._embedding_size]), name="Mrs")
            self.Mse = tf.Variable(self._init([self._embedding_size,self._embedding_size]), name="Mse")

            #self.GT = tf.Variable(self._init([self._rel_size,1]), name="GT")

        self._nil_vars = set([self.EE.name, self.QE.name, self.RE.name]) #need to keep first line 0

    def _pretranse(self):
        with tf.variable_scope(self._name):
            h = self._KBs[:,0] #(batch)
            r = self._KBs[:,1] #(batch)
            t = self._KBs[:,2] #(batch)
            tt = self._paddings

            h_emb = tf.nn.embedding_lookup(self.EE, h) #(batch,e)
            r_emb = tf.nn.embedding_lookup(self.RE, r)
            t_emb = tf.nn.embedding_lookup(self.EE, t)
            tt_emb = tf.nn.embedding_lookup(self.EE, tt)
            l_emb = tf.matmul((h_emb+r_emb), self.Mse) #M(h+r)
            s = (l_emb-t_emb)*(l_emb-t_emb)
            ss = (l_emb-tt_emb)*(l_emb-tt_emb)

            loss = self._margin + tf.reduce_sum(s, 1) - tf.reduce_sum(ss, 1)
            loss = tf.maximum(self._zeros,loss)

            return loss


    def _inference(self):
        with tf.variable_scope(self._name):
            #initial
            loss = tf.reshape(self._zeros,[-1,1],name='loss')  #(none,1)
            s_index = tf.reshape(self._paths[:,0],[-1,1]) #(none,1)

            q_emb = tf.nn.embedding_lookup(self.QE, self._queries) #Ax_ij shape is (batch, sentence_size ,embedding_size)
            q = tf.reduce_sum(q_emb, 1) #shape is (batch,embed)

            state = tf.nn.embedding_lookup(self.EE, s_index) #(b,1)->(b,1,e)
            state = tf.squeeze(state,[1]) #(b,e)

            p = s_index

            for hop in range(self._hops):
                step = 2 * hop
                gate = tf.matmul(q, tf.matmul(self.RE, self.Mrq), transpose_b = True) + tf.matmul(state, tf.matmul(self.RE, self.Mrs), transpose_b = True) #(b,e)*(e,14) ->(b,14)
                rel_logits = gate
                r_index = tf.argmax(rel_logits,1)  #(b,)
                
                gate = tf.nn.softmax(gate) #(b,r)
                
                #gumble-softmax: gate is unnormalized logits, 
                #u = tf.random_uniform(shape=tf.shape(gate),minval=0,maxval=1.0) #(b,r)
                #g = -tf.log(-tf.log(u+1e-20)+1e-20)
                #tau = tf.nn.relu(tf.matmul(gate,self.GT))+1e-8 #(batch,1)
                #gate = tf.nn.softmax((gate) / tau) #(batch,v)
                

                real_rel_onehot = tf.one_hot(self._paths[:,step+1], self._rel_size, on_value=1.0, off_value=0.0, axis=-1) #(b,rel_size)
                predict_rel_onehot = tf.one_hot(r_index, self._rel_size, on_value=1.0, off_value=0.0, axis=-1)

                state = state + tf.matmul(gate, tf.matmul(self.RE, self.Mrs))

                loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=rel_logits, labels=real_rel_onehot),[-1,1]) #(b,1)

                q = q - tf.matmul(gate,tf.matmul(self.RE, self.Mrq))

                value = tf.matmul(state, self.Mse)

                ans = tf.matmul(value, self.EE, transpose_b=True) #(b,ent)

                t_index = tf.argmax(ans,1)

                #if r_index == 0, stop inference, ans = previous ans; if not r_index==0, ans = ans
                t_index = tf.cast(t_index,tf.float32)
                r_index = tf.cast(r_index,tf.float32)
                t_index = r_index /(r_index+1e-15) * t_index + (1 - r_index /(r_index+1e-15)) * tf.cast(p[:,-1],tf.float32)

                p = tf.concat(axis=1,values=[p,tf.reshape(tf.cast(r_index,tf.int32),[-1,1])])
                p = tf.concat(axis=1,values=[p,tf.reshape(tf.cast(t_index,tf.int32),[-1,1])])


                real_ans_onehot = tf.one_hot(self._paths[:,step+2], self._ent_size, on_value=1.0, off_value=0.0, axis=-1) #(b,rel_size)


                

                loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=ans, labels=real_ans_onehot),[-1,1]) #(b,1)

            #FOR IRN-weak
            #loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=ans, labels=tf.cast(self._answers, tf.float32)),[-1,1])     

            return loss, p

    def match(self):
        """
        show most similar words_id to each relation embedding
        """
        #self.QE = tf.nn.l2_normalize(self.QE,1)
        #self.RE = tf.nn.l2_normalize(self.RE,1)
        Similar = tf.matmul(tf.matmul(self.RE,self.Mrq), self.QE, transpose_b=True) #(R,e) * (e,E)->(R,E)
        self.match_op = tf.nn.top_k(Similar,k=5)
        _,idx = self._sess.run(self.match_op)
        return idx

    def batch_pretrain(self, KBs, queries, answers, answers_id, paths):
        """
        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, ent_size)
            paths: Tensor

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        nexample = KBs.shape[0]
        keys = np.repeat(np.reshape(np.arange(self._rel_size),[1,-1]),nexample,axis=0) 
        pad = np.random.randint(low = 0, high = self._ent_size, size = nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._keys: keys, self._KBs: KBs, self._queries: queries, self._answers: answers, self._answers_id: answers_id, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros, self._istrain:0}
        loss, _, = self._sess.run([self.KB_loss_op, self.KB_train_op], feed_dict=feed_dict)
        #self.EE = tf.nn.l2_normalize(self.EE,1)
        #self.RE = tf.nn.l2_normalize(self.RE,1)
        return loss

    def batch_fit(self, KBs, queries, answers, answers_id, paths):
        """
        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, ent_size)
            paths: Tensor

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        nexample = queries.shape[0]
        keys = np.repeat(np.reshape(np.arange(self._rel_size),[1,-1]),nexample,axis=0) 
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._keys : keys, self._KBs: KBs, self._queries: queries, self._answers: answers, self._answers_id: answers_id, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros, self._istrain:0}
        loss, _ = self._sess.run([self.QA_loss_op, self.QA_train_op], feed_dict=feed_dict)
        self.EE = tf.nn.l2_normalize(self.EE,1)
        self.RE = tf.nn.l2_normalize(self.RE,1)
        self.QE = tf.nn.l2_normalize(self.QE,1)
        return loss

    def predict(self,KBs, queries, paths):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: id (None, 1)  ,predict_op = max(1, [None,ent_size])
        """
        nexample = queries.shape[0]
        keys = np.repeat(np.reshape(np.arange(self._rel_size),[1,-1]),nexample,axis=0) 
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._keys:keys, self._KBs: KBs, self._queries: queries, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros,self._istrain : 1}
        return self._sess.run(self.QA_predict_op, feed_dict=feed_dict)

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

class IRN_C(object):
    def __init__(self, config, sess):
        self._data_file = config.data_file
        self._margin = 2
        self._batch_size = config.batch_size
        self._vocab_size = config.nwords 
        self._rel_size = config.nrels
        self._ent_size = config.nents
        self._sentence_size = config.query_size
        self._embedding_size = config.edim
        self._path_size = config.path_size
        self._memory_size = config.nrels

        self._hops = config.nhop 
        self._max_grad_norm = config.max_grad_norm
        self._init = tf.contrib.layers.xavier_initializer()
        #self._init = tf.random_normal_initializer(stddev=config.init_std)

        self._opt = tf.train.AdamOptimizer()
        self._name = "IRN_C"
        self._checkpoint_dir = config.checkpoint_dir+'/'+self._name

        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        self._build_inputs()
        self._build_vars()
        self._saver = tf.train.Saver(max_to_keep=10)


        self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="encoding")

        KB_batch_loss = self._pretranse()
        KB_loss_op = tf.reduce_sum(KB_batch_loss, name="KB_loss_op")
        KB_grads_and_vars = self._opt.compute_gradients(KB_loss_op,[self.EE,self.RE,self.Mse])
        KB_nil_grads_and_vars = []
        for g, v in KB_grads_and_vars:
            if v.name in self._nil_vars:
                KB_nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                KB_nil_grads_and_vars.append((g, v))
        print "KB_grads_and_vars"
        for g,v in KB_nil_grads_and_vars:
            print g, v.name   
        KB_train_op = self._opt.apply_gradients(KB_grads_and_vars, name="KB_train_op")
        KBE_norm_op = tf.nn.l2_normalize(self.EE,1)
        KBR_norm_op = tf.nn.l2_normalize(self.RE,1)


        #cross entropy as loss for QA:
        batch_loss_1, p_1, ans_1 = self._inference(self._paths[:,0,:])
        batch_loss_2, p_2, ans_2 = self._inference(self._paths[:,1,:]) 
        QA_loss_op = tf.reduce_sum(batch_loss_1+batch_loss_2, name="QA_loss_op")

        # gradient pipeline, seem not affect much
        QA_grads_and_vars = self._opt.compute_gradients(QA_loss_op)
        
        QA_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in QA_grads_and_vars if g is not None]

        QA_grads_and_vars = [(add_gradient_noise(g), v) for g,v in QA_grads_and_vars]
        QA_nil_grads_and_vars = []
        for g, v in QA_grads_and_vars:
            if v.name in self._nil_vars:
                QA_nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                QA_nil_grads_and_vars.append((g, v))
               
        print "QA_grads_and_vars"
        for g,v in QA_nil_grads_and_vars:
            print g, v.name
        #grads_and_vars = [(tf.Print(g, [v.name,str(g.get_shape()),g], summarize=1e1/2), v) for g, v in grads_and_vars]

        QA_train_op = self._opt.apply_gradients(QA_nil_grads_and_vars, name="QA_train_op")
        fans = ans_1+ans_2
        final_ans = tf.reshape(tf.cast(tf.argmax(fans,1),tf.int32),[-1,1])

        # predict ops
        QA_predict_op = tf.concat(axis=1,values=[p_1,p_2,final_ans]) #(none,11)


        # assign ops
        self.KB_loss_op = KB_loss_op
        self.KB_train_op = KB_train_op
        self.KBE_norm_op = KBE_norm_op
        self.KBR_norm_op = KBR_norm_op
        self.QA_loss_op = QA_loss_op
        self.QA_predict_op = QA_predict_op
        self.QA_train_op = QA_train_op


        init_op = tf.global_variables_initializer()
        self._sess = sess
        self._sess.run(init_op)


    def _build_inputs(self):
        self._KBs = tf.placeholder(tf.int32, [None,3], name="KBs") #_KB
        self._keys = tf.placeholder(tf.int32, [None, self._memory_size],name="keys")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._paths = tf.placeholder(tf.int32, [None, 2, self._path_size], name="paths") #id for [e1,r1,t, e2,r2,t]
        self._answers = tf.placeholder(tf.int32, [None, self._ent_size], name="answers") #id-hot for answer
        self._answers_id = tf.placeholder(tf.int32, [None], name="answers_id") #id for answer
        self._paddings = tf.placeholder(tf.int64, [None], name="paddings") #for id_padding
        self._ones = tf.placeholder(tf.float32, [None], name="paddings") #for multiple
        self._zeros = tf.placeholder(tf.float32, [None], name="paddings") #for add

        self._istrain = tf.placeholder(tf.int32,name="ground_truth")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            
            nil_word_slot = tf.zeros([1, self._embedding_size])
            nil_rel_slot = tf.zeros([1, self._embedding_size])
            E = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._ent_size-1, self._embedding_size]) ])
            Q = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            R = tf.concat(axis=0, values=[ nil_rel_slot, self._init([self._rel_size-1, self._embedding_size]) ])
            self.EE = tf.Variable(E, name="EE") # encode entity to vector to calculate weight
            self.QE = tf.Variable(Q, name="QE")# encode question-words to vector
            self.RE = tf.Variable(R, name="RE") # encode relation to vector
            #self.RE = self.QE[:self._rel_size]
            self.Mrq = tf.Variable(self._init([self._embedding_size,self._embedding_size]), name="Mrq")
            self.Mrs = tf.Variable(self._init([self._embedding_size,self._embedding_size]), name="Mrs")
            self.Mse = tf.Variable(self._init([self._embedding_size,self._embedding_size]), name="Mse")

            #self.GT = tf.Variable(self._init([self._rel_size,1]), name="GT")

        self._nil_vars = set([self.EE.name, self.QE.name, self.RE.name]) #need to keep first line 0

    def _pretranse(self):
        with tf.variable_scope(self._name):
            h = self._KBs[:,0] #(batch)
            r = self._KBs[:,1] #(batch)
            t = self._KBs[:,2] #(batch)
            tt = self._paddings

            h_emb = tf.nn.embedding_lookup(self.EE, h) #(batch,e)
            r_emb = tf.nn.embedding_lookup(self.RE, r)
            t_emb = tf.nn.embedding_lookup(self.EE, t)
            tt_emb = tf.nn.embedding_lookup(self.EE, tt)
            l_emb = tf.matmul((h_emb+r_emb), self.Mse) #M(h+r)
            s = (l_emb-t_emb)*(l_emb-t_emb)
            ss = (l_emb-tt_emb)*(l_emb-tt_emb)

            loss = self._margin + tf.reduce_sum(s, 1) - tf.reduce_sum(ss, 1)
            loss = tf.maximum(self._zeros,loss)

            return loss


    def _inference(self, _paths):
        with tf.variable_scope(self._name):
            #initial
            loss = tf.reshape(self._zeros,[-1,1],name='loss')  #(none,1)
            s_index = tf.reshape(_paths[:,0],[-1,1]) #(none,1)

            q_emb = tf.nn.embedding_lookup(self.QE, self._queries) #Ax_ij shape is (batch, sentence_size ,embedding_size)
            q = tf.reduce_sum(q_emb, 1) #shape is (batch,embed)

            state = tf.nn.embedding_lookup(self.EE, s_index) #(b,1)->(b,1,e)
            state = tf.squeeze(state,[1]) #(b,e)

            p = s_index


            for hop in range(self._hops):
                gate = tf.matmul(q, tf.matmul(self.RE, self.Mrq), transpose_b = True) + tf.matmul(state, tf.matmul(self.RE, self.Mrs), transpose_b = True)
                #gate = tf.matmul(q, self.RE, transpose_b = True) + tf.matmul(state, self.RE, transpose_b = True) #(b,e)*(e,14) ->(b,14)
                rel_logits = gate
                r_index = tf.cast(tf.argmax(rel_logits,1),tf.int32)  #(b,)
                gate = tf.nn.softmax(gate)
                
                #gumble-softmax: gate is unnormalized logits, 
                #u = tf.random_uniform(shape=tf.shape(gate),minval=0,maxval=1.0) #(b,r)
                #g = -tf.log(-tf.log(u+1e-20)+1e-20)
                #tau = tf.nn.relu(tf.matmul(gate,self.GT))+1e-8 #(batch,1)
                #gate = tf.nn.softmax((gate+g) / tau) #(batch,v)

                real_rel_onehot = tf.one_hot(_paths[:,2*hop+1], self._rel_size, on_value=1.0, off_value=0.0, axis=-1) #(b,rel_size)
                predict_rel_onehot = tf.one_hot(r_index, self._rel_size, on_value=1.0, off_value=0.0, axis=-1)


                #correct wrong ans
                '''
                train_state = state + tf.matmul(real_rel_onehot, tf.matmul(self.RE, self.Mrs)) #(b,14)*(14,e) (avg with weights) -> (b,e)
                test_state = state + tf.matmul(predict_rel_onehot, tf.matmul(self.RE, self.Mrs)) #(b,14)*(14,e) (avg with weights) -> (b,e)
                state = tf.cond(tf.equal(self._istrain,tf.constant(0)),lambda:train_state,lambda:test_state)
                '''

                state = state + tf.matmul(gate, tf.matmul(self.RE, self.Mrs))
                #state = tf.nn.l2_normalize(state,1)
    
                loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=rel_logits, labels=real_rel_onehot),[-1,1]) #(b,1)
                
                #correct wrong ans
                '''
                train_q = q - tf.matmul(tf.nn.embedding_lookup(self.RE, _paths[:,2*hop+1]), self.Mrq)
                test_q  = q - tf.matmul(tf.nn.embedding_lookup(self.RE, r_index), self.Mrq)
                q = tf.cond(tf.equal(self._istrain,tf.constant(0)),lambda:train_q,lambda:test_q)
                '''

                q = q - tf.matmul(gate,tf.matmul(self.RE, self.Mrq))
                

                value = tf.matmul(state,self.Mse)
                ans = tf.matmul(value, self.EE, transpose_b=True) #(b,ent)
                t_index = tf.cast(tf.argmax(ans,1),tf.int32)

                p = tf.concat(axis=1,values=[p,tf.reshape(r_index,[-1,1])])
                p = tf.concat(axis=1,values=[p,tf.reshape(t_index,[-1,1])])

                real_ans_onehot = tf.one_hot(_paths[:,2*hop+2], self._ent_size, on_value=1.0, off_value=0.0, axis=-1) #(b,rel_size)

                loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=ans, labels=real_ans_onehot),[-1,1]) #(b,1)

            #loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=ans, labels=tf.cast(self._answers, tf.float32)),[-1,1])     

            return loss, p, ans

    def batch_pretrain(self, KBs, queries, answers, answers_id, paths):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, ent_size)
            paths: Tensor

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        nexample = KBs.shape[0]
        keys = np.repeat(np.reshape(np.arange(self._rel_size),[1,-1]),nexample,axis=0) 
        pad = np.random.randint(low = 0, high = self._ent_size, size = nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._keys: keys, self._KBs: KBs, self._queries: queries, self._answers: answers, self._answers_id: answers_id, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros, self._istrain :0}
        loss, _, _, _ = self._sess.run([self.KB_loss_op, self.KB_train_op, self.KBE_norm_op, self.KBR_norm_op], feed_dict=feed_dict)
        return loss

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
        keys = np.repeat(np.reshape(np.arange(self._rel_size),[1,-1]),nexample,axis=0) 
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._keys : keys, self._KBs: KBs, self._queries: queries, self._answers: answers, self._answers_id: answers_id, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros, self._istrain :0}
        loss, _ = self._sess.run([self.QA_loss_op, self.QA_train_op], feed_dict=feed_dict)
        return loss

    def predict(self,KBs, queries, paths):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, 3)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: id (None, 1)  ,predict_op = max(1, [None,ent_size])
        """
        nexample = queries.shape[0]
        keys = np.repeat(np.reshape(np.arange(self._rel_size),[1,-1]),nexample,axis=0) 
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)
        feed_dict = {self._keys:keys, self._KBs: KBs, self._queries: queries, self._paths: paths, self._paddings: pad, self._ones: ones, self._zeros: zeros, self._istrain :1}
        return self._sess.run(self.QA_predict_op, feed_dict=feed_dict)

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

