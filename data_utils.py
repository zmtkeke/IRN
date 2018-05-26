from __future__ import absolute_import

import os
import re
import numpy as np
from collections import Counter

#process Path-QA or Conj-QA data&KB

# kb: h \t r \t t

# form: question \t  ans  \t  e1#r1#e2#r2#e3#<end>#e3  \t   ans1/ans2/   \t   e1#r1#e2///e2#r2#e3#///s#r#t///s#r#t

# form: question \t  ans  \t  e1#r1#ans#<end>#ans*e2#r2#ans#<end>#ans  \t   ans1/   \t   e1#r1#e2///e2#r2#e3#///s#r#t///s#r#t  \t   e1/e2

# form: question \t ans \t e1#r1#e2#rc2#ec2#r2#e3#rc3#ec3#<end>#e3#<end>#e3 \t  ans1/   \t   e1#r1#e2///e2#r2#e3#///s#r#t///s#r#t

def process_data_c(KB_file, data_file, word2id, rel2id, ent2id, words, relations, entities):#relations is set, other is list(), *2id is dict()
    read_KB(KB_file, entities, relations)
    data,sentence_size,memory_size = read_data(data_file, words)

    #set ids
    if len(word2id)==0:
        word2id['<unk>'] = 0
    if len(rel2id)==0:
        rel2id['<end>'] = 0
    if len(ent2id)==0:
        ent2id['<unk>'] = 0

    for r in relations:
        # same r_id in rel2id and word2id
        if not rel2id.has_key(r):
            rel2id[r] = len(rel2id)
        if not word2id.has_key(r):
            word2id[r] = len(word2id)
    for e in entities:
        if not ent2id.has_key(e):
            ent2id[e] = len(ent2id)
    for word in words:
        if not word2id.has_key(word):
            word2id[word] = len(word2id)
    
    print ('here are %d words in word2id(vocab)' %len(word2id))  #75080
    print ('here are %d relations in rel2id(rel_vocab)' %len(rel2id)) #13+1
    print ('here are %d entities in ent2id(ent_vocab)' %len(ent2id)) #13+1

    Triples, KBs, tails_size = get_KB(KB_file,ent2id,rel2id)

    print "#records or Triples", len(np.nonzero(KBs)[0])



    Q = []
    QQ = []
    A = []
    AA = []
    P = []
    PP = []
    S = []
    SS = []
    D = []
    DD = []

    for query, answer, path, answerset, subgraph, subject in data:

        query = query.strip().split()
        ls = max(0, sentence_size-len(query))
        q = [word2id[w] for w in query] + [0] * ls
        Q.append(q)
        QQ.append(query)

        a = np.zeros(len(ent2id)) # if use new ans-vocab, add 0 for 'end'
        a[ent2id[answer]] = 1
        A.append(a)
        AA.append(ent2id[answer])

        #p = [[ent2id[],rel2id[],ent2id[],rel2id[],ent2id[]], [], []]
        # POSITION+'#'+"plays_position_inverse"+'#'+PLAYER+'*'+CLUB+'#'+"plays_in_club_inverse"+'#'+PLAYER
        path = path.strip().split('*') #path = [POSITION+'#'+"plays_position_inverse"+'#'+PLAYER, CLUB+'#'+"plays_in_club_inverse"+'#'+PLAYER]
        p=[]
        for subpath in path:
            subpath = subpath.split("#")
            p.append([ent2id[subpath[0]], rel2id[subpath[1]], ent2id[subpath[2]],rel2id[subpath[3]],ent2id[subpath[4]]])
        P.append(p)  #N*2*3
        PP.append(path)

        
        sg = []
        subgraph = subgraph.split('///') #subgraph is a list including many triple-str  for memn2n-t
        #isubgraph=list(set(subgraph.replace('///','#').split('#')))
        ls = max(0, memory_size-len(subgraph))   
 
        b = 0
        for t in subgraph:
            t = t.split('#')
            if not len(t)==3:
                print "subgraph not a triple form!"
                print t
            tt = [ent2id[t[0]],rel2id[t[1]],ent2id[t[2]]]

            if not tt in Triples.tolist():
                b += 1
                continue

            sg.append(tt)

        '''
        for t in isubgraph:
            sg.append([ent2id[t]])
        '''

        for i in range(ls):
            #sg.append( [0,0,0] )
            sg.append([0])
        D.append(sg)
        DD.append(subgraph)
        
        anset = answerset.split('/')
        anset = anset[:-1]
        ass=[]
        for a in anset:
            ass.append(ent2id[a])
        S.append(ass)
        SS.append(anset)

    return np.array(Q),np.array(A),np.array(P),np.array(D),np.array(S),QQ,AA,PP,DD,SS,Triples,KBs,sentence_size,memory_size, tails_size

def process_data(KB_file, data_file, word2id, rel2id, ent2id, words, relations, entities): #relations is set, other is list(), *2id is dict()
    read_KB(KB_file, entities, relations)
    data,sentence_size,memory_size = read_data(data_file, words)

    #set ids
    if len(word2id)==0:
        word2id['<unk>'] = 0
    if len(rel2id)==0:
        rel2id['<end>'] = 0
    if len(ent2id)==0:
        ent2id['<unk>'] = 0

    for r in relations:
        # same r_id in rel2id and word2id
        if not rel2id.has_key(r):
            rel2id[r] = len(rel2id)
        if not word2id.has_key(r):
            word2id[r] = len(word2id)
    for e in entities:
        if not ent2id.has_key(e):
            ent2id[e] = len(ent2id)
    for word in words:
        if not word2id.has_key(word):
            word2id[word] = len(word2id)

    print ('here are %d words in word2id(vocab)' %len(word2id))  #75080
    print ('here are %d relations in rel2id(rel_vocab)' %len(rel2id)) #13+1
    print ('here are %d entities in ent2id(ent_vocab)' %len(ent2id)) #13+1

    Triples, KBs,tails_size = get_KB(KB_file,ent2id,rel2id)

    print "#records or Triples", len(np.nonzero(KBs)[0])



    Q = []
    QQ = []
    A = []
    AA = []
    P = []
    PP = []
    S = []
    SS = []
    D = []
    DD = []

    for query, answer, path, answerset, subgraph in data:
        path = path.strip().split('#') #path = [s,r1,m,r2,t]
        #answer = path[-1]

        query = query.strip().split()
        ls = max(0, sentence_size-len(query))
        q = [word2id[w] for w in query] + [0] * ls
        Q.append(q)
        QQ.append(query)

        a = np.zeros(len(ent2id)) # if use new ans-vocab, add 0 for 'end'
        a[ent2id[answer]] = 1
        A.append(a)
        AA.append(ent2id[answer])

        #p = [ ent2id[path[0]], rel2id[path[1]], ent2id[path[2]], rel2id[path[3]], ent2id[path[4]] ]
        
        p=[]
        for i in range(len(path)):
            if i % 2 == 0:
                e = ent2id[path[i]]
               # e = np.zeros(len(relations))
               # e[0] = ent2id[path[i]]
                p.append(e)
            else:
                r = rel2id[path[i]]
               # r = np.zeros(len(relations))
               # r[rel2id[path[i]]] =1
                p.append(r)
        
        #p.append(rel2id[path[3]])
        #p.append(ent2id[path[4]])
        P.append(p)
        PP.append(path)

        
        sg = []
        subgraph = subgraph.split('///') #subgraph is a list including many triple-str  for memn2n-t |||| 3 in this function & 1 in read-doc
        #isubgraph=list(set(subgraph.replace('///','#').split('#')))
        ls = max(0, memory_size-len(subgraph))
        
        
        for t in subgraph:
            t = t.split('#')
            if not len(t)==3:
                print "subgraph not a triple form!"
                print t
            tt = [ent2id[t[0]],rel2id[t[1]],ent2id[t[2]]]

            if not tt in Triples.tolist():
                ls += 1  #add padding
                continue

            sg.append(tt)

        '''
        for t in isubgraph:
            sg.append([ent2id[t],0,0])
        '''

        for i in range(ls):
            sg.append( [0,0,0] )
        D.append(sg)
        DD.append(subgraph)
        
        anset = answerset.split('/')
        anset = anset[:-1]
        ass=[]
        for a in anset:
            ass.append(ent2id[a])
        S.append(ass)
        SS.append(anset)



   # return Q,A,P,D,QQ,AA,PP,DD,KBs,sentence_size,memory_size,tails_size
    return np.array(Q),np.array(A),np.array(P),np.array(D),np.array(S),QQ,AA,PP,DD,SS,Triples,KBs,sentence_size,memory_size, tails_size



def read_KB(KB_file, entities, relations):
    #example in KB_file: KBs.txt h \t r \t t

    if os.path.isfile(KB_file):
        with open(KB_file) as f:
            lines = f.readlines()
    else:
        raise Exception("!! %s is not found!!" % KB_file)

    for line in lines:
        line = line.strip().split('\t')
        entities.add(line[0])
        entities.add(line[2])
        relations.add(line[1])


def get_KB(KB_file,ent2id,rel2id):
    nwords = len(ent2id)
    nrels = len(rel2id)
    tails = np.zeros([nwords*nrels,1], 'int32')
    #KBmatrix = np.zeros([nwords, nrels,nwords], 'int32')
    KBmatrix = np.zeros([nwords * nrels,nwords], 'int32')
    Triples = []

    f = open(KB_file)
    control = 1
    b = 0
    for line in f.readlines():
        line = line.strip().split('\t')

        '''  delete half triples
        control += 1
        if control % 2 == 0:
            b += 1
            continue
        '''

        h = ent2id[line[0]]
        r = rel2id[line[1]]
        t = ent2id[line[2]]
        Triples.append([h,r,t])
        #[h,r]->[h*nrels+r]
        lenlist = tails[h*nrels+r]
        KBmatrix[h*nrels+r,lenlist] = t
        tails[h*nrels+r]+=1

    print "delete triples:", b

    return np.array(Triples), KBmatrix[:,:np.max(tails)], np.max(tails)



def read_data(data_file, words):
    #example in data_file: WC-C: q+'\t'+ans+'\t'+p+'\t'+ansset+'\t'+c+'\t'+sub+'\n'
    
    if os.path.isfile(data_file):
        with open(data_file) as f:
            lines = f.readlines()
    else:
        raise Exception("!! %s is not found!!" % data_file)

    data = []
    questions = []
    doc = []

    for line in lines:
        line = line.strip().split('\t')
        data.append(line)
        for w in line[0].strip().split():
            words.add(w)
        questions.append(line[0].strip().split())
        doc.append(line[4].strip().split('///')) #for memn2n-triple
        #doc.append(list(set(line[4].strip().replace('///','#').split('#'))))  #for memn2n-entity

    sentence_size = max(len(i) for i in questions)
    memory_size = max(len(i) for i in doc)

    return data, sentence_size , memory_size





def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

