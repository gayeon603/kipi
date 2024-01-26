import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import scipy.sparse as sp
from print_hook import PrintHook
import datetime
from time import time
from setproctitle import setproctitle

print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_logical_devices('GPU')

if len(gpus) > 1: 
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('\n\n Running on multiple GPUs ', [gpu.name for gpu in gpus])

class Controller(tf.keras.Model):
    def __init__(self, dim1):
        super(Controller, self).__init__()

        self.linear1 = tf.keras.layers.Dense(64, activation='relu')
        self.linear2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
        self.linear1.build(input_shape=(None, dim1))
        self.linear2.build(input_shape=(None, 64))

        self.call(tf.keras.Input(shape=(dim1,)))  # Initialize the weights

    def call(self, x):
        z1 = self.linear1(x)
        res = self.linear2(z1)
        #res = tf.nn.softmax(res)
        return res

class Recommender:
    def __init__(self, sess, handler):
        self.sess = sess
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        self.weights = self._init_weights()
        self.behEmbeds = NNs.defineParam('behEmbeds', [args.behNum, args.latdim // 2])
        mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR20', 'NDCG20', 'HR25', 'NDCG25', 'HR30', 'NDCG30', 'HR35', 'NDCG35', 'HR100', 'NDCG100']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):  
            
        self.controller = Controller(args.latdim // 2 * 9)
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            init = tf.compat.v1.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        train_time = 0
        test_time = 0
            
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            t0 = time()
            reses = self.trainEpoch(self.controller)
            t1 = time()
            train_time += t1-t0
            print('Train_time',t1-t0,'Total_time',train_time)
            log(self.makePrint('Train', ep, reses, test))
            if test:                  
                t2 = time()
                reses = self.testEpoch()
                t3 = time()
                test_time += t3-t2
                print('Test_time',t3-t2,'Total_time',test_time)
                log(self.makePrint('Test', ep, reses, test))                                                                    
            if ep % args.tstEpoch == 0:
                self.saveHistory()
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()


    def messagePropagate(self, lats, adj):
        return Activate(tf.sparse.sparse_dense_matmul(adj, lats), self.actFunc)


    def defineModel(self):
        uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim // 2], reg=True) 
        iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim // 2], initializer = np.load('Datasets/' + args.data + '/item_embedding.npy'), reg=True) 

        allEmbed = tf.concat([uEmbed0, iEmbed0], axis = 0) 
        self.ulat = [0] * (args.behNum)
        self.ilat = [0] * (args.behNum)
        for beh in range(args.behNum):
            self.ego_embeddings=allEmbed
            ego_embeddings = allEmbed
            all_embeddings = [ego_embeddings]
            if args.multi_graph == False:
                for index in range(args.gnn_layer):
                    symm_embeddings = tf.sparse.sparse_dense_matmul(self.adjs[beh], all_embeddings[-1])
                    if args.encoder == 'lightgcn':
                        lightgcn_embeddings = symm_embeddings
                        all_embeddings.append(lightgcn_embeddings) 
                    elif args.encoder == 'gccf':
                        gccf_embeddings = Activate(symm_embeddings, self.actFunc)
                        all_embeddings.append(gccf_embeddings)
                    elif args.encoder == 'gcn':
                        gcn_embeddings = Activate(
                            tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights[
                                'b_gc_%d' % index], self.actFunc)
                        all_embeddings.append(gcn_embeddings)
                    elif args.encoder == 'ngcf':
                        gcn_embeddings = Activate(
                            tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights[
                                'b_gc_%d' % index], self.actFunc)
                        bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                        bi_embeddings = Activate(
                            tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index],
                            self.actFunc)
                        all_embeddings.append(gcn_embeddings + bi_embeddings)
                    
            elif args.multi_graph == True:
                for index in range(args.gnn_layer):
                    if index == 0:
                        symm_embeddings = tf.sparse.sparse_dense_matmul(self.adjs[beh], all_embeddings[-1]) 
                        if args.encoder == 'lightgcn':
                            lightgcn_embeddings = symm_embeddings
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gccf':
                            gccf_embeddings = Activate(symm_embeddings, self.actFunc)
                            all_embeddings.append(gccf_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gcn':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'ngcf':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                            bi_embeddings = Activate(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + bi_embeddings + all_embeddings[-1])
                    else:
                        atten = FC(ego_embeddings, args.behNum, reg=True, useBias=True,
                                   activation=self.actFunc, name='attention_f%d_%d'%(beh,index), reuse=True)
                        temp_embeddings = []
                        for inner_beh in range(args.behNum):
                            neighbor_embeddings = tf.sparse.sparse_dense_matmul(self.adjs[inner_beh], symm_embeddings)
                            temp_embeddings.append(neighbor_embeddings)
                        all_temp_embeddings = tf.stack(temp_embeddings, 1)
                        symm_embeddings = tf.reduce_sum(tf.einsum('abc,ab->abc', all_temp_embeddings, atten), axis=1, keepdims=False)
                        if args.encoder == 'lightgcn':
                            lightgcn_embeddings = symm_embeddings
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gccf':
                            gccf_embeddings = Activate(symm_embeddings, self.actFunc)
                            all_embeddings.append(gccf_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gcn':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'ngcf':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                            bi_embeddings = Activate(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + bi_embeddings + all_embeddings[-1])

            all_embeddings = tf.add_n(all_embeddings) # shape : [#user+#item, dim]
            self.ulat[beh], self.ilat[beh] = tf.split(all_embeddings, [args.user, args.item], 0)
        self.ulat_merge, self.ilat_merge = tf.add_n(self.ulat), tf.add_n(self.ilat)


    def _init_weights(self):
        all_weights = dict()
        initializer = tf.compat.v1.random_normal_initializer(stddev=0.01)  

        self.weight_size_list = [args.latdim // 2] + [args.latdim // 2] * args.gnn_layer

        for k in range(args.gnn_layer):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

  

    def pairwise_predict(self, src, cluster_center_emb, cluster_emb_2, cluster_emb_3):
        neg = self.neg_ids[src]
        batIds = self.batids 
        batch_item_id = self.batch_item_ids[src] 
        
        user_emb_ego, pos_emb_ego, neg_emb_ego, user_emb, pos_emb, neg_emb, concept1_pos, concept1_neg,  concept2_pos, concept2_neg, concept3_pos, concept3_neg, uni_center = \
                self.concept_level_embeddings(batIds, batch_item_id, neg, self.cluster_ids, cluster_center_emb, cluster_emb_2, cluster_emb_3, src)
        state = self.concept_aware_state(user_emb_ego, pos_emb_ego, neg_emb_ego, concept1_pos, concept1_neg, concept2_pos, concept2_neg, concept3_pos, concept3_neg, 
                                                uni_center) 
        loss_weight = self.controller(state)
        self.loss_weight[src] = loss_weight

    
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)
        src_emb = tf.concat([src_ulat, src_ilat], axis=-1)
        
        
        behavior_pairs=[]
        for index in range(4):
            name=[src,index]
            name.sort()
            if index !=src:
                beh_emb=tf.concat([tf.nn.embedding_lookup(self.ulat[index], uids), tf.nn.embedding_lookup(self.ilat[index], iids)], axis=-1)
            else:
                continue
            beh_emb=tf.concat([tf.nn.embedding_lookup(self.ulat[index], uids), tf.nn.embedding_lookup(self.ilat[index], iids)], axis=-1)
            bp_network = FC(tf.concat([src_emb, beh_emb], axis=-1), args.latdim, reg=True, useBias=True,
                                activation='softmax', name='pairwise_'+str(name[0])+str(name[1]), reuse=True)

            behavior_pairs.append(bp_network)
      
        behavior_pairs_concat = tf.stack(behavior_pairs, axis=2) 
        behavior_pairs_concat=tf.reduce_sum(behavior_pairs_concat, axis=-1) 
    
        gate_out = FC(behavior_pairs_concat, args.behNum, reg=True, useBias=True,
                          activation='softmax', name='gate_softmax_' + str(src), reuse=True)   
        w1 = tf.reshape(gate_out, [-1, args.behNum, 1])
                  
            
        score_info = []
        for index in range(args.behNum):
            score_info.append(
                tf.nn.embedding_lookup(self.ulat[index], uids) * tf.nn.embedding_lookup(self.ilat[index], iids))
        predEmbed = tf.stack(score_info, axis=2)
        out = tf.reshape(predEmbed @ w1, [-1, args.latdim // 2])

        preds = tf.squeeze(tf.reduce_sum(out, axis=-1))

        return preds * args.mult
    
    
    def create_multiple_adj_mat(self, adj_mat):
        def left_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate left_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def right_adj_single(adj):
            rowsum = np.array(adj.sum(0))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = adj.dot(d_mat_inv)
            print('generate right_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def symm_adj_single(adj_mat):
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            rowsum = np.array(adj_mat.sum(0))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv_trans = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv_trans)
            print('generate symm_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        left_adj_mat = left_adj_single(adj_mat)
        right_adj_mat = right_adj_single(adj_mat)
        symm_adj_mat = symm_adj_single(adj_mat)

        return left_adj_mat.tocsr(), right_adj_mat.tocsr(), symm_adj_mat.tocsr()

    def mult(self,a,b):
        return (a*b).sum(1)
    
    def coef(self,buy_mat,view_mat):
        buy_dense = np.array(buy_mat.todense())
        view_dense = np.array(view_mat.todense())
        buy = buy_dense-buy_dense.sum(1).reshape(-1,1)/buy_dense.shape[1]
        view = view_dense-view_dense.sum(1).reshape(-1,1)/view_dense.shape[1]
        return self.mult(buy,view)/np.sqrt((self.mult(buy,buy))*self.mult(view,view))
      
    def mult_tmall(self,a,b):
        return a.multiply(b).sum(1)
    
    def coef_tmall(self,buy_mat,view_mat):
        buy = buy_mat
        view = view_mat
        return np.array(self.mult_tmall(buy,view))/np.sqrt(np.array(self.mult_tmall(buy,buy))*np.array(self.mult_tmall(view,view)))
    
    def prepareModel(self):
        self.actFunc = 'leakyRelu'
        self.adjs = []
        self.uids, self.iids = [], []
        self.batids, self.batch_item_ids, self.neg_ids = [], [], []
        self.batch_label_tensor = []
        self.batch_label = []
        self.loss_weight = {}
        self.left_trnMats, self.right_trnMats, self.symm_trnMats, self.none_trnMats = [], [], [], []
        self.cluster_ids = []

        for i in range(args.behNum):
            R = self.handler.trnMats[i].tolil() # #User X #Item sparse matrix
            
            coomat = sp.coo_matrix(R)
            coomat_t = sp.coo_matrix(R.T)
            row = np.concatenate([coomat.row, coomat_t.row + R.shape[0]])
            col = np.concatenate([R.shape[0] + coomat.col, coomat_t.col])
            data = np.concatenate([coomat.data.astype(np.float32), coomat_t.data.astype(np.float32)])
            adj_mat = sp.coo_matrix((data, (row, col)), shape=(args.user + args.item, args.user + args.item))

            
            left_trn, right_trn, symm_trn = self.create_multiple_adj_mat(adj_mat)
            self.left_trnMats.append(left_trn)
            self.right_trnMats.append(right_trn)
            self.symm_trnMats.append(symm_trn)
            self.none_trnMats.append(adj_mat.tocsr())
        if args.normalization == "left":
            self.final_trnMats = self.left_trnMats
        elif args.normalization == "right":
            self.final_trnMats = self.right_trnMats
        elif args.normalization == "symm":
            self.final_trnMats = self.symm_trnMats
        elif args.normalization == 'none':
            self.final_trnMats = self.none_trnMats
        self.batch_label = self.handler.trnMats

        self.batids = (tf.compat.v1.placeholder(name='batids', dtype=tf.int32, shape=[None]))
        
        
        for i in range(args.behNum):
            adj = self.final_trnMats[i]
            idx, data, shape = transToLsts(adj, norm=False)
            batch_idx, batch_data, batch_shape = transToLsts(self.batch_label[i], norm=False)
            self.adjs.append(tf.sparse.SparseTensor(idx, data, shape)) 
            self.batch_label_tensor.append(tf.sparse.SparseTensor(batch_idx, batch_data, batch_shape))
            self.uids.append(tf.compat.v1.placeholder(name='uids' + str(i), dtype=tf.int32, shape=[None]))
            self.iids.append(tf.compat.v1.placeholder(name='iids' + str(i), dtype=tf.int32, shape=[None]))
            self.batch_item_ids.append(tf.compat.v1.placeholder(name='batch_item_ids' + str(i), dtype=tf.int32, shape=[None]))
            self.neg_ids.append(tf.compat.v1.placeholder(name='neg_ids' + str(i), dtype=tf.int32, shape=[None]))
            
            
  
        self.defineModel()
        
        with open('Datasets/' + args.data + '/cluster_ids', 'rb') as fs:
            data = pickle.load(fs)
            self.cluster_ids = data
        self.cluster_ids = tf.convert_to_tensor(self.cluster_ids)
        cluster_center_emb, cluster_emb_1, cluster_emb_2, cluster_emb_3 = self.compute_cluster_center(self.cluster_ids)

        self.preLoss = 0
  
        for src in range(args.behNum):
            if args.decoder == 'pairwise':
                preds = self.pairwise_predict(src, cluster_emb_1, cluster_emb_2, cluster_emb_3)

            sampNum = tf.shape(self.uids[src])[0] // 2
            posPred = tf.slice(preds, [0], [sampNum]) 
            negPred = tf.slice(preds, [sampNum], [-1]) 
        
            
            if not self.loss_weight:
                self.preLoss += tf.reduce_mean(tf.nn.softplus(-(posPred - negPred)))
            else:
                self.preLoss += tf.reduce_mean(tf.nn.softplus(-(posPred - negPred))* self.loss_weight[src])

            if src == args.behNum - 1:
                self.targetPreds = preds
        self.regLoss = args.reg * Regularize()
        
        self.loss = self.preLoss + self.regLoss
        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.compat.v1.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray() 
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum 
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        batch_item_id = []
        neg_samples = []
        cur = 0
        for i in range(batch): 
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
                batch_item_id.append(poslocs[0])
                neg_samples.append(neglocs[0])
            else:
                poslocs = np.random.choice(posset, sampNum)
                batch_item_id.append(poslocs[0])
                neglocs = negSamp(temLabel[i], sampNum, args.item)
                neg_samples.append(neglocs[0])
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs, batch_item_id, neg_samples

    def trainEpoch(self, cont): 
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum] 
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))
        for i in range(steps): 
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            self.batch_label = []
            batIds = sfIds[st: ed]

            target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
            
            feed_dict = {}
            feed_dict[self.batids] = batIds
            
            for beh in range(args.behNum):
                uLocs, iLocs, batch_item_id, neg_samples = self.sampleTrainBatch(batIds, self.handler.trnMats[beh])
                neg = np.squeeze(neg_samples)
                
                self.batch_label.append(self.handler.trnMats[beh][batIds, :])

                feed_dict[self.uids[beh]] = uLocs
                feed_dict[self.iids[beh]] = iLocs
                feed_dict[self.batch_item_ids[beh]] = batch_item_id
                feed_dict[self.neg_ids[beh]] = neg
                
            res = self.sess.run(target, feed_dict=feed_dict,
                                options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

            preLoss, regLoss, loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss
 
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret
    
    def compute_cluster_center(self, cluster_ids):
        cluster_center_emb = None
        for i in range(78):
            index = tf.compat.v1.where(tf.equal(cluster_ids, i))[:,0]

            tmp_emb = tf.math.reduce_mean(tf.gather(self.ilat_merge, index), axis=0, keepdims=True)
            cluster_center_emb = tmp_emb if cluster_center_emb == None else tf.concat([cluster_center_emb, tmp_emb], 0)
        

        concept_embedding=np.load('Datasets/'+ args.data + '/concept_embedding_1.npy')
        cluster_emb_1=concept_embedding
        concept_embedding=np.load('Datasets/'+ args.data + '/concept_embedding_2.npy')
        cluster_emb_2=concept_embedding
        concept_embedding=np.load('Datasets/'+ args.data + '/concept_embedding_3.npy')
        cluster_emb_3=concept_embedding
        
        return cluster_center_emb, cluster_emb_1, cluster_emb_2, cluster_emb_3
    
    def concept_level_embeddings(self, user, pos, neg, cluster_ids, cluster_center_emb, cluster_emb_2, cluster_emb_3, beh):
        
        u_g_embeddings, i_g_embeddings = self.ulat[beh], self.ilat[beh]
        user_id = user
        pos_id = tf.convert_to_tensor(pos)
        neg_id = tf.convert_to_tensor(neg)

        user_emb, pos_emb, neg_emb = tf.gather(u_g_embeddings, user_id), tf.gather(i_g_embeddings, pos_id), tf.gather(i_g_embeddings, neg_id)

        with open('Datasets/'+ args.data +'/cluster_ids_2', 'rb') as fs:
            cluster_ids_2 = pickle.load(fs)

        with open('Datasets/'+ args.data +'/cluster_ids_3', 'rb') as fs:
            cluster_ids_3 = pickle.load(fs)

   
        pos_item_cluster_id2 = tf.nn.embedding_lookup(list(cluster_ids_2.values()), pos)
        neg_item_cluster_id2 = tf.nn.embedding_lookup(list(cluster_ids_2.values()), neg)
        concept2_pos = tf.math.reduce_mean(tf.gather(cluster_emb_2, pos_item_cluster_id2), axis=0, keepdims=True)
        concept2_neg = tf.math.reduce_mean(tf.gather(cluster_emb_2, neg_item_cluster_id2), axis=0, keepdims=True)


        pos_item_cluster_id3 = tf.nn.embedding_lookup(list(cluster_ids_3.values()), pos)
        neg_item_cluster_id3 = tf.nn.embedding_lookup(list(cluster_ids_3.values()), neg)
        concept3_pos = tf.math.reduce_mean(tf.gather(cluster_emb_3, pos_item_cluster_id3), axis=0, keepdims=True)
        concept3_neg = tf.math.reduce_mean(tf.gather(cluster_emb_3, neg_item_cluster_id3), axis=0, keepdims=True)
        

        pos_item_cluster_id = tf.gather(cluster_ids, pos_id) 
        neg_item_cluster_id = tf.gather(cluster_ids, neg_id)


        concept1_pos = tf.gather(cluster_center_emb, pos_item_cluster_id)
        concept1_neg = tf.gather(cluster_center_emb, neg_item_cluster_id) 

        # unified center
        
        batch_label =  self.batch_label_tensor[beh]
    
        uni_center = tf.sparse.sparse_dense_matmul(batch_label, self.ilat[beh])

        batch_label = tf.sparse.to_dense(batch_label)
        num_rel = tf.expand_dims(tf.math.reduce_sum(batch_label, 1), 1)
        uni_center = uni_center / num_rel 

        user_emb_ego = tf.gather(self.ego_embeddings[:args.user], user_id)
        pos_emb_ego = tf.gather(self.ego_embeddings[args.user:], pos_id)
        neg_emb_ego = tf.gather(self.ego_embeddings[args.user:], neg_id)

        return user_emb_ego, pos_emb_ego, neg_emb_ego, user_emb, pos_emb, neg_emb, concept1_pos, concept1_neg, concept2_pos, concept2_neg,concept3_pos, concept3_neg,  uni_center
    
    def concept_aware_state(self, user, pos, neg, concept1_pos, concept1_neg, concept2_pos, concept2_neg, concept3_pos, concept3_neg,  uni_center):
        A = tf.pow(user - pos, 2)  
        a = tf.pow(user - neg, 2)  
        B = tf.pow(user - concept1_pos, 2)  
        b = tf.pow(user - concept1_neg, 2)  
        C = tf.pow(concept1_pos - pos, 2)  
        c = tf.pow(concept1_neg - neg, 2)  
        
        A_pdt = user * pos  
        a_pdt = user * neg  
        B_pdt = user * concept1_pos  
        b_pdt = user * concept1_neg  
        C_pdt = concept1_pos * pos  
        c_pdt = concept1_neg * neg  

        return tf.concat((
            A_pdt, a_pdt, a_pdt - A_pdt,
            B_pdt, b_pdt, b_pdt - B_pdt,
            C_pdt, c_pdt, c_pdt - C_pdt,
        ), 1)

    def max_norm(self, param, max_val=1, eps=1e-8):
        norm = param.norm(2, dim=1, keepdim=True)
        desired = tf.clip_by_value(norm, 0, max_val)
        param = param * (desired / (eps + norm))

        return param
    
    def sampleTestBatch(self, batIds, labelMat):
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds] 
        temLabel = labelMat[batIds].toarray() 
        temlen = batch * 10000
  
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        tstLocs = [None] * batch
        batch_item_id = []
        neg_samples = []
        cur = 0
        for i in range(batch):
            posloc = temTst[i] 
            negset = np.reshape(np.argwhere(temLabel[i] == 0), [-1])
        
            rdnNegSet = np.random.permutation(negset)[:9999] 
            batch_item_id.append(posloc)
            neg_samples.append(rdnNegSet[0])
            locset = np.concatenate((rdnNegSet, np.array([posloc]))) 
            tstLocs[i] = locset
  
            for j in range(10000):
                uLocs[cur] = batIds[i]
                iLocs[cur] = locset[j]
                cur += 1
        return uLocs, iLocs, temTst, tstLocs, batch_item_id, neg_samples


    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2 
        
        ids = self.handler.tstUsrs 
        
        num = len(ids)
        tstBat = args.batch # 256
        steps = int(np.ceil(num / tstBat))
        for i in range(steps):
            self.batch_label = []
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}
            uLocs, iLocs, temTst, tstLocs, batch_item_id, neg_samples = self.sampleTestBatch(batIds, self.handler.trnMats[-1]) 
            neg = np.squeeze(neg_samples)
            
            self.batch_label.append(self.handler.trnMats[-1][batIds, :])
            
            feed_dict[self.uids[-1]] = uLocs
            feed_dict[self.iids[-1]] = iLocs
            feed_dict[self.batch_item_ids[-1]] = batch_item_id
            feed_dict[self.neg_ids[-1]] = neg
            
            preds = self.sess.run(self.targetPreds, feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            hit, ndcg = self.calcRes(np.reshape(preds, [ed - st, 10000]), temTst, tstLocs)
            epochHit += hit
            epochNdcg += ndcg
        
        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def calcRes(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if temTst[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))
        return hit, ndcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    log_dir = 'log/' + args.data + '/' + os.path.basename(__file__)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')

    def my_hook_out(text):
        log_file.write(text)
        log_file.flush()
        return 1, 0, text

    ph_out = PrintHook()
    ph_out.Start(my_hook_out)

    print("Use gpu id:", args.gpu_id)
    for arg in vars(args):
        print(arg + '=' + str(getattr(args, arg)))

    logger.saveDefault = True
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    with tf.compat.v1.Session(config=config) as sess:
        recom = Recommender(sess, handler)
        recom.run()
