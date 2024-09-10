from mlplib import *
from nnet.dplib import *



class ResNetRelu(Linear):
    def __init__(self, idim, odim,
                 rng=None,
                 irange=0.1):

        super(ResNetRelu, self).__init__(idim, odim, rng, irange)

    def fprop(self, inputs):
        a = super(ResNetRelu, self).fprop(inputs)
        #a = numpy.nan_to_num(a)
        #self.y = numpy.clip(a, 0, 0.0)
        self.y = numpy.clip(a, 0, 20.0)
        h = self.y + inputs
        return h

    def bprop(self, h, igrads):
        #igrads = numpy.nan_to_num(igrads)
        deltas = (self.y > 0)*igrads
        #deltas = numpy.nan_to_num(deltas)
        ograds = numpy.dot(deltas, self.W.T) + igrads
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ResNetRelu.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'resnetrelu'



class BCECost(Cost):

    def cost(self, y, t, **kwargs):

        nll = t * numpy.log(y) + (1.0-t) * numpy.log(1.0-y)
        return -numpy.mean(numpy.sum(nll, axis=1))

    def grad(self, y, t, **kwargs):
        return y - t

    def get_name(self):
        return 'bce'


def BLogistic_Init(num_units = [784, 10], rng = None):
    
    OutAct = 'Sigmoid'

    #OutAct = 'Softmax'
    
    if OutAct == 'Sigmoid':
        cost = BCECost()
    elif OutAct == 'Softmax':
        cost = CECost()
    model = MLP(cost = cost)
   
    print('Constructing a %s single layer (%dx%d).'%(OutAct,num_units[0], num_units[1]))
    model.add_layer(eval("%s(idim = num_units[0], odim = num_units[1], rng = rng)"%OutAct))

    return model





def BNNet_Init(num_units = [784, 128, 128, 10], rng = None):
    
    
    HidAct = 'Relu' #'Tanh'
    OutAct = 'Sigmoid'
    
    if OutAct == 'Sigmoid':
        cost = BCECost()
    elif OutAct == 'Softmax':
        cost = CECost()
    model = MLP(cost = cost)
    
    for i in xrange(0, len(num_units)-2):
        print('Constructing a %s hidden layer (%dx%d).'%(HidAct,num_units[i], num_units[i+1]))
        model.add_layer(eval("%s(idim = num_units[i], odim = num_units[i+1], rng = rng)"%HidAct))

    print('Constructing a %s output layer (%dx%d).'%(OutAct,num_units[-2], num_units[-1]))
    model.add_layer(eval("%s(idim = num_units[-2], odim = num_units[-1], rng = rng)"%OutAct))

    return model



def BResNet_Init(num_units = [784, 128, 30, 10], rng = None, OutAct = 'Sigmoid'):
    
    
    HidAct = 'ResNetRelu'
    #OutAct = 'Sigmoid'
   
    #OutAct = 'Softmax'
 
    if OutAct == 'Sigmoid':
        cost = BCECost()
    elif OutAct == 'Softmax':
        cost = CECost()
    model = MLP(cost = cost)
   
    print('Constructing a Linear hidden layer (%dx%d).'%(num_units[0], num_units[1]))
    model.add_layer(eval("Linear(idim = num_units[0], odim = num_units[1], rng = rng)"))
 
    for i in xrange(0, num_units[2]):
        print('Constructing a %s hidden layer (%dx%d).'%(HidAct,num_units[1], num_units[1]))
        model.add_layer(eval("%s(idim = num_units[1], odim = num_units[1], rng = rng)"%HidAct))

    print('Constructing a %s output layer (%dx%d).'%(OutAct,num_units[1], num_units[3]))
    model.add_layer(eval("%s(idim = num_units[1], odim = num_units[3], rng = rng)"%OutAct))

    return model



    
    
def BNNet_Train(model, lr_scheduler, tr_data, cv_data = None, dp_scheduler = None, l1_weight = 0.0, l2_weight = 0.0, batch_size = 512):

    print(  '********************************')
    print(  '*        Training BNNet        *')
    print(  '********************************') 

    optimiser = SGDOptimiser(lr_scheduler = lr_scheduler, dp_scheduler = dp_scheduler) 
    print('Initialising training data...')
    # batch_size = 512 works well. batch_size = 10000 can be used with NewBob/Train when the dataset is big.
    train_dp = BMchDataProvider(dset = tr_data, batch_size = batch_size, max_num_batches = -100, randomize = True)
    if cv_data != None:
        print('Initialising cross validation data...')
        valid_dp = BMchDataProvider(dset = cv_data, batch_size = 512, max_num_batches = -100, randomize = False)
    else:
        valid_dp = BMchDataProvider(dset = tr_data, batch_size = 512, max_num_batches = -100, randomize = False)
        
    print('Training the neural network...')
    tr_stat, cv_stat = optimiser.train(model, train_dp, valid_dp)
    
    print('Results:')
    ct, acc = tr_stat[-1]
    print('On Training Set: Error = %.3f, Accuracy = %.3f%%.'%(ct, acc*100.))
    if cv_data != None:
        ct, acc = cv_stat[-1]
        print('On Cross-validation Set: Error = %.3f, Accuracy = %.3f%%.'%(ct, acc*100.))
   
    return model, tr_stat, cv_stat

def BNNet_Decode_Ope_Score(model, feat, opeidx):
    y = model.fprop(feat)
    y = y[:,opeidx]
    #y = list(y)
    x = list(xrange(y.shape[0]))
    z = numpy.asarray([x,y])
    z = z.T.tolist()
    for i in xrange(len(z)):
        z[i][0] = int(z[i][0])
    z = sorted(z, key = lambda p: p[1], reverse = True)
    return z
    


    
    
def BNNet_Semantic_Learning(model, lr_scheduler, data_list, dp_scheduler = None, l1_weight = 0.0, l2_weight = 0.0, batch_size = 512):

    print(  '********************************')
    print(  '*        Training BNNet        *')
    print(  '********************************') 

    tr_data = data_list[0]
    cv_data = data_list[1]
    ev_data = data_list[2]

    optimiser = SGDOptimiser(lr_scheduler = lr_scheduler, dp_scheduler = dp_scheduler) 
    print('Initialising training data...')
    
    train_dp = BSemanticDataProvider(dset = tr_data, batch_size = batch_size, max_num_batches = -100, randomize = True)

    if cv_data != None:
        print('Initialising cross validation data...')
        valid_dp = BSemanticDataProvider(dset = cv_data, batch_size = 512, max_num_batches = -100, randomize = False)
    else:
        valid_dp = BSemanticDataProvider(dset = tr_data, batch_size = batch_size, max_num_batches = -100, randomize = False)

    if ev_data != None:
        print('Initialising test data...')
        eval_dp = BSemanticDataProvider(dset = ev_data, batch_size = 512, max_num_batches = -100, randomize = False)
        
    print('Training the neural network...')
    tr_stat, cv_stat = optimiser.train(model, train_dp, valid_dp)

    if ev_data != None:
        print('Testing the neural network...')
        ev_stat = optimiser.validate(model, eval_dp)
    else:
        ev_stat = [-1,-1]
    
    print('Results:')
    ct, acc = tr_stat[-1]
    print('On Training Set: Error = %.3f, Accuracy = %.3f%%.'%(ct, acc*100.))
    if cv_data != None:
        ct, acc = cv_stat[-1]
        print('On Cross-validation Set: Error = %.3f, Accuracy = %.3f%%.'%(ct, acc*100.))
  
    ct, acc = ev_stat
    print('On Testing Set: Error = %.3f, Accuracy = %.3f%%.'%(ct, acc*100.))

    return model, tr_stat, cv_stat, ev_stat


    
