from mlplib import *
from mlp.dataset import *
import random

class BMchDataProvider(MNISTDataProvider):

    def __init__(self, dset,
                 batch_size=10,
                 max_num_batches=-1,
                 max_num_examples=-1,
                 randomize=True,
                 rng=None,
                 conv_reshape=False):
                 
        self.batch_size = batch_size
        self.randomize = randomize
        self._curr_idx = 0
        self.rng = rng

        if self.rng is None:
            seed=[2015,10,1]
            self.rng = numpy.random.RandomState(seed)
        
        assert max_num_batches != 0, (
            "max_num_batches should be != 0"
        )

        dset_path = dset
        assert os.path.isfile(dset_path), (
            "File %s was expected to exist!." % dset_path
        )

        x = []
        t = []
        f = open(dset_path,"r")
        flag = 0
        
        pv = f.readline()
        self.feat_dim = self.get_para(pv,"feat_dim")
        pv = f.readline()
        self.num_classes = self.get_para(pv,"num_opes")
      
        for line in f.readlines():
            if flag == 0:
                p = line.split()
                inp = []
                for item in p:
                    inp.append(float(item))
                x.append(inp)
            else:
                tgt = int(line)
                t.append(tgt)
            flag = 1 - flag

        cttgt = numpy.zeros(self.num_classes)
        for item in t:
            cttgt[item] = cttgt[item] + 1
        
        print "Target Count:\n", cttgt


        self._max_num_batches = max_num_batches
        
        if max_num_examples > 0 and max_num_batches < 0:
            self._max_num_batches = max_num_examples / self.batch_size      


        #x = x * 10
        #t = t * 10
        # Make Isolation Data
        num_trans = len(x)
        num_iso = int(num_trans)
        x_iso = []
        fd = self.feat_dim
        fdh = fd / 2
        i = 0
        iflag = 0
        while i < num_iso:
            p = int(random.random() * num_trans)
            q = int(random.random() * num_trans)
            u = x[p][0:fdh] + x[q][fdh:fd]
            if not(u in x):
                x_iso.append(u)
                i += 1
                iflag = 0
            else:
                iflag += 1
                if iflag > 10000:
                    print "Warning: Not able to find Iso Relation."
                    break
        t_iso = [-1] * num_iso
        
        
        x = x + x_iso
        t = t + t_iso

        # End of Isolation Data


        xp = []
        tp = []
        for i in xrange(len(x)):
            u = x[i]
            v = t[i]
            if not(u in xp):
                xp.append(u)
                tp.append([v])
            else:
                j = xp.index(u)
                tp[j].append(v)

        tp = self.multi_classes(tp)

        lxp = len(xp)
        if self.batch_size > lxp:
            print "Warning: Only has %d training data. Set batch size to %d."%(lxp,lxp)
            self.batch_size = lxp
        x = numpy.asarray(xp)
        t = numpy.asarray(tp)

        #t = numpy.asarray(t)

        """
        pp = numpy.asarray([[1,6,3],[7,1,2],[0,3,1],[-5,6,1]])
        print pp[pp[:,1].argsort()]
        ppp

        print x.shape
        print t.shape
        dt = numpy.concatenate((x,t.reshape(t.shape[0],1)),axis=1)
        sdt = numpy.sort(dt)
        print dt
        print sdt
        ppp
        """

        self.x = x
        self.t = t
        self.conv_reshape = conv_reshape

        self._rand_idx = None
        if self.randomize:
            self._rand_idx = self.__randomize()
            
 
    def __randomize(self):
        return self._MNISTDataProvider__randomize()
        
       
    def get_para(self,x,pname):
        y = x.split("=")
        if y[0] != pname:
            print "Error: Not able to get parameter %s from %s"%(pname,x)
            return None
        res = int(y[1])
        return res

    def multi_classes(self, y):
        rval = numpy.zeros((len(y), self.num_classes), dtype=numpy.float32)
        for i in xrange(len(y)):
            for j in xrange(len(y[i])):
                if y[i][j] == -1:
                    break
                rval[i, y[i][j]] = 1
        return rval
 
    def _MNISTDataProvider__to_one_of_k(self, y):
        # Override the original method, because we do not use one-hot encodings.
        return y











class BSemanticDataProvider(MNISTDataProvider):

    def __init__(self, dset,
                 batch_size=10,
                 max_num_batches=-1,
                 max_num_examples=-1,
                 randomize=True,
                 rng=None,
                 conv_reshape=False):
                 
        self.batch_size = batch_size
        self.randomize = randomize
        self._curr_idx = 0
        self.rng = rng

        if self.rng is None:
            seed=[2015,10,1]
            self.rng = numpy.random.RandomState(seed)
        
        assert max_num_batches != 0, (
            "max_num_batches should be != 0"
        )

        dset_path = dset
        assert os.path.isfile(dset_path), (
            "File %s was expected to exist!." % dset_path
        )
        self._max_num_batches = max_num_batches

        if max_num_examples > 0 and max_num_batches < 0:
            self._max_num_batches = max_num_examples / self.batch_size

        f = open(dset_path,"r")
        
        x = []
        t = []
        for P in f.readlines():
            U = P.replace("\n","")
            while U[-1] == " ":
                U = U[0:len(U)-1]
            U = U.split(" ")

            if len(U) == 1:
                t.append(int(U[0]))
            else:
                x.append(map(int,U))

        x = numpy.asarray(x)
        t = numpy.asarray(t)
        
        self.x = x
        self.t = t
        self.num_classes = 2
        self.conv_reshape = conv_reshape

        self._rand_idx = None
        if self.randomize:
            self._rand_idx = self.__randomize()
           
 
    def __randomize(self):
        return self._MNISTDataProvider__randomize()
        

