import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import random
import numpy
from torchvision import datasets, transforms
from torch.autograd import Variable

if torch.cuda.is_available():
    print "Use a GPU to perform training tasks..."
    use_gpu = True
else:  
    print "Use a CPU to perform training tasks..."



class MLP_ReLU(nn.Module):
    def __init__(self, num_nodes = [50, 100, 100, 100, 10]):
        super(MLP_ReLU, self).__init__()
        self.num_nodes = num_nodes
        self.layers = []
        for i in xrange(len(self.num_nodes)-1):
            w = nn.Linear(self.num_nodes[i], self.num_nodes[i+1])
            self.layers.append(w)
            add_layer_statement = "self.layer_%d = self.layers[%d]"%(i,i)
            exec(add_layer_statement)

    def forward(self, x):
        y = x
        for i in xrange(len(self.layers)-1):
            y = nnf.relu(self.layers[i](y))
        y = self.layers[len(self.layers)-1](y)
        y = nnf.log_softmax(y, dim = 1)
        return y



class FNN_RBB(nn.Module):
    def __init__(self, num_nodes = [50, [100, 10], 10]):
        super(FNN_RBB, self).__init__()
        self.num_nodes = num_nodes

        self.linear_layer = nn.Linear(self.num_nodes[0], self.num_nodes[1][0])

        self.layers = []
        for i in xrange(self.num_nodes[1][1]):
            w = nn.Linear(self.num_nodes[1][0], self.num_nodes[1][0])
            self.layers.append(w)
            add_layer_statement = "self.layer_%d = self.layers[%d]"%(i,i)
            exec(add_layer_statement)

        i = self.num_nodes[1][1]
        w = nn.Linear(self.num_nodes[1][0], self.num_nodes[2])
        self.layers.append(w)
        add_layer_statement = "self.layer_%d = self.layers[%d]"%(i,i)
        exec(add_layer_statement)

    def forward(self, x):
        x0 = self.linear_layer(x)
        y = x0 + 0
        for i in xrange(len(self.layers)-1):
            y = nnf.relu(self.layers[i](y)) + x0
        y = self.layers[len(self.layers)-1](y)
        y = nnf.log_softmax(y, dim = 1)
        return y






class AE_ReLU(nn.Module):
    def __init__(self, num_nodes = [100, 50, 25, 50, 100]):
        super(AE_ReLU, self).__init__()
        self.in_tensor = None
        self.num_nodes = num_nodes
        self.layers = []
        for i in xrange(len(self.num_nodes)-1):
            w = nn.Linear(self.num_nodes[i], self.num_nodes[i+1])
            self.layers.append(w)
            add_layer_statement = "self.layer_%d = self.layers[%d]"%(i,i)
            exec(add_layer_statement)

    def forward(self, x):
        self.in_tensor = torch.tensor(x, requires_grad=True)
        
        y = self.in_tensor
        for i in xrange(len(self.layers)-1):
            y = nnf.relu(self.layers[i](y))
        y = self.layers[len(self.layers)-1](y)
        #y = nnf.log_sigmoid(y, dim = 1)
        return y




class MLP_AE_ReLU(nn.Module):
    def __init__(self, ae_model, num_nodes = [50, 100, 100, 100, 10]):
        super(MLP_AE_ReLU, self).__init__()

        self.num_nodes = num_nodes
        self.auto_encoder = []
        for i in xrange(len(ae_model.layers)):
            if i >= len(ae_model.layers) / 2:
                break
            self.auto_encoder.append(ae_model.layers[i])
            add_auto_encoder_statement = "self.auto_encoder_%d = self.auto_encoder[%d]"%(i,i)
            exec(add_auto_encoder_statement)

        # A transformation matrix connecting the auto-encoder and 1st hidden layer.
        ae_output_dim = self.auto_encoder[-1].weight.size()[0]
        self.auto_encoder_tfm = nn.Linear(ae_output_dim, self.num_nodes[1])

        self.layers = []
        for i in xrange(len(self.num_nodes)-1):
            w = nn.Linear(self.num_nodes[i], self.num_nodes[i+1])
            self.layers.append(w)
            add_layer_statement = "self.layer_%d = self.layers[%d]"%(i,i)
            exec(add_layer_statement)

    def forward(self, x):

        y_ae = x
        #print y_ae.size()

        for i in xrange(len(self.auto_encoder)):
            y_ae = nnf.relu(self.auto_encoder[i](y_ae))
            #print y_ae.size()

        #print self.auto_encoder_tfm.weight.size()
        
        y = x
        for i in xrange(len(self.layers)-1):
            if i == 0:
                y_ae = self.auto_encoder_tfm(y_ae)
                y = nnf.relu(self.layers[i](y) + y_ae)
            else:
                y = nnf.relu(self.layers[i](y))
        y = self.layers[len(self.layers)-1](y)
        y = nnf.log_softmax(y, dim = 1)

        return y


#random.seed(gen_seed)
#torch.manual_seed(gen_seed)

class GAN_G_ReLU(nn.Module):
    def __init__(self, num_nodes = [20, 100, 100, 100, 784]):
        super(GAN_G_ReLU, self).__init__()

        self.num_nodes = num_nodes
        self.layers = []
        for i in xrange(len(self.num_nodes)-1):
            w = nn.Linear(self.num_nodes[i], self.num_nodes[i+1])
            self.layers.append(w)
            add_layer_statement = "self.layer_%d = self.layers[%d]"%(i,i)
            exec(add_layer_statement)

    def forward(self, x):
        y = x
        for i in xrange(len(self.layers)-1):
            y = nnf.relu(self.layers[i](y))
        #y = self.layers[len(self.layers)-1](y)
        y = torch.sigmoid(self.layers[len(self.layers)-1](y))
        return y




def train_auto_encoder(model, optimizer, train_data):
    loss_tot = 0.0
    num_data = 0
    for batch_idx, feat in enumerate(train_data):
        output = model(feat)
        loss = nnf.mse_loss(output, feat, reduction='mean')
        loss_tot = loss_tot + loss.data.item() * len(feat)
        num_data = num_data + len(feat)

        #if batch_idx % 100 == 0:
        #    print "Batch Idx:", batch_idx
        #    print "Loss:", loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_ave = loss_tot / num_data
    print "Average Loss:", loss_ave
    return model, loss_ave



def train_auto_encoder_newbob_tr(model, train_data, max_epochs, init_lrate, imp0, imp1):
    # Get initial loss.
    train_loss_prev = 100000.0
    lrate = init_lrate
    imp_flag = 0
    for epoch in xrange(max_epochs):
        print "Epoch %d, learning rate = %.6lf:"%(epoch+1, lrate)
        optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.5)
        model, train_loss = train_auto_encoder(model, optimizer, train_data)
        if imp_flag in [0, 1] and train_loss_prev - train_loss <= imp0:
            lrate = lrate / 2
            imp_flag = 1
        if imp_flag == 1 and train_loss_prev - train_loss <= imp1:
            imp_flag = 2
            break
        #if imp_flag == 1:
        #    lrate = lrate / 2

        train_loss_prev = train_loss 
    return model




def train_nnet(model, optimizer, train_data):
    loss_tot = 0.0
    num_data = 0
    for batch_idx, (feat, target) in enumerate(train_data):
        output = model(feat)
        loss = nnf.nll_loss(output, target, reduction='mean')
        if "%s"%loss.data.item() in ["nan","inf","-inf"]:
            continue
        loss_tot = loss_tot + loss.data.item() * len(feat)
        num_data = num_data + len(feat)

        #if batch_idx % 100 == 0:
        #    print "Batch Idx:", batch_idx
        #    print "Loss:", loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_ave = loss_tot / num_data
    print "Average Loss:", loss_ave
    return model, loss_ave

def test_nnet(model, test_data):
    loss_tot = 0.0
    num_data = 0
    correct_pred = 0
    for batch_idx, (feat, target) in enumerate(test_data):
        with torch.no_grad():
            output = model(feat)
            pred = output.data.max(1, keepdim=True)[1]
            correct_pred = correct_pred + pred.eq(target.data.view_as(pred)).cpu().sum()
            num_data = num_data + len(feat)
            loss = nnf.nll_loss(output, target)
            loss_tot = loss_tot + loss.data.item() * len(feat) 

    acc_pred = correct_pred.item() * 1.0 / num_data
    loss_ave = loss_tot / num_data

    print "Test Loss:", loss_ave
    print "Correct Num:", correct_pred.item()
    print "Test Num:", num_data
    print "Accuracy:", 100. * acc_pred
    return loss_ave


def train_nnet_newbob_tr(model, train_data, max_epochs, init_lrate, imp0, imp1):
    # Get initial loss.
    train_loss_prev = 100000.0
    lrate = init_lrate
    imp_flag = 0
    for epoch in xrange(max_epochs):
        print "Epoch %d, learning rate = %.6lf:"%(epoch+1, lrate)
        optimizer = optim.SGD(model.parameters(), lr=lrate, momentum=0.5)
        model, train_loss = train_nnet(model, optimizer, train_data)
        if imp_flag in [0, 1] and train_loss_prev - train_loss <= imp0:
            lrate = lrate / 2
            imp_flag = 1
        if imp_flag == 1 and train_loss_prev - train_loss <= imp1:
            imp_flag = 2
            break
        #if imp_flag == 1:
        #    lrate = lrate / 2

        train_loss_prev = train_loss 
    return model




def train_GAN(model_G, optimizer_G, model_D, optimizer_D, train_data, use_gpu = False):
    loss_D_tot = 0.0
    loss_G_tot = 0.0
    num_D_data = 0
    num_G_data = 0
    correct_D_pred = 0

    for batch_idx, feat in enumerate(train_data):

        # Generate training data for D

        rand_in_tensor = torch.randn(feat.size()[0], model_G.layers[0].weight.size()[1])
        if use_gpu == True:
            rand_in_tensor = rand_in_tensor.cuda()

        feat_fake = model_G(rand_in_tensor)

        feat_D = torch.cat((feat, feat_fake), 0)
        if use_gpu == True:
            feat_D = feat_D.cuda()
        # 0 - real data
        # 1 - fake data
        n = feat_D.size()[0] / 2
        target_D = [0] * n + [1] * n
        target_D = torch.tensor(target_D)
        if use_gpu == True:
             target_D = target_D.cuda()

        # Train D.
        output_D = model_D(feat_D)

        pred_D = output_D.data.max(1, keepdim=True)[1]
        num_correct_pred_D = pred_D.eq(target_D.data.view_as(pred_D)).sum()
        correct_D_pred = correct_D_pred + num_correct_pred_D

        loss_D = nnf.nll_loss(output_D, target_D, reduction='mean')
        loss_D_tot = loss_D_tot + loss_D.data.item() * len(target_D)

        optimizer_D.zero_grad()
        loss_D.backward()
        #xxx = model_D.layers[0].weight + 0
        optimizer_D.step()
        #yyy = xxx - model_D.layers[0].weight
        #print yyy

        num_D_data = num_D_data + len(feat_D)

        #Train G.
        no_impr_epochs = 0
        num_correct_pred_fake_prev = -1
        while no_impr_epochs < 8: # 2 4 8..
            feat_fake = model_G(rand_in_tensor)
            output_fake = model_D(feat_fake)
            target_fake = torch.tensor([0] * feat_fake.size()[0])
            if use_gpu == True:
                target_fake = target_fake.cuda()


            pred_fake = output_fake.data.max(1, keepdim=True)[1]
            num_correct_pred_fake = pred_fake.eq(target_fake.data.view_as(pred_fake)).sum()
            #print num_correct_pred_fake.item()
            if num_correct_pred_fake.item() == feat_fake.size()[0]:
                break
            if num_correct_pred_fake.item() > num_correct_pred_fake_prev:
                num_correct_pred_fake_prev = num_correct_pred_fake.item() + 0
                no_impr_epochs = 0
            else:
                no_impr_epochs = no_impr_epochs + 1

            loss_fake = nnf.nll_loss(output_fake, target_fake, reduction='mean')
            num_G_data = num_G_data + feat_fake.size()[0]
            loss_G_tot = loss_G_tot + loss_fake.data.item() * len(target_fake)

            
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            #print model_G.layers[0].weight.grad
            #raw_input("asdfas")
            loss_fake.backward()
            #xxx = model_D.layers[0].weight + 0
            optimizer_G.step()
            #yyy = xxx - model_D.layers[0].weight
            #print yyy
            #print model_G.layers[0].weight.grad
            #print model_D.layers[0].weight.grad

        #num_G_data = num_G_data + len(feat_D)

        #if batch_idx % 100 == 0:
        #    print "Batch Idx:", batch_idx
        #    print "Loss:", loss.data.item()

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
    if num_G_data == 0:
        loss_G_ave = 10000
    else:
        loss_G_ave = loss_G_tot / num_G_data
    loss_D_ave = loss_D_tot / num_D_data
    print "Average Loss of G:", loss_G_ave
    print "Average Loss of D:", loss_D_ave
    print "Accuracy of D:", correct_D_pred.item() * 1.0 / num_D_data
    return model_G, loss_G_ave, model_D, loss_G_ave

"""
# The Newbob learning rate strategy is problematic for GAN.
def train_GAN_newbob_tr(model_G, model_D, train_data, max_epochs, init8lrate, patience, imp0, imp1, use_gpu):
8# Get initial loss.
    train_loss_prev = 100000.0
    lrate = init_lrate
    imp_flag = 0
    no_impr = 0
    for epoch in xrange(max_epochs):
        print "Epoch %d, learning rate = %.6lf:"%(epoch+1, lrate)
        optimizer_G = optim.SGD(model_G.parameters(), lr=lrate, momentum=0.5)
        optimizer_D = optim.SGD(model_D.parameters(), lr=lrate, momentum=0.5)
        model_G, loss_G_ave, model_D, loss_D_ave = train_GAN(model_G, optimizer_G, model_D, optimizer_D, train_data, use_gpu = use_gpu)
        train_loss = loss_G_ave + loss_D_ave
        print "Total Average Loss is: %.6lf."%train_loss


        if imp_flag == 1 and train_loss_prev - train_loss <= imp1 and no_impr == patience:
            imp_flag = 2
            break

        if imp_flag in [0, 1] and train_loss_prev - train_loss <= imp0:
            no_impr = no_impr + 1
            if no_impr > patience:
                lrate = lrate / 2
                imp_flag = 1
                no_impr = 0
                train_loss_prev = train_loss
         #if imp_flag == 1:
        #    lrate = lrate / 2

        if train_loss_prev - train_loss > imp1:
            train_loss_prev = train_loss
            no_impr = 0
    return model_G, model_D
"""


def train_GAN_multi_epochs(model_G, model_D, train_data, max_epochs, init_lrate, lrate_dec_prop, use_gpu):
    # Get initial loss.
    lrate = init_lrate
    imp_flag = 0
    for epoch in xrange(max_epochs):
        if epoch >= max_epochs:
            break
        if lrate / init_lrate < 0.01:
            break
        print "Epoch %d, learning rate = %.6lf:"%(epoch+1, lrate)
        optimizer_G = optim.SGD(model_G.parameters(), lr=lrate * 10, momentum=0.5)
        optimizer_D = optim.SGD(model_D.parameters(), lr=lrate, momentum=0.5)
        model_G, loss_G_ave, model_D, loss_D_ave = train_GAN(model_G, optimizer_G, model_D, optimizer_D, train_data, use_gpu = use_gpu)
        train_loss = loss_G_ave + loss_D_ave
        print "Total Average Loss is: %.6lf."%train_loss
        lrate = lrate * lrate_dec_prop

    return model_G, model_D


"""

train_dataset = datasets.MNIST(root='../../pytorch/data/', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(root='../../pytorch/data/', train=False, transform=transforms.ToTensor())

batch_size = 32

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#model = ConvNNExp()
#model_mlp = MLP_ReLU(num_nodes = [784, 128, 128, 128, 128, 128, 10])
model_mlp = FNN_RBB(num_nodes = [784, [128, 4], 10])
#optimizer_sgd = optim.SGD(model_mlp.parameters(), lr=0.01, momentum=0.5)

use_gpu = True

mnist_train_data = []
mnist_train_data_feat = []
for batch_idx, (feat, target) in enumerate(train_loader):
    if len(mnist_train_data) > 100: break
    feat = feat.reshape(feat.size()[0], 784)
    if use_gpu == True:
        mnist_train_data.append((feat.cuda(), target.cuda()))
        mnist_train_data_feat.append(feat.cuda())
    else:
        mnist_train_data.append((feat, target))
        mnist_train_data_feat.append(feat)
mnist_test_data = []
for batch_idx, (feat, target) in enumerate(test_loader):
    feat = feat.reshape(feat.size()[0], 784)
    mnist_test_data.append((feat, target))


random.seed(777)
torch.manual_seed(777)



model_G = GAN_G_ReLU(num_nodes = [256, 256, 256, 256, 784])
model_D = MLP_ReLU(num_nodes = [784, 256, 256, 256, 2])
if use_gpu == True:
    model_G = model_G.cuda()
    model_D = model_D.cuda()
train_GAN_multi_epochs(model_G, model_D, mnist_train_data_feat, max_epochs = 1000, init_lrate = 0.01, lrate_dec_prop = 0.995, use_gpu = use_gpu)


pppppp


model_G = GAN_G_ReLU(num_nodes = [128, 128, 128, 128, 784])
optimizer_G = optim.SGD(model_G.parameters(), lr=0.01, momentum=0.5)
model_D = MLP_ReLU(num_nodes = [784, 128, 128, 128, 2])
optimizer_D = optim.SGD(model_D.parameters(), lr=0.01, momentum=0.5)
if use_gpu == True:
        model_G = model_G.cuda()
        model_D = model_D.cuda()
for i in xrange(20):
    print "Epoch", i
    model_G, loss_G_ave, model_D, loss_D_ave = train_GAN(model_G, optimizer_G, model_D, optimizer_D, mnist_train_data_feat, use_gpu = use_gpu)
ppppp

model_ae = AE_ReLU(num_nodes = [784, 128, 64, 128, 784])
if use_gpu == True:
    model_ae = model_ae.cuda()
model_ae = train_auto_encoder_newbob_tr(model_ae, mnist_train_data_feat, max_epochs = 100, init_lrate = 0.01, imp0 = 0.001, imp1 = 0.0001)
model_mlp_ae = MLP_AE_ReLU(model_ae, num_nodes = [784, 128, 128, 128, 10])
if use_gpu == True:
    model_mlp_ae = model_mlp_ae.cuda()
    
model_mlp_ae = train_nnet_newbob_tr(model_mlp_ae, mnist_train_data, max_epochs = 100, init_lrate = 0.01, imp0 = 0.01, imp1 = 0.001)

if use_gpu == True:
    # After using GPU to perform the training task, we convert the trained model to CPU version.
    model_mlp_ae = model_mlp_ae.cpu()

test_nnet(model_mlp_ae, mnist_test_data)

ppp

model_mlp = train_nnet_newbob_tr(model_mlp, mnist_train_data, max_epochs = 100, init_lrate = 0.01, imp0 = 0.01, imp1 = 0.001)

test_nnet(model_mlp, mnist_test_data)


ppp

for epoch in xrange(10):
    print "Epoch %d:"%(epoch+1)
    model_mlp, train_loss = train_nnet(model_mlp, optimizer_sgd, mnist_train_data)
    test_nnet(model_mlp, mnist_test_data)
"""
