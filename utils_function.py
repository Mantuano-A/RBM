import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import copy
#suppress warnings
warnings.filterwarnings('ignore')

from matplotlib import colors
from tqdm import tqdm
from collections import Counter
from IPython.display import HTML, Javascript


def myDataset(a):
    copies = []
    label = []
    for i in range(len(a)):
        for _ in range(2000):
            copies.append(flipped(5,a[i][0]))
            label.append(a[i][1])
    return pd.DataFrame({'Copies':copies, 'Label':label})

def plotReconstructed(img1,img2,img3,img4 = []):
    n_img = 4
    
    if len(img4) > 0:
        n_img += 1
        
    fig, ax = plt.subplots(1, n_img, figsize=(16, 6))
    mycmap = colors.ListedColormap(['#000000','#ffffff','red'])
    boundaries = [-2, 0.5, 2, 25]
    mynorm = colors.BoundaryNorm(boundaries, mycmap.N, clip=True)
    
    ax[0].imshow(img1.reshape(28, 28), mycmap, norm = mynorm)
    ax[0].set_title('Original\n', fontsize = 18)
    ax[1].imshow(img2, mycmap, norm = mynorm)
    ax[1].set_title('Corrupted\n', fontsize = 18)
    
    ax[2].imshow(plt.imread('arrow.png'))
    ax[2].axis('off')
    
    ax[3].imshow(img3.reshape(28,28), mycmap, norm = mynorm)
    ax[3].set_title('Reconstructed\n', fontsize = 18) 
    
    if len(img4) > 0:
        ax[4].imshow(img4.reshape(28,28), cmap = mycmap, norm = mynorm)
        ax[4].set_title('Errors\n', fontsize = 18)
        
    for i in range(n_img):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()
    
def generateArchetype(num = 5):
    a = []
    for i in range(num):
        a.append((np.random.binomial(1, 0.5, 28*28),i))
    return a

def differences(img1,img2, p = 0, tot = 28*28, Print = True):
    count = 0
    diff = copy.deepcopy(img1).reshape(28, 28)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if diff[i][j] != img2[i][j]:
                diff[i][j] = 20
                count += 1
    if Print == True:
        if p > 0:
            print("\nAfter flipped", p, "% of pixel the accuracy is: ", round(100-count*100/tot,2),"%")   
        else:
            print("\nAccuracy of: ", round(100-count*100/tot,2),"%")
    return diff

def flipped(perc,archetype):
    img = copy.deepcopy(archetype)
    for i in range(len(img)):
        if np.random.random_sample() < perc/100:
            img[i] = (1+img[i])%2
    return img

def batch_iterator(X, y = None, batch_size = 64):
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]
            
def sigma (x):
    return 1/(1+ np.exp(-x))

def reconstruction(img, rbm, corrupted = 0, k = 100, alpha = 0.8):
    # Image Reconstruction
    # k_iter Number of Gibbs iterations
    # alpha Decay factor
    
    # Array to store the reconstruction
    X_recon = np.zeros((28, 28-corrupted))
    b = img.copy().reshape(-1)

    for i in range(k):
        b = rbm._gibbs(b)
        if corrupted > 0: 
            X_recon += alpha**(i) * b.reshape(28,28)[:,corrupted:]
            #keep the first columns unchanged
            b.reshape(28,28)[:,:corrupted] = img[:,:corrupted]
        else:
            X_recon += alpha**(i) * b.reshape(28,28)[:,corrupted:]
            
    return X_recon

class RBM():
    
    def __init__(self, n_visible, n_hidden, learning_rate = 0.1, batch_size = 1, n_iterations = 100, classifier = False, n_label = 0):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        
        self.training_errors = []
        self.training_reconstructions = []
        self.W = np.zeros(shape = (n_visible, self.n_hidden))
        self.v0 = np.zeros(n_visible)       # Bias visible
        self.h0 = np.zeros(self.n_hidden)   # Bias hidden
    
        if classifier:
            self.n_label = n_label
            self.U = np.zeros(shape = (n_label, self.n_hidden))
            self.z0 = np.zeros(n_label)
            
    def _initialize_weights(self, classifier):
        self.W = np.random.normal(scale = 0.1, size = (self.n_visible, self.n_hidden))

        if classifier:
            self.U = np.random.normal(scale = 0.1, size = (self.n_label, self.n_hidden))
    
    def _save(self, classifier = False, label = "" ):
        if classifier:
            path = "trained/"+label+"Class_RBM_"+str(self.n_hidden)+"_hidden_"+str(self.lr)+"_lr_"+".csv"
        else:
            path = "trained/"+label+"RBM_"+str(self.n_hidden)+"_hidden_"+str(self.lr)+"_lr_"+".csv"
        f = open(path, "wb")
        pickle.dump(self.__dict__, f)
        f.close()
    
    def _train(self, X, y = None, classifier = False, output = False):
        
        self._initialize_weights(classifier)

        if classifier: self._CD1_Classification(X, y, output)
        else: self._CD1_Generative(X, output)
            
 ##################GENERATIVE#########################    

        
    def _mean_hiddens(self, v):
        #Computes the probabilities P(h=1|v). 
        
        return sigma(v.dot(self.W) + self.h0)
    
    def _sample_hiddens(self, v):
        #Sample from the distribution P(h|v).

        p = self._mean_hiddens(v)
        return self._sample(p)
    
    def _sample_visibles(self, h):
        #Sample from the distribution P(v|h).
        
        p = sigma(h.dot(self.W.T) + self.v0)
        return self._sample(p)
    
    def _gibbs(self, v):
        #Perform one Gibbs sampling step.
        
        h_ = self._sample_hiddens(v)
        v_ = self._sample_visibles(h_)

        return v_
    
    def _sample(self, X):
        
        return X > np.random.random_sample(size = X.shape)
    
    def _CD1_Generative(self, X, output):

        for _ in tqdm(range(self.n_iterations), disable = output):
            batch_errors = []
            
            for batch in batch_iterator(X, batch_size = self.batch_size):
                #v_0 = batch 
                
                # Positive phase ---> E_v,j,o[sisj] = E_s(0)[sisj]
                positive_hidden = self._mean_hiddens(batch)  # E(h)_0=1*P(h=1|v) -1*P(h=-1|v)
                positive_associations = batch.T.dot(positive_hidden) #E(h)_0*v_0
                hidden_states = self._sample_hiddens(batch) # hidden to use in the second part of sample    
                
                # Negative phase ---> E_j,o[sisj] = E_s(inf)[sisj]
                negative_visible = self._sample_visibles(hidden_states) # v_k
                negative_hidden = self._mean_hiddens(negative_visible)   # E(h)_k=1*P(h=1|v) -1*P(h=-1|v)
                negative_associations = negative_visible.T.dot(negative_hidden) #E(h)_k*v_k
                
                self.W  += self.lr * (positive_associations - negative_associations)
                self.h0 += self.lr * (positive_hidden.sum(axis = 0) - negative_hidden.sum(axis = 0))
                self.v0 += self.lr * (batch.sum(axis = 0) - negative_visible.sum(axis = 0))

                batch_errors.append(np.mean((batch - negative_visible) ** 2))

            self.training_errors.append(np.mean(batch_errors))
            
            # Reconstruct a batch of images from the training set
            self.training_reconstructions.append(self._reconstruct(X[:25])) 
            if np.mean(batch_errors)*1000 < 1: return
            
    def _reconstruct(self, X):
        positive_hidden = sigma(X.dot(self.W) + self.h0)
        hidden_states = self._sample(positive_hidden)
        negative_visible = sigma(hidden_states.dot(self.W.T) + self.v0)
        return negative_visible
    
 ##################CLASSIFICATION#########################    
    def _CD1_Classification(self, X, y, output):

        for _ in tqdm(range(self.n_iterations), disable = output):
            batch_errors = []
            label_errors = []
            
            for batch in batch_iterator(X, y, batch_size = self.batch_size):
                visX = batch[0]
                visY = self._eigenvector(batch[1])    
                
                # Positive phase 
                hidden_states = self._sample_hiddensC(visX, visY)
                positive_associationsW = np.outer(visX,hidden_states)
                positive_associationsU = np.outer(visY,hidden_states)
                
                # Negative phase     
                visRecon = self._sample_visibles(hidden_states)
                targetRecon = self._sample_visiblesC(hidden_states)
                negative_hidden = self._mean_hiddensC(visRecon, targetRecon)   
                negative_associationsW = np.outer(visRecon,negative_hidden)       
                negative_associationsU = np.outer(targetRecon,negative_hidden)        
                
                self.W  += self.lr * (positive_associationsW - negative_associationsW)
                self.U  += self.lr * (positive_associationsU - negative_associationsU)
                self.h0 += self.lr * (hidden_states.sum(axis = 0) - negative_hidden.sum(axis = 0))
                self.v0 += self.lr * (visX.sum(axis = 0) - visRecon.sum(axis = 0))
                self.z0 += self.lr * (visY.sum(axis = 0) - targetRecon.sum(axis = 0))
                
                batch_errors.append(np.mean((visX - visRecon) ** 2))
                label_errors.append(np.mean((visY - targetRecon) ** 2))
            
            self.training_errors.append(np.mean(batch_errors))
                
    def _mean_hiddensC(self, v, y):
        #Computes the probabilities P(h=1|v). 
       
        return sigma(np.inner(v, self.W.T) + np.inner(y, self.U.T) + self.h0)
    
    
    def _sample_hiddensC(self, v, y):
        #Sample from the distribution P(h|v).

        p = self._mean_hiddensC(v,y)
        return self._sample(p)
    
    def _sample_visiblesC(self, h):
        #Sample from the distribution P(v|h).
        
        p = sigma(np.inner(h, self.U) + self.z0)
        return self._sample(p)
                
    def _eigenvector(self, n):
        canonic_base = np.zeros(self.n_label)
        canonic_base[n] = 1
        return canonic_base
    
    def _predict(self,testsetX):
        labels = np.zeros(len(testsetX))

        for i in range(len(testsetX)):
            reconY = self._classify(testsetX[i])
            labels[i] = reconY.argmax(axis=0) 
        return labels
    
    def _classify(self, testSampleX):
        #clamped visibile nodes of the image
        hidden = sigma(np.inner(testSampleX, self.W.T) + self.h0)
        target = sigma(np.inner(hidden, self.U) + self.z0)
        
        return target
    
    def _generate(self, label):
        #clamped label nodes
        label = self._eigenvector(label)
        hidden = sigma(np.inner(label, self.U.T) + self.h0)
        img = sigma(np.inner(hidden, self.W) + self.v0)
        
        return img    
    
    # Computes the free energy of a given visible vector (formula due to Hinton "Practical Guide ...") 
    def _compute_free_energy(self, visibleX, visibleY):
        x = np.zeros(self.n_hidden)
        for j in range(self.n_hidden):
            x[j] = self.h0[j] + np.inner(np.transpose(visibleX),self.W[:,j])+ np.inner(np.transpose(visibleY),self.U[:,j])
        return (-np.inner(visibleX,self.v0) - sum([max(0,x[i]) for i in range(len(x))]))
    
    """
    Performs classification on given dataset
    uses free energy method (Hinton, p.17)
    return predicted labels
    """
    def _predictF_E(self,testsetX):
        labels = np.zeros(len(testsetX))
        #print testsetX.shape
        for i in range(len(testsetX)):
            min_fe = 99999
            label_min_fe = None
            visibleX = testsetX[i]
            #for each label
            for j in range(self.n_label):
                visibleY = np.zeros(self.n_label)
                visibleY[j] = 1
                #print visibleY
                #compute free energy
                fe = self._compute_free_energy(visibleX, visibleY)
                if fe < min_fe:
                    min_fe = fe
                    label_min_fe = j
            #returns label with minimal free energy
            labels[i] = label_min_fe
        return labels
    
def grayToBlack(img, eps):
    for i in range(len(img)):
        img[i] = 0 if img[i]<eps else 1
    return img

def loadRBM(path, classifier = False):
    f = open(path, "rb")
    aux = pickle.load(f)
    rbm = RBM(n_visible = aux["n_visible"] ,
              n_hidden = aux["n_hidden"], 
              n_iterations = aux["n_iterations"], 
              batch_size = aux["batch_size"], 
              learning_rate = aux["lr"])
    
    rbm.training_errors = aux["training_errors"]
    rbm.training_reconstructions = aux["training_reconstructions"]
    rbm.W = aux["W"]
    rbm.v0 = aux["v0"]
    rbm.h0 = aux["h0"]
    
    if classifier:
            rbm.n_label = aux["n_label"]
            rbm.U = aux["U"]
            rbm.z0 = aux["z0"]
    
    return rbm

def training_error_fixed_h(n_hidden,learning_rate,label = ""):    
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))
    plt.suptitle("Training error plot", fontsize = 25)
    for i in range(len(n_hidden)):
        h = n_hidden[i]
        l = i//2
        k = 0 if i%2== 0 else 1
        for j in range(len(learning_rate)):        
            lr = learning_rate[j]
            path = "trained/"+label+"RBM_"+str(h)+"_hidden_"+str(lr)+"_lr_"+".csv"
            rbm_loaded = loadRBM(path)
            ax[l,k].plot(range(len(rbm_loaded.training_errors)), rbm_loaded.training_errors, label = "lr:"+str(lr))
            ax[l,k].set_title(str(h)+" hidden nodes", fontsize = 16)
            ax[l,k].legend(loc = 1)
            ax[l,k].set_xlim([0, 10])
            ax[l,k].set_ylim([0, 0.4])   
            
    plt.show()
    plt.close()
    
def training_error_fixed_lr(best_lr,n_hidden,label = ""):
    for h in n_hidden:        
            path = "trained/"+label+"RBM_"+str(h)+"_hidden_"+str(best_lr)+"_lr_"+".csv"
            rbm_loaded = loadRBM(path)
            plt.plot(range(len(rbm_loaded.training_errors)), rbm_loaded.training_errors, label = str(h)+" hidden nodes")
            plt.legend(loc = 1)
            plt.xlim([0, 10])
            plt.ylim([0, 0.2])  
            
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    
    plt.show()
    
def accuracy(rbm, testdata, testlabel):
    acc = 0
    k = rbm._predict(testdata)
    for i in range(len(testlabel)):
        if k[i] == testlabel[i]:
            acc += 1
    return round(acc/len(testlabel)*100,2)

def plotAccuracy(label = "_Rad_"):
    plt.figure(figsize=(40, 10))
    learning_rate = (1, 0.1, 0.01, 0.001, 0.0001)
    n_hidden = (5, 10, 20, 30, 40, 50)
    it = [5,10,15]
    plt.suptitle("Plot accuracy", fontsize = 16)
    for i in range(len(it)):
        ax = plt.subplot(1, len(it), i+1)
        
        for h in n_hidden:
            acc = []
            for lr in learning_rate:
                path = "trained/"+str(it[i])+label+"Class_RBM_"+str(h)+"_hidden_"+str(lr)+"_lr_"+".csv"
                rbm_loaded = loadRBM(path, classifier = True)
                acc.append(accuracy(rbm_loaded, testdata, testlabel))
            ax.set_title(str(it[i])+" iterations", fontsize = 12)    
            ax.plot([0,4,8,12,16] ,acc ,'-o',label = "n° hidden: " + str(h))
            ax.set_xticks([0,4,8,12,16])
            ax.set_ylim([-2,102])
            ax.set_xticklabels(learning_rate)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Learning rate')
            ax.legend()