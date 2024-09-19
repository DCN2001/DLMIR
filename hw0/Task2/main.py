import os
import numpy as np
import argparse
import joblib
from tqdm import tqdm
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

#The class of traditional ML method
class Inst_Classifier():
    def __init__(self, args):
        self.datapath = args.datapath
        if args.pooling =="standard":
            self.scaler = StandardScaler()
        elif args.pooling == "normalize":
            self.scaler = MinMaxScaler()
        #ADSR
        self.adsr = args.adsr
        #Create label encoder
        self.classes = ["bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "synth_lead", "vocal"]
        self.label_encoder = LabelEncoder()
        self.encoded_label = self.label_encoder.fit_transform(self.classes)
        #Model initializing
        if args.model=="SVM":
            self.model = svm.SVC(probability=True)
        elif args.model=="RF":
            self.model = RandomForestClassifier(random_state=1)   
        
        self.mtx_savepath = os.path.join("./conf_mtx",args.model+"_"+args.pooling+"_"+args.adsr+".png")

    def load_data(self,datapath):
        #Load path of both preprocessed trainset and testset
        train_path = os.path.join(datapath,"train")
        test_path = os.path.join(datapath, "test")

        X_train, Y_train, X_test, Y_test = [], [], [], []
        instru_list = os.listdir(train_path)        #All the instrument classes 
        for instru in tqdm(instru_list, desc="Loading data", colour="red"):
            audiolist_train = os.listdir(os.path.join(train_path,instru))   #The audio under the instrument folder of trainset
            for audio in audiolist_train:
                feature = np.load(os.path.join(train_path,instru,audio))        #Load the pre-extracted features
                #Select the adsr feature or not for training 
                if self.adsr=="on":
                    X_train.append(feature)
                else:
                    X_train.append(feature[:-3])
                #Label 
                Y_train.append(instru)
            
            audiolist_test = os.listdir(os.path.join(test_path,instru))
            for audio in audiolist_test:
                feature = np.load(os.path.join(test_path,instru,audio))          #Load the pre-extracted features
                #Select the adsr feature or not (correspond to that in training phase) 
                if self.adsr=="on":
                    X_test.append(feature)
                else:
                    X_test.append(feature[:-3])
                #Label 
                Y_test.append(instru)

        X_train, Y_train, X_test, Y_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)   #Transform from list to np.array
        return X_train, Y_train, X_test, Y_test
    
    #Function for plotting confusion matrix
    def plot_conf_mtx(self, conf_mtx, classes, savepath):
        plt.figure(figsize=(10,7))
        sns.heatmap(conf_mtx, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
        plt.title("Confusion matrix")
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(savepath)


    def train(self):
        X_train, Y_train, X_test, Y_test = self.load_data(self.datapath)       #Call the funciton for loading data  
        #Feature pooling
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        #Label encoding
        Y_train_encoded = self.label_encoder.transform(Y_train)
        Y_test_encoded = self.label_encoder.transform(Y_test)
        #Training
        print("\n Training model.....")
        self.model.fit(X_train, Y_train_encoded)

        #Directly evaluate on test set
        print("\n Predicting......")
        Y_pred_encoded = self.model.predict(X_test)
        top1_acc = accuracy_score(Y_test_encoded,Y_pred_encoded)     #Top1 acc

        y_prob = self.model.predict_proba(X_test)       #The predicted probability 
        top3 = np.argsort(y_prob, axis=1)[:, -3:]       #The top 3 probability
        top3_acc = 0    
        #If the top 3 classes have one match to the label then accumulate top3 acc
        for i in range(top3.shape[0]):
            if np.isin(Y_test_encoded[i],top3[i,:]):
                top3_acc += 1
        top3_acc = top3_acc/top3.shape[0]

        #Confusion matrix
        Y_pred = self.label_encoder.inverse_transform(Y_pred_encoded)
        Y_labels = self.label_encoder.inverse_transform(Y_test_encoded)
        conf_matrix = confusion_matrix(Y_labels, Y_pred)
        
        #Show result
        print(f"top 1 acc: {100*top1_acc:.2f}%")
        print(f"top 3 acc: {100*top3_acc:.2f}%")
        print("Confusion matrix: ")
        print(conf_matrix)
        self.plot_conf_mtx(conf_matrix,self.classes,self.mtx_savepath)


#-------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="/mnt/gestalt/home/dcn2001/NSynth_pre_task2", help="NSynth feature path")
    parser.add_argument("--model", type=str, default="SVM", help="SVM or RF(Random Forest)")
    parser.add_argument("--pooling", type=str, default="normalize", help="SVM or RF(Random Forest)")
    parser.add_argument("--adsr", type=str, default="on", help="Use adsr feature or not")
    args = parser.parse_args()

    model = Inst_Classifier(args)
    model.train()