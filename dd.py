from collections import deque
import numpy as np
import pandas as pd
import math
from keras import initializers
from sklearn.preprocessing import MinMaxScaler,Normalizer
from keras.optimizers import Adam,Adagrad
from keras.layers import Dense, LSTM, TimeDistributed, Dropout,Activation,RepeatVector,Embedding,Input,Conv2D,Concatenate
from keras.layers.normalization import Layer
from keras.models import Sequential,Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_layer_normalization import LayerNormalization
from numpy import *
from keras import backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.externals import joblib


'''
def model_train(x,y):
    model = Sequential()
    model.add(Dense(32,input_shape=(x.shape[1],x.shape[2])))
    model.add(LayerNormalization())
    model.add(LSTM(128,activation='softsign'))
    model.add(Dropout(0.5))
    model.add(RepeatVector(50))
    model.add(LayerNormalization())
    model.add(LSTM(128,activation='softsign',return_sequences=True))
    model.add(TimeDistributed(Dense(7)))
    model.compile(loss='mse', optimizer=Adam(lr=0.0005,clipnorm=1.))
    model.summary()
    # fit network
    model.fit(x, y, epochs=1, batch_size=128,verbose=1, shuffle=False)
    return model      
'''
def model_train(x1,x2,x3,x4,x5,x6,x7,y1,y2,y3,y4,y5,y6,y7):
    egostate = Input(shape = (x1.shape[1],x1.shape[2]))
    lfstate = Input(shape = (x2.shape[1],x2.shape[2]))
    fstate = Input(shape = (x3.shape[1],x3.shape[2]))
    rfstate = Input(shape = (x4.shape[1],x4.shape[2]))
    lbstate = Input(shape = (x5.shape[1],x5.shape[2]))
    bstate = Input(shape = (x6.shape[1],x6.shape[2]))
    rbstate = Input(shape = (x7.shape[1],x7.shape[2]))
    egostate1 = Dense(32)(egostate)
    lfstate1 = Dense(32)(lfstate)
    fstate1 = Dense(32)(fstate)
    rfstate1 = Dense(32)(rfstate)
    lbstate1 = Dense(32)(lbstate)
    bstate1 = Dense(32)(bstate)
    rbstate1 = Dense(32)(rbstate)
    egostate1 = LayerNormalization()(egostate1)
    lfstate1 = LayerNormalization()(lfstate1)
    fstate1 = LayerNormalization()(fstate1)
    rfstate1 = LayerNormalization()(rfstate1)
    lbstate1 = LayerNormalization()(lbstate1)
    bstate1 = LayerNormalization()(bstate1)
    rbstate1 = LayerNormalization()(rbstate1)
    encoder1 = LSTM(128,activation = 'softsign', return_sequences=True, return_state = True)
    egostate1,ego_h,ego_c = encoder1(egostate1)
    lfstate1,lf_h,lf_c = encoder1(lfstate1)
    fstate1,f_h,f_c = encoder1(fstate1)
    rfstate1,rf_h,rf_c = encoder1(rfstate1)
    lbstate1,lb_h,lb_c = encoder1(lbstate1)
    bstate1,b_h,b_c = encoder1(bstate1)
    rbstate1,rb_h,rb_c = encoder1(rbstate1)
    egoin = Concatenate()([lbstate1,lfstate1,bstate1,egostate1,fstate1,rbstate1,rfstate1])
    lfin = Concatenate()([lbstate1,lfstate1,egostate1,fstate1])
    fin = Concatenate()([lfstate1,egostate1,fstate1,rfstate1])
    rfin = Concatenate()([egostate1,fstate1,rbstate1,rfstate1])
    lbin = Concatenate()([lbstate1,lfstate1,bstate1,egostate1])
    bein = Concatenate()([lbstate1,bstate1,egostate1,rbstate1])
    rbin = Concatenate()([bstate1,egostate1,rbstate1,rfstate1])
    egoin = LayerNormalization()(egoin)
    lfin = LayerNormalization()(lfin)
    fin = LayerNormalization()(fin)
    rfin = LayerNormalization()(rfin)
    lbin = LayerNormalization()(lbin)
    bein = LayerNormalization()(bein)
    rbin = LayerNormalization()(rbin)
    encoder2 = LSTM(128,activation = 'softsign', return_state=True)
    encoder3 = LSTM(128,activation = 'softsign',  return_state=True)
    egoout,ego_h1,ego_c1 = encoder2(egoin)
    lfout,lf_h1,lf_c1 = encoder3(lfin)
    fout,f_h1,f_c1 = encoder3(fin)
    rfout,rf_h1,rf_c1 = encoder3(rfin)
    lbout,lb_h1,lb_c1 = encoder3(lbin)
    bout,b_h1,b_c1 = encoder3(bein)
    rbout,rb_h1,rb_c1 = encoder3(rbin)
    ego_state = [ego_h1,ego_c1]
    lf_state = [lf_h1,lf_c1]
    f_state = [f_h1,f_c1]
    rf_state = [rf_h1,rf_c1]
    lb_state = [lb_h1,lb_c1]
    b_state = [b_h1,b_c1]
    rb_state = [rb_h1,rb_c1]
    egoout = RepeatVector(5)(egoout)
    lfout = RepeatVector(5)(lfout)
    fout = RepeatVector(5)(fout)
    rfout = RepeatVector(5)(rfout)
    lbout = RepeatVector(5)(lbout)
    bout = RepeatVector(5)(bout)
    rbout = RepeatVector(5)(rbout)
    egoout = Dense(32)(egoout)
    lfout = Dense(32)(lfout)
    fout = Dense(32)(fout)
    rfout = Dense(32)(rfout)
    lbout = Dense(32)(lbout)
    bout = Dense(32)(bout)
    rbout = Dense(32)(rbout)
    egoout = LayerNormalization()(egoout)
    lfout = LayerNormalization()(lfout)
    fout = LayerNormalization()(fout)
    rfout = LayerNormalization()(rfout)
    lbout = LayerNormalization()(lbout)
    bout = LayerNormalization()(bout)
    rbout = LayerNormalization()(rbout)
    decoder1 = LSTM(128,activation = 'softsign', return_sequences=True)
    egoout = decoder1(egoout,initial_state=ego_state)
    lfout = decoder1(lfout,initial_state=lf_state)
    fout = decoder1(fout,initial_state=f_state)
    rfout = decoder1(rfout,initial_state=rf_state)
    lbout = decoder1(lbout,initial_state=lb_state)
    bout = decoder1(bout,initial_state=b_state)
    rbout = decoder1(rbout,initial_state=rb_state)
    egode = Concatenate()([lbout,lfout,bout,egoout,fout,rbout,rfout])
    lfde = Concatenate()([lbout,lfout,egoout,fout])
    fde = Concatenate()([lfout,egoout,fout,rfout])
    rfde = Concatenate()([egoout,fout,rbout,rfout])
    lbde = Concatenate()([lbout,lfout,bout,egoout])
    bde = Concatenate()([lbout,bout,egoout,rbout])
    rbde = Concatenate()([bout,egoout,rbout,rfout])
    egode = LayerNormalization()(egode)
    lfde = LayerNormalization()(lfde)
    fde = LayerNormalization()(fde)
    rfde = LayerNormalization()(rfde)
    lbde = LayerNormalization()(lbde)
    bde = LayerNormalization()(bde)
    rbde = LayerNormalization()(rbde)
    decoder3 = LSTM(128,activation = 'softsign', return_sequences=True)
    decoder2 = LSTM(128,activation = 'softsign', return_sequences=True)
    egode = decoder3(egode)
    lfde = decoder2(lfde)
    fde = decoder2(fde)
    rfde = decoder2(rfde)
    lbde = decoder2(lbde)
    bde = decoder2(bde)
    rbde = decoder2(rbde)
    egode = TimeDistributed(Dense(1))(egode)
    lfde = TimeDistributed(Dense(1))(lfde)
    fde = TimeDistributed(Dense(1))(fde)    
    rfde = TimeDistributed(Dense(1))(rfde)    
    lbde = TimeDistributed(Dense(1))(lbde)    
    bde = TimeDistributed(Dense(1))(bde)    
    rbde = TimeDistributed(Dense(1))(rbde)
    model = Model([egostate,lfstate,fstate,rfstate,lbstate,bstate,rbstate], [egode,lfde,fde,rfde,lbde,bde,rbde])
    model.compile(loss='mse', optimizer=Adam(lr=0.0005,clipnorm=1.))
    model.summary()
    # fit network
    model.fit([x1,x2,x3,x4,x5,x6,x7], [y1,y2,y3,y4,y5,y6,y7], epochs=200, batch_size=128,verbose=1, shuffle=False)
    return model 
'''
def rmse(y_true, y_pred):
    loss = K.sqrt(K.mean((K.square(y_pred[:,:2] - y_true[:,:2])*5)+(K.square(y_pred[:,2:4] - y_true[:,2:4])*10)+(K.square(y_pred[:,4:] - y_true[:,4:])*1)))
    return loss
'''

data1 = pd.read_csv('processdata1/i-80.1sv.csv',usecols = [6,8,9,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46])
data2 = pd.read_csv('processdata1/i-80.2sv.csv',usecols = [6,8,9,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46])
data3 = pd.read_csv('processdata1/i-80.3sv.csv',usecols = [6,8,9,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46])
data4 = pd.read_csv('processdata1/us-101.1sv.csv',usecols = [6,8,9,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46])
data5 = pd.read_csv('processdata1/us-101.2sv.csv',usecols = [6,8,9,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46])
#data6 = pd.read_csv('processdata1/us-101.3sv.csv',usecols = [6,8,9,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46])
data7 = pd.read_csv('data/i-80.1sv.csv',usecols = [8,10,11,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48])
data8 = pd.read_csv('data/i-80.2sv.csv',usecols = [8,10,11,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48])
data9 = pd.read_csv('data/i-80.3sv.csv',usecols = [8,10,11,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48])
data10 = pd.read_csv('data/us-101.1sv.csv',usecols = [7,9,10,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])
data11 = pd.read_csv('data/us-101.2sv.csv',usecols = [7,9,10,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])
#data12 = pd.read_csv('data/us-101.3sv.csv',usecols = [7,9,10,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])
data13 = pd.read_csv('datasv/i-80.1sv.csv',usecols = [8,10,11,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48])
data14 = pd.read_csv('datasv/i-80.2sv.csv',usecols = [8,10,11,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48])
data15 = pd.read_csv('datasv/i-80.3sv.csv',usecols = [8,10,11,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48])
data16 = pd.read_csv('datasv/us-101.1sv.csv',usecols = [7,9,10,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])
#data17 = pd.read_csv('datasv/us-101.3sv.csv',usecols = [7,9,10,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])
data1 = data1.to_numpy()
data2 = data2.to_numpy()
data3 = data3.to_numpy()
data4 = data4.to_numpy()
data5 = data5.to_numpy()
#data6 = data6.to_numpy()
data7 = data7.to_numpy()
data8 = data8.to_numpy()
data9 = data9.to_numpy()
data10 = data10.to_numpy()
data11 = data11.to_numpy()
#data12 = data12.to_numpy()
data13 = data13.to_numpy()
data14 = data14.to_numpy()
data15 = data15.to_numpy()
data16 = data16.to_numpy()
#data17 = data17.to_numpy()
'''
a1 = int(len(data1)*0.8)
b1 = int(len(data2)*0.8)
c1 = int(len(data3)*0.8)
d1 = int(len(data4)*0.8)
e1 = int(len(data5)*0.8)
f1 = int(len(data6)*0.8)
dataa1 = data1[:a1]
dataa2 = data1[a1:]
datab1 = data2[:b1]
datab2 = data2[b1:]
datac1 = data3[:c1]
datac2 = data3[c1:]
datad1 = data4[:d1]
datad2 = data4[d1:]
datae1 = data5[:e1]
datae2 = data5[e1:]
dataf1 = data6[:f1]
dataf2 = data6[f1:]
data = np.vstack((dataa1,datab1,datac1,datad1,datae1,dataa2,datab2,datac2,datad2,datae2))
'''
data = np.vstack((data1,data2,data3,data4,data5,data7,data8,data9,data10,data11,data13,data14,data15,data16))

data = data.astype('float32')
data = data[:,1:]

data = data.reshape(-1,4200)
data = np.random.permutation(data)
data_in = data[:,:2800]
data_out = data[:,2800:]
data_in = data_in.reshape(-1,28)
data_out = data_out.reshape(-1,28)

egox = np.array(data_in[:,1]).reshape(-1,1)
egoy = np.array(data_in[:,0]).reshape(-1,1)
egovx = np.array(data_in[:,2]).reshape(-1,1)
egovy = np.array(data_in[:,3]).reshape(-1,1)
lfx = np.array(data_in[:,5]).reshape(-1,1)
lfy = np.array(data_in[:,4]).reshape(-1,1)
lfvx = np.array(data_in[:,6]).reshape(-1,1)
lfvy = np.array(data_in[:,7]).reshape(-1,1)
fx = np.array(data_in[:,9]).reshape(-1,1)
fy = np.array(data_in[:,8]).reshape(-1,1)
fvx = np.array(data_in[:,10]).reshape(-1,1)
fvy = np.array(data_in[:,11]).reshape(-1,1)
rfx = np.array(data_in[:,13]).reshape(-1,1)
rfy = np.array(data_in[:,12]).reshape(-1,1)
rfvx = np.array(data_in[:,14]).reshape(-1,1)
rfvy = np.array(data_in[:,15]).reshape(-1,1)
lbx = np.array(data_in[:,17]).reshape(-1,1)
lby = np.array(data_in[:,16]).reshape(-1,1)
lbvx = np.array(data_in[:,18]).reshape(-1,1)
lbvy = np.array(data_in[:,19]).reshape(-1,1)
bx = np.array(data_in[:,21]).reshape(-1,1)
by = np.array(data_in[:,20]).reshape(-1,1)
bvx = np.array(data_in[:,22]).reshape(-1,1)
bvy = np.array(data_in[:,23]).reshape(-1,1)
rbx = np.array(data_in[:,25]).reshape(-1,1)
rby = np.array(data_in[:,24]).reshape(-1,1)
rbvx = np.array(data_in[:,26]).reshape(-1,1)
rbvy = np.array(data_in[:,27]).reshape(-1,1)
data = np.hstack((egox,egoy,egovx,egovy,lfx,lfy,lfvx,lfvy,fx,fy,fvx,fvy,rfx,rfy,rfvx,rfvy,lbx,lby,lbvx,lbvy,bx,by,bvx,bvy,rbx,rby,rbvx,rbvy))
data = data.reshape(-1,2800)
data = np.hstack((data[:,252:280],data[:,532:560],data[:,812:840],data[:,1092:1120],data[:,1372:1400],data[:,1652:1680],data[:,1932:1960],data[:,2212:2240],data[:,2492:2520],data[:,2772:2800]))

egovx_out = np.array(data_out[:,2]).reshape(-1,1)
egovy_out = np.array(data_out[:,3]).reshape(-1,1)
lfvx_out = np.array(data_out[:,6]).reshape(-1,1)
lfvy_out = np.array(data_out[:,7]).reshape(-1,1)
fvx_out = np.array(data_out[:,10]).reshape(-1,1)
fvy_out = np.array(data_out[:,11]).reshape(-1,1)
rfvx_out = np.array(data_out[:,14]).reshape(-1,1)
rfvy_out = np.array(data_out[:,15]).reshape(-1,1)
lbvx_out = np.array(data_out[:,18]).reshape(-1,1)
lbvy_out = np.array(data_out[:,19]).reshape(-1,1)
bvx_out = np.array(data_out[:,22]).reshape(-1,1)
bvy_out = np.array(data_out[:,23]).reshape(-1,1)
rbvx_out = np.array(data_out[:,26]).reshape(-1,1)
rbvy_out = np.array(data_out[:,27]).reshape(-1,1)
vx = np.hstack((egovx_out,lfvx_out,fvx_out,rfvx_out,lbvx_out,bvx_out,rbvx_out))
vy = np.hstack((egovy_out,lfvy_out,fvy_out,rfvy_out,lbvy_out,bvy_out,rbvy_out))
vx = np.array(vx).reshape(-1,350)
vx = np.hstack((vx[:,63:70],vx[:,133:140],vx[:,203:210],vx[:,273:280],vx[:,343:350]))
vy = np.array(vy).reshape(-1,350)
vy = np.hstack((vy[:,63:70],vy[:,133:140],vy[:,203:210],vy[:,273:280],vy[:,343:350]))
data1 = np.hstack((data,vx))
data2 = np.hstack((data,vy))

train_numx = int(len(data1)*0.8)
datasetx = data1[:train_numx]
testsetx = data1[train_numx:]
test_numx = len(testsetx)
trainx1 , trainy1 = datasetx[:,:-35],datasetx[:,-35:]
testx1 , testy1 = testsetx[:,:-35],testsetx[:,-35:]
trainx1 = trainx1.reshape((trainx1.shape[0],10,28))
testx1 = testx1.reshape((testx1.shape[0],10,28))
trainy1 = trainy1.reshape((trainy1.shape[0],5,7))
testy1 = testy1.reshape((testy1.shape[0],5,7))

train_numy = int(len(data2)*0.8)
datasety = data2[:train_num2]
testsety = data2[train_num2:]
test_numy = len(testsety)
trainx2 , trainy2 = datasety[:,:-35],datasety[:,-35:]
testx2 , testy2 = testsety[:,:-35],testsety[:,-35:]
trainx2 = trainx2.reshape((trainx2.shape[0],10,28))
testx2 = testx2.reshape((testx2.shape[0],10,28))
trainy2 = trainy2.reshape((trainy2.shape[0],5,7))
testy2 = testy2.reshape((testy2.shape[0],5,7))

egoput1 = trainx1[:,:,:4]
lfput1 = trainx1[:,:,4:8]
fput1 = trainx1[:,:,8:12]
rfput1 = trainx1[:,:,12:16]
lbput1 = trainx1[:,:,16:20]
bput1 = trainx1[:,:,20:24]
rbput1 = trainx1[:,:,24:]
egoput2 = testx1[:,:,:4]
lfput2 = testx1[:,:,4:8]
fput2 = testx1[:,:,8:12]
rfput2 = testx1[:,:,12:16]
lbput2 = testx1[:,:,16:20]
bput2 = testx1[:,:,20:24]
rbput2 = testx1[:,:,24:]

egoout1 = trainy1[:,:,:1]
lfout1 = trainy1[:,:,1:2]
fout1 = trainy1[:,:,2:3]
rfout1 = trainy1[:,:,3:4]
lbout1 = trainy1[:,:,4:5]
bout1 = trainy1[:,:,5:6]
rbout1 = trainy1[:,:,6:]
egoout2 = testy1[:,:,:1]
lfout2 = testy1[:,:,1:2]
fout2 = testy1[:,:,2:3]
rfout2 = testy1[:,:,3:4]
lbout2 = testy1[:,:,4:5]
bout2 = testy1[:,:,5:6]
rbout2 = testy1[:,:,6:]

egoput1 = trainx2[:,:,:4]
lfput1 = trainx2[:,:,4:8]
fput1 = trainx2[:,:,8:12]
rfput1 = trainx2[:,:,12:16]
lbput1 = trainx2[:,:,16:20]
bput1 = trainx2[:,:,20:24]
rbput1 = trainx2[:,:,24:]
egoput2 = testx2[:,:,:4]
lfput2 = testx2[:,:,4:8]
fput2 = testx2[:,:,8:12]
rfput2 = testx2[:,:,12:16]
lbput2 = testx2[:,:,16:20]
bput2 = testx2[:,:,20:24]
rbput2 = testx2[:,:,24:]

egoout3 = trainy2[:,:,:1]
lfout3 = trainy2[:,:,1:2]
fout3 = trainy2[:,:,2:3]
rfout3 = trainy2[:,:,3:4]
lbout3 = trainy2[:,:,4:5]
bout3 = trainy2[:,:,5:6]
rbout3 = trainy2[:,:,6:]
egoout4 = testy2[:,:,:1]
lfout4 = testy2[:,:,1:2]
fout4 = testy2[:,:,2:3]
rfout4 = testy2[:,:,3:4]
lbout4 = testy2[:,:,4:5]
bout4 = testy2[:,:,5:6]
rbout4 = testy2[:,:,6:]

modelx = model_train(egoput1,lfput1,fput1,rfput1,lbput1,bput1,rbput1,egoout1,lfout1,fout1,rfout1,lbout1,bout1,rbout1)
modely = model_train(egoput3,lfput3,fput3,rfput3,lbput3,bput3,rbput3,egoout3,lfout3,fout3,rfout3,lbout3,bout3,rbout3)

trainpredictx1 = modelx.predict([egoput2,lfput2,fput2,rfput2,lbput2,bput2,rbput2])
trainpredictx  = np.array(trainpredictx1).reshape(-1,5)
prevx = trainpredictx[:15044,:]
prevxlf = trainpredictx[15044:30088,:]
prevxf = trainpredictx[30088:45132,:]
prevxrf = trainpredictx[45132:60176,:]
prevxlb = trainpredictx[60176:75220,:]
prevxb = trainpredictx[75220:90264,:]
prevxrb = trainpredictx[90264:,:]
egox = egox.reshape(-1,100)
egox0 = egox[60172:,99]

egox_out = np.array(data_out[:,1]).reshape(-1,50) 

[rows,cols] = prevx.shape
predictx = [None]*rows
for i in range(len(predictx)):
    predictx[i] = [0]*cols

for i in range(rows):
    for j in range(cols):
        if j == 0:
            predictx[i][j] = egox0[i]+1*prevx[i][j]
        else:
            predictx[i][j] = predictx[i][j-1]+prevx[i][j]*1

predictx = np.array(predictx).reshape(-1,5)

prex1 = np.array(predictx[:,0]).reshape(-1,1)
prex2 = np.array(predictx[:,1]).reshape(-1,1)
prex3 = np.array(predictx[:,2]).reshape(-1,1)
prex4 = np.array(predictx[:,3]).reshape(-1,1)
prex5 = np.array(predictx[:,4]).reshape(-1,1)
truthx1 = np.array(egox_out[60172:,9]).reshape(-1,1)
truthx2 = np.array(egox_out[60172:,19]).reshape(-1,1)
truthx3 = np.array(egox_out[60172:,29]).reshape(-1,1)
truthx4 = np.array(egox_out[60172:,39]).reshape(-1,1)
truthx5 = np.array(egox_out[60172:,49]).reshape(-1,1)

deltax1 = np.abs(prex1-truthx1)
deltax2 = np.abs(prex2-truthx2)
deltax3 = np.abs(prex3-truthx3)
deltax4 = np.abs(prex4-truthx4)
deltax5 = np.abs(prex5-truthx5)

meanx1 = np.mean(deltax1)
meanx2 = np.mean(deltax2)
meanx3 = np.mean(deltax3)
meanx4 = np.mean(deltax4)
meanx5 = np.mean(deltax5)


trainpredictx2 = modely.predict([egoput2,lfput2,fput2,rfput2,lbput2,bput2,rbput2])
trainpredicty  = np.array(trainpredictx2).reshape(-1,5)
prevy = trainpredicty[:15044,:]
prevylf = trainpredicty[15044:30088,:]
prevyf = trainpredicty[30088:45132,:]
prevyrf = trainpredicty[45132:60176,:]
prevylb = trainpredicty[60176:75220,:]
prevyb = trainpredicty[75220:90264,:]
prevyrb = trainpredicty[90264:,:]
egoy = egoy.reshape(-1,100)
egoy0 = egoy[60172:,99]

egoy_out = np.array(data_out[:,0]).reshape(-1,50) 

[rows,cols] = prevy.shape
predicty = [None]*rows
for i in range(len(predicty)):
    predicty[i] = [0]*cols

for i in range(rows):
    for j in range(cols):
        if j == 0:
            predicty[i][j] = egoy0[i]+1*prevy[i][j]
        else:
            predicty[i][j] = predicty[i][j-1]+prevy[i][j]*1

predicty = np.array(predicty).reshape(-1,5)

prey1 = np.array(predicty[:,0]).reshape(-1,1)
prey2 = np.array(predicty[:,1]).reshape(-1,1)
prey3 = np.array(predicty[:,2]).reshape(-1,1)
prey4 = np.array(predicty[:,3]).reshape(-1,1)
prey5 = np.array(predicty[:,4]).reshape(-1,1)
truthy1 = np.array(egoy_out[60172:,9]).reshape(-1,1)
truthy2 = np.array(egoy_out[60172:,19]).reshape(-1,1)
truthy3 = np.array(egoy_out[60172:,29]).reshape(-1,1)
truthy4 = np.array(egoy_out[60172:,39]).reshape(-1,1)
truthy5 = np.array(egoy_out[60172:,49]).reshape(-1,1)

deltay1 = np.abs(prey1-truthy1)
deltay2 = np.abs(prey2-truthy2)
deltay3 = np.abs(prey3-truthy3)
deltay4 = np.abs(prey4-truthy4)
deltay5 = np.abs(prey5-truthy5)

meany1 = np.mean(deltay1)
meany2 = np.mean(deltay2)
meany3 = np.mean(deltay3)
meany4 = np.mean(deltay4)
meany5 = np.mean(deltay5)

print(meanx1, meanx2, meanx3, meanx4, meanx5)
print(meany1, meany2, meany3, meany4, meany5)

a = 4





