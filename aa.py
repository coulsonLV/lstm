from collections import deque
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler,Normalizer
from keras.optimizers import Adam,Adagrad
from keras.layers import Dense, LSTM, TimeDistributed, Dropout,Activation,RepeatVector,Embedding,Input,Conv2D,Concatenate,
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
    egostate = Dense(32)(egostate)
    lfstate = Dense(32)(lfstate)
    fstate = Dense(32)(fstate)
    rfstate = Dense(32)(rfstate)
    lbstate = Dense(32)(lbstate)
    bstate = Dense(32)(bstate)
    rbstate = Dense(32)(rbstate)
    egostate = LayerNormalization()(egostate)
    lfstate = LayerNormalization()(lfstate)
    fstate = LayerNormalization()(fstate)
    rfstate = LayerNormalization()(rfstate)
    lbstate = LayerNormalization()(lbstate)
    bstate = LayerNormalization()(bstate)
    rbstate = LayerNormalization()(rbstate)
    encoder1 = LSTM(128,activation = 'softsign', return_sequences=True)
    egostate = encoder1(egostate)
    lfstate = encoder1(lfstate)
    fstate = encoder1(fstate)
    rfstate = encoder1(rfstate)
    lbstate = encoder1(lbstate)
    bstate = encoder1(bstate)
    rbstate = encoder1(rbstate)
    egoin = concatenate([lbstate,lfstate,bstate,egostate,fstate,rbstate,rfstate])
    lfin = concatenate([lbstate,lfstate,egostate,fstate])
    fin = concatenate([lfstate,egostate,fstate,rfstate])
    rfin = concatenate([egostate,fstate,rbstate,rfstate])
    lbin = concatenate([lbstate,lfstate,bstate,egostate])
    bein = concatenate([lbstate,bstate,egostate,rbstate])
    rbin = concatenate([bstate,egostate,rbstate,rfstate])
    egoin = LayerNormalization()(egoin)
    lfin = LayerNormalization()(lfin)
    fin = LayerNormalization()(fin)
    rfin = LayerNormalization()(rfin)
    lbin = LayerNormalization()(lbin)
    bein = LayerNormalization()(bein)
    rbin = LayerNormalization()(rbin)
    encoder2 = LSTM(128,activation = 'softsign', return_state=True)
    egoout,ego_h1,ego_c1 = encoder2(egoin)
    lfout,lf_h1,lf_c1 = encoder2(lfin)
    fout,f_h1,f_c1 = encoder2(fin)
    rfout,rf_h1,rf_c1 = encoder2(rfin)
    lbout,lb_h1,lb_c1 = encoder2(lbin)
    bout,b_h1,b_c1 = encoder2(bein)
    rbout,rb_h1,rb_c1 = encoder2(rbin)
    ego_state = [ego_h1,ego_c1]
    lf_state = [lf_h1,lf_c1]
    f_state = [f_h1,f_c1]
    rf_state = [rf_h1,rf_c1]
    lb_state = [lb_h1,lb_c1]
    b_state = [b_h1,b_c1]
    rb_state = [rb_h1,rb_c1]
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
    rf_out = decoder1(rfout,initial_state=rf_state)
    lbout = decoder1(lbout,initial_state=lb_state)
    bout = decoder1(bout,initial_state=b_state)
    rbout = decoder1(rbout,initial_state=rb_state)
    egode = concatenate([lbout,lfout,bout,egoout,fout,rbout,rfout])
    lfde = concatenate([lbout,lfout,egoout,fout])
    fde = concatenate([lfout,egoout,fout,rfout])
    rfde = concatenate([egoout,fout,rbout,rfout])
    lbde = concatenate([lbout,lfout,bout,egoout])
    bde = concatenate([lbout,bout,egoout,rbout])
    rbde = concatenate([bout,egoout,rbout,rfout])
    egode = LayerNormalization()(egode)
    lfde = LayerNormalization()(lfde)
    fde = LayerNormalization()(fde)
    rfde = LayerNormalization()(rfde)
    lbde = LayerNormalization()(lbde)
    bde = LayerNormalization()(bde)
    rbde = LayerNormalization()(rbde)
    decoder2 = LSTM(128,activation = 'softsign', return_sequences=True)
    egode = decoder2(egode)
    lfde = decoder2(lfde)
    fde = decoder2(fde)
    rfde = decoder2(rfde)
    lbde = decoder2(lbde)
    bde = decoder2(bde)
    rbde = decoder2(rde)
    model = Model([egostate,lfstate,fstate,rfstate,lbstate,bstate,rbstate], [egode,lfde,fde,rfde,lbde,bde,rbde])
    model.compile(loss=rmse, optimizer=Adam(lr=0.0005,clipnorm=1.))
    model.summary()
    # fit network
    model.fit([x1,x2,x3,x4,x5,x6,x7], [y1,y2,y3,y4,y5,y6,y7], epochs=1, batch_size=128,verbose=1, shuffle=False)
    return model 


def rmse(y_true, y_pred):
    loss = K.sqrt(K.mean(K.square(y_pred[:,:4] - y_true[:,:4])*10)+K.mean(K.square(y_pred[:,-12:] - y_true[:,-12:])*1))
    return loss

data1 = pd.read_csv('processdata/i-80.1sv.csv',usecols = [5,7,8,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])
data2 = pd.read_csv('processdata/i-80.2sv.csv',usecols = [5,7,8,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])
data3 = pd.read_csv('processdata/i-80.3sv.csv',usecols = [5,7,8,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])
data4 = pd.read_csv('processdata/us-101.1sv.csv',usecols = [5,7,8,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])
data5 = pd.read_csv('processdata/us-101.2sv.csv',usecols = [5,7,8,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])
data6 = pd.read_csv('processdata/us-101.3sv.csv',usecols = [5,7,8,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])
data1 = data1.to_numpy()
data2 = data2.to_numpy()
data3 = data3.to_numpy()
data4 = data4.to_numpy()
data5 = data5.to_numpy()
data6 = data6.to_numpy()
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
data = data.astype('float32')
data = data[:,1:]

data = data.reshape(-1,4200)
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
data = data[:,252:280]+data[:,532:560]+data[:,812:840]+data[:,1092:1120]+data[:,1372:1400]+data[:,1652:1680]+data[:,1932:1960]+data[:,2212:2240]+data[:,2492:2520]+data[:,2772:2800]

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
vx = vx[:,63:70]+vx[:,133:140]+vx[:,203:210]+vx[:,273:280]+vx[:,343:350]
vy = np.array(vy).reshape(-1,350)
vy = vy[:,63:70]+vy[:,133:140]+vy[:,203:210]+vy[:,273:280]+vy[:,343:350]
data1 = np.hstack((data,vx))
data2 = np.hstack((data,vy))

train_numx = int(len(data1)*0.8)
datasetx = data1[:train_numx]
datasetx = np.random.permutation(datasetx)
testsetx = data1[train_numx:]
test_numx = len(testsetx)
trainx1 , trainy1 = datasetx[:,:-35],datasetx[:,-35:]
testx1 , testy1 = testsetx[:,:-35],testsetx[:,-35:]
trainx1 = trainx1.reshape((trainx1.shape[0],10,28))
testx1 = testx1.reshape((testx1.shape[0],10,28))
trainy1 = trainy1.reshape((trainy1.shape[0],5,7))
testy1 = testy1.reshape((testy1.shape[0],5,7))

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

model = model_train([egoput1,lfput1,fput1,rfput1,lbput1,bput1,rbput1],[egoout1,lfout1,fout1,rfout1,lbout1,bout1,rbout1])

trainpredictx = model.predict([egoput2,lfput2,fput2,rfput2,lbput2,bput2,rbput2])
trainpredictx  = np.array(trainpredictx).reshape(-1,7)
prevx = np.array(trainpredictx[:,0]).reshape(-1,5)
prevxlf = np.array(trainpredictx[:,1]).reshape(-1,5)
prevxf = np.array(trainpredictx[:,2]).reshape(-1,5)
prevxrf = np.array(trainpredictx[:,3]).reshape(-1,5)
prevxlb = np.array(trainpredictx[:,4]).reshape(-1,5)
prevxb = np.array(trainpredictx[:,5]).reshape(-1,5)
prevxrb = np.array(trainpredictx[:,6]).reshape(-1,5)

egox = egox.reshape(-1,100)
egox0 = egox[3160:,99]
lfx = lfx.reshape(-1,100)
lfx0 = lfx[3160:,99]
fx = fx.reshape(-1,100)
fx0 = fx[3160:,99]
rfx = rfx.reshape(-1,100)
rfx0 = rfx[3160:,99]
lbx = lbx.reshape(-1,100)
lbx0 = lbx[3160:,99]
bx = bx.reshape(-1,100)
bx0 = bx[3160:,99]
rbx = rbx.reshape(-1,100)
rbx0 = rbx[3160:,99]

egox_out = np.array(data_out[:,1]).reshape(-1,50) 
egoy_out = np.array(data_out[:,0]).reshape(-1,50) 
lfx_out = np.array(data_out[:,5]).reshape(-1,50) 
lfy_out = np.array(data_out[:,4]).reshape(-1,50)
fx_out = np.array(data_out[:,9]).reshape(-1,50)
fy_out = np.array(data_out[:,8]).reshape(-1,50)
rfx_out = np.array(data_out[:,13]).reshape(-1,50)
rfy_out = np.array(data_out[:,12]).reshape(-1,50) 
lbx_out = np.array(data_out[:,17]).reshape(-1,50) 
lby_out = np.array(data_out[:,16]).reshape(-1,50)
bx_out = np.array(data_out[:,21]).reshape(-1,50) 
by_out = np.array(data_out[:,20]).reshape(-1,50) 
rbx_out = np.array(data_out[:,25]).reshape(-1,50) 
rby_out = np.array(data_out[:,24]).reshape(-1,50)

def calculate(vx, vx0, xout):
    [rows,cols] = vx.shape
    predictx = [None]*rows
    for i in range(len(predictx)):
        predictx[i] = [0]*cols

    for i in range(rows):
        for j in range(cols):
            if j == 0:
                predictx[i][j] = vx0[i]+1*vx[i][j]
            else:
                predictx[i][j] = predictx[i][j-1]+vx[i][j]*1

    predictx = np.array(predictx).reshape(-1,5)

    prex1 = np.array(predictx[:,0]).reshape(-1,1)
    prex2 = np.array(predictx[:,1]).reshape(-1,1)
    prex3 = np.array(predictx[:,2]).reshape(-1,1)
    prex4 = np.array(predictx[:,3]).reshape(-1,1)
    prex5 = np.array(predictx[:,4]).reshape(-1,1)
    truthx1 = np.array(xout[3160:,9]).reshape(-1,1)
    truthx2 = np.array(xout[3160:,19]).reshape(-1,1)
    truthx3 = np.array(xout[3160:,29]).reshape(-1,1)
    truthx4 = np.array(xout[3160:,39]).reshape(-1,1)
    truthx5 = np.array(xout[3160:,49]).reshape(-1,1)

    deltax1 = np.abs(prex1-truthx1)
    deltax2 = np.abs(prex2-truthx2)
    deltax3 = np.abs(prex3-truthx3)
    deltax4 = np.abs(prex4-truthx4)
    deltax5 = np.abs(prex5-truthx5)
    deltax1 = deltax1[deltax1 < 100]
    deltax2 = deltax2[deltax2 < 100] 
    deltax3 = deltax3[deltax3 < 100]
    deltax4 = deltax4[deltax4 < 100]
    deltax5 = deltax5[deltax5 < 100]

    a1,a2,a3,a4,a5 = len(deltax1),len(deltax2),len(deltax3),len(deltax4),len(deltax5)
    suma1,suma2,suma3,suma4,suma5,fca1,fca2,fca3,fca4,fca5 = 0,0,0,0,0,0,0,0,0,0
    for i in range(len(deltax1)):
        suma1 = suma1 + deltax1[i]
    for i in range(len(deltax2)):
        suma2 = suma2 + deltax2[i]
    for i in range(len(deltax3)):
        suma3 = suma3 + deltax3[i]
    for i in range(len(deltax4)):
        suma4 = suma4 + deltax4[i]
    for i in range(len(deltax5)):
        suma5 = suma5 + deltax5[i]
    meanx1 = suma1/a1
    meanx2 = suma2/a2
    meanx3 = suma3/a3
    meanx4 = suma4/a4
    meanx5 = suma5/a5
    '''
    for i in range(len(deltax1)):
        fca1 = (deltax1[i]-meanx1)**2+fca1
    for i in range(len(deltax2)):
        fca2 = (deltax2[i]-meanx2)**2+fca2
    for i in range(len(deltax3)):
        fca3 = (deltax3[i]-meanx3)**2+fca3
    for i in range(len(deltax4)):
        fca4 = (deltax4[i]-meanx4)**2+fca4
    for i in range(len(deltax5)):
        fca5 = (deltax5[i]-meanx5)**2+fca5
    fcx1 = fca1/a1
    fcx2 = fca2/a2
    fcx3 = fca3/a3
    fcx4 = fca4/a4
    fcx5 = fca5/a5
    '''
    return meanx1, meanx2, meanx3, meanx4, meanx5

megox1, megox2, megox3, megox4, megox5 = calculate(prevx, egox0, egox_out)
print(meanx1,meanx2,meanx3,meanx4,meanx5)
'''
mlfx1, mlfx2, mlfx3, mlfx4, mlfx5, fclfx1, fclfx2, fclfx3, fclfx4, fclfx5 = calculate(prevxlf, lfx0, lfx_out)
mfx1, mfx2, mfx3, mfx4, mfx5, fcfx1, fcfx2, fcfx3, fcfx4, fcfx5 = calculate(prevxf, fx0, fx_out)
mrfx1, mrfx2, mrfx3, mrfx4, mrfx5, fcrfx1, fcrfx2, fcrfx3, fcrfx4, fcrfx5 = calculate(prevxrf, rfx0, rfx_out)
mlbx1, mlbx2, mlbx3, mlbx4, mlbx5, fclbx1, fclbx2, fclbx3, fclbx4, fclbx5 = calculate(prevxlb, lbx0, lbx_out)
mbx1, mbx2, mbx3, mbx4, mbx5, fcbx1, fcbx2, fcbx3, fcbx4, fcbx5 = calculate(prevxb, bx0, bx_out)
mrbx1, mrbx2, mrbx3, mrbx4, mrbx5, fcrbx1, fcrbx2, fcrbx3, fcrbx4, fcrbx5 = calculate(prevxrb, rbx0, rbx_out)
meanx1 = (megox1+mlfx1+mfx1+mrfx1+mlbx1+mbx1+mrbx1)/7
meanx2 = (megox2+mlfx2+mfx2+mrfx2+mlbx2+mbx2+mrbx2)/7
meanx3 = (megox3+mlfx3+mfx3+mrfx3+mlbx3+mbx3+mrbx3)/7
meanx4 = (megox4+mlfx4+mfx4+mrfx4+mlbx4+mbx4+mrbx4)/7
meanx5 = (megox5+mlfx5+mfx5+mrfx5+mlbx5+mbx5+mrbx5)/7
fcx1 = (fcegox1+fclfx1+fcfx1+fcrfx1+fclbx1+fcbx1+fcrbx1)/7
fcx2 = (fcegox2+fclfx2+fcfx2+fcrfx2+fclbx2+fcbx2+fcrbx2)/7
fcx3 = (fcegox3+fclfx3+fcfx3+fcrfx3+fclbx3+fcbx3+fcrbx3)/7
fcx4 = (fcegox4+fclfx4+fcfx4+fcrfx4+fclbx4+fcbx4+fcrbx4)/7
fcx5 = (fcegox5+fclfx5+fcfx5+fcrfx5+fclbx5+fcbx5+fcrbx5)/7
print(fcx1,fcx2,fcx3,fcx4,fcx5)
'''






