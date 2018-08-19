import numpy as np
from keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler,Normalizer,LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.utils import to_categorical
from keras.models import Model,load_model
from keras.optimizers import SGD,Adam
from data_process import data_load

scaler = MinMaxScaler(feature_range=(0, 1))
encode_l = LabelEncoder()
# scaler = Normalizer()
def create_rnn(input_data, output_data,i):
    print("训练第" + str(i + 1) + "个标签")
    # data
    input_data = np.array(input_data)
    output_data = np.array(output_data[:,i])

    output_data = encode_l.fit_transform(output_data)
    class_num_lis = output_data[np.argmax(output_data)]+1
    input_data = np.reshape(input_data,(-1,3))
    output_data = np.reshape(output_data,(-1,1))
    in_train,in_test,out_train,out_test = train_test_split(input_data, output_data, test_size = 0.3, random_state=0)
    out_train = to_categorical(out_train, num_classes=class_num_lis)
    out_test = to_categorical(out_test, num_classes=class_num_lis)
    in_train = scaler.fit_transform(in_train)
    in_test = scaler.transform(in_test)
    in_train = np.reshape(in_train, (in_train.shape[0], 1, in_train.shape[1]))
    in_test = np.reshape(in_test, (in_test.shape[0], 1, in_test.shape[1]))
    out_train = scaler.fit_transform(out_train)
    out_test = scaler.transform(out_test)
    # build model
    model = text_cnn(d_i=class_num_lis)
    batch_size = 4
    epochs = 50
    model.fit(in_train, out_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              shuffle=True)
    model.save('D:\\test_data\\validate_data\\model'+str(i+1)+'.h5')
    scores = model.evaluate(in_test, out_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    del model

def text_cnn(d_i=6):

    # Inputs(,3)
    comment_seq = Input(shape=[1,3], name='x_seq')
    # Embeddings layers(1,1)
    # emb_comment = Embedding(max_features, embed_size)(comment_seq)
    #conv layers
    convs = []
    filter_sizes = [3, 4]
    for i,fsz in enumerate(filter_sizes):
        l_conv = Conv1D(filters=3, kernel_size=fsz, activation='relu', padding="same")(comment_seq)  # (4,4)
        l_pool = MaxPooling1D(1,1, padding="same")(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)

    merge = concatenate(convs, axis=1)  # Flatten()
    # out = Dropout(0.5)(merge)
    output = Dense(d_i, activation='softmax')(merge)
    model = Model([comment_seq], output)
    #     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # print(model.summary())
    lr = [1e-2,1e-2,5e-4]
    adam = Adam(lr=lr[i])
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])
    return model

# input_data, output_data = data_load('D:\\test_data\\validate_data\\data.csv')
# for i in range(0,3):
#     # if i==2:
#         create_rnn(input_data, output_data,i)
def test(list1=[7 ,11 ,135]):
    input_data, output_data = data_load('D:\\test_data\\validate_data\\data.csv')
    print("predict in true scence,,,,,,,,,,,,,,,,,,,,,,,,,")
    ls = []
    for i in range(0,3):
        # data
        input_data = np.array(input_data)
        output_data1 = np.array(output_data[:,i])
        output_data1 = encode_l.fit_transform(output_data1)
        input_data = np.reshape(input_data, (-1, 3))
        output_data1 = np.reshape(output_data1, (-1, 1))
        in_train, in_test, out_train, out_test = train_test_split(input_data, output_data1, test_size=0.3, random_state=0)
        in_train = scaler.fit_transform(in_train)
        in_test = scaler.transform(in_test)
        a = np.array([list1])
        a = scaler.transform(a)

        model = load_model('D:\\test_data\\validate_data\\model'+str(i+1)+'.h5')
        a = np.reshape(a,(a.shape[0],1,a.shape[1]))
        pre_y = model.predict(a)
        y = [np.argmax(x) for x in list(pre_y)]
        y = encode_l.inverse_transform(y)

        ls.append(y[0])
    print(ls)
    if ls[0]==7:
        return 0
    else:
        return test(ls)
test()
"""
8 10 2004 6 2 24
7 0 9064 8 10 24
8 10 9064 6 2 24
1 0 7732 8 10 24
8 10 7732 6 2 24
7 11 135 9 0 20
9 0 146 5 4 21
5 4 242 3 11 22
3 11 233 1 2 23
1 4 6056 8 10 24
8 10 6056 6 2 24
1 0 3262 8 10 24
8 10 3262 6 2 24
"""