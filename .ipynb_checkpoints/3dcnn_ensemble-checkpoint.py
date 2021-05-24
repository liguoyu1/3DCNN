import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json

from threadpool import *
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Input, Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, Input, average)
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
# from tensorflow.keras.utils import multi_gpu_model

from videoto3d import Videoto3D
from gapcalculator import calculate_gap
import tensorflow as tf
print(tf.__version__)
from keras.backend.tensorflow_backend import set_session
config= tf.ConfigProto()
# print("session:{}".format(tf.Session(config=config)))

config.gpu_options.per_process_gpu_memory_fraction = 1.0
print("\n\n\n\n\nconfig:{}\n\n\n\n\n".format(config))
session = tf.Session(config=config)
set_session(session)

print("\n\n\n\n\nsession:{}\n\n\n\n\n".format(session))

# exit(0)

def GAP(labels, predicts):
    pass


# from videoto3d import Videoto3D
# import videoto3d


def plot_history(history, result_dir, name):
    print("history: {}".format(history))
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '{}_accuracy.png'.format(name)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_loss.png'.format(name)))
    plt.close()


def save_history(history, result_dir, name):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result_{}.txt'.format(name)),
              'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def loadTestData(video_dir, vid3d, result_dir, nclass=82, depth = 32,color=False, skip=True, index=1, div=5):
    X = []
    sample_name = []
    img_rows, img_cols, frames = 32, 32, depth
    channel = 3 if color else 1
    # load video and frame extract to 3d
    data_dir = os.path.join(video_dir, "videos/video_5k/test_5k")
    files = os.listdir(data_dir)  # [:20]
    #print("files: {}".format(files))
    isExists=os.path.exists(result_dir)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目
        # 创建目录操作函数
        os.makedirs(result_dir) 
        print(result_dir+' 创建成功')
    
    files_size = len(files)
    profile_sz = int(files_size / div)
    end_index = int((index+1) * profile_sz)
    start_index = int(index *profile_sz)
    pbar = tqdm(total=len(files[start_index: end_index]))

    for filename in files[start_index: end_index]:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(data_dir, filename)
        X.append(vid3d.video3d(name, color=color, skip=skip))
        sample_name.append(filename)
    pbar.close()
    x = np.array(X).transpose((0, 2, 3, 1))
    if color:
        x = np.array(X).transpose((0, 2, 3, 4, 1))
    
    X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    X = X.astype('float32')
    fname_npz = os.path.join(result_dir,'dataset_{}_{}_{}_{}_{}.npz'.format(nclass, depth,
                                              skip, index, div))
    np.savez(fname_npz, X=X, Y=sample_name)
    print('Saved dataset to dataset.npz.')


def load_data_v2(video_dir, vid3d, result_dir, nclass=82, depth = 32,color=False, skip=True, index=1, div=5):
    X = []
    labels = []
    sample = {}
    label_index = {}
    img_rows, img_cols, frames = 32, 32, depth
    channel = 3 if color else 1
    # load label index
    with open(os.path.join(video_dir, 'label_id.txt'), 'r') as fp:
        for line in fp.readlines():
            items = line.split("\t")
            label_index[items[0]] = int(items[1])
    # load sample label
    with open(os.path.join(video_dir, "tagging/GroundTruth/tagging_info.txt"), 'r') as fp:
        for line in fp.readlines():
            items = line.strip().split("\t")
            sample[items[0]] = [label_index[v] for v in items[1].split(",")]

    # load video and frame extract to 3d
    data_dir = os.path.join(video_dir, "videos/video_5k/train_5k")
    files = os.listdir(data_dir) # [:100]
    print("files size: {}".format(len(files)))
    files_size = len(files)
    profile_sz = int(files_size / div)
    end_index = int((index+1) * profile_sz)
    start_index = int(index *profile_sz)
    pbar = tqdm(total=len(files[start_index: end_index]))

    for filename in files[start_index: end_index]:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(data_dir, filename)
        #print("all_filename: {}".format(name))
        label = sample[filename]
        labels.append(label)
        X.append(vid3d.video3d(name, color=color, skip=skip))
    pbar.close()
    
    fname_npz = os.path.join(result_dir,'dataset_{}_{}_{}_{}_{}.npz'.format(nclass, depth,
                                              skip, index, div))
    isExists=os.path.exists(result_dir)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目
        # 创建目录操作函数
        os.makedirs(result_dir) 
        print(result_dir+' 创建成功')
    
    x = np.array(X).transpose((0, 2, 3, 1))
    y = labels
    if color:
        x = np.array(X).transpose((0, 2, 3, 4, 1))
        y = labels
    #else:
    #    x, y = , labels
    X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    Y = label_transform(y, nclass)
    # print("labels : {}".format(Y))
    X = X.astype('float32')
    np.savez(fname_npz, X=X, Y=Y)
    print('Saved dataset to dataset.npz.')


def label_transform(labels, nb_class=82):
    size = len(labels)
    print("sample_size:{}".format(size))
    train_labels = np.zeros((size, nb_class))
    for num, label in enumerate(labels):
        for cla in label:
           train_labels[num][cla] = 1
    return train_labels



def loaddata(video_dir, vid3d, nclass, result_dir, color=False, skip=True,
             is_test=False):
    files = os.listdir(video_dir)
    X = []
    labels = []
    labellist = []

    pbar = tqdm(total=len(files))

    for filename in files:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(video_dir, filename)
        label = vid3d.get_UCF_classname(filename)
        if label not in labellist:
            if len(labellist) >= nclass:
                continue
            labellist.append(label)
        labels.append(label)
        X.append(vid3d.video3d(name, color=color, skip=skip))

    pbar.close()
    with open(os.path.join(result_dir, 'label_id.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels



def create_3dcnn(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        input_shape)))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3)))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3)))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='sigmoid'))
    print("output: {}".format(model.outputs))


    return model


def get_labels_dic(tag_dir):
    label_index = {}
    index_label = {}
    with open(os.path.join(tag_dir, 'label_id.txt'), 'r') as fp:
        for line in fp.readlines():
            items = line.split("\t")
            label_index[items[0]] = int(items[1])
            index_label[int(items[1])] = items[0]
    return index_label, label_index
    

def getTop_k(label_index, predicte, topk=20):
    result_value = []
    for key in label_index:
        result_value.append((key, predicte[label_index[key]]))
    
    result_sorted = sorted(result_value, key=lambda x: x[1], reverse=True)[: topk]
    labels = [v[0] for v in result_sorted]
    scores = [v[1] for v in result_sorted]
    return labels, scores


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='/home/tione/notebook/algo-2021/dataset/',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=82)
    parser.add_argument('--output', type=str, default="/home/tione/notebook/3dcnn/result/output_3dcnn_esamble_v3")
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--nmodel', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    img_rows, img_cols, frames = 32, 32, args.depth
    channel = 3 if args.color else 1

    vid3d = Videoto3D(img_rows, img_cols, frames)
    nb_classes = args.nclass
    threadsize = 5
    div_size = 25
    result_dir = os.path.join(args.output, 'frame_{}'.format(args.depth))
    if not os.path.exists(result_dir):
        pool = ThreadPool(threadsize)
        func_var = [([args.videos, vid3d, result_dir, args.nclass, args.depth,args.color, args.skip, i, div_size], None)for i in range(div_size)]
        requests = makeRequests(load_data_v2, func_var)
        [pool.putRequest(req) for req in requests]
        pool.wait()
        #exit(0)
    
    fname_npz = os.path.join(result_dir,'dataset_{}_{}_{}_{}_{}.npz'.format(args.nclass, args.depth,
                                              args.skip, 0, div_size))
    loadeddata = np.load(fname_npz)
    X=loadeddata["X"]
    Y=loadeddata["Y"]
    for i in range(1, div_size):
        fname_npz = os.path.join(result_dir,'dataset_{}_{}_{}_{}_{}.npz'.format(args.nclass, args.depth,
                                              args.skip, i, div_size))
        loadeddata = np.load(fname_npz)
        X=np.append(X, loadeddata["X"], axis=0)
        Y=np.append(Y,loadeddata["Y"], axis=0)
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=4)

    # Define model
    models = []
    for i in range(args.nmodel):
        print('model{}:'.format(i))
        models.append(create_3dcnn(X.shape[1:], nb_classes))
        models[-1].compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])
        history = models[-1].fit(X_train, Y_train, validation_data=(
            X_test, Y_test), batch_size=args.batch, epochs=args.epoch,
                                 verbose=1, shuffle=True)
        plot_history(history, args.output, i)
        save_history(history, args.output, i)

    model_inputs = [Input(shape=X.shape[1:]) for _ in range(args.nmodel)]
    model_outputs = [models[i](model_inputs[i]) for i in range(args.nmodel)]
    model_outputs = average(inputs=model_outputs)
    model = Model(inputs=model_inputs, outputs=model_outputs)
    # model = multi_gpu_model(model, 1)  # 设置 gpu 训练
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    # plot_model(model, show_shapes=True,
    #            to_file=os.path.join(args.output, 'model.png'))

    model_json = model.to_json()
    with open(os.path.join(args.output, 'tencent_3dcnnmodel.json'),
              'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, 'tencent_3dcnnmodel.hd5'))

    loss, acc = model.evaluate([X_test] * args.nmodel, Y_test, verbose=0)
    
    test_predictor = model.predict([X_test] * args.nmodel)
    gap = calculate_gap(test_predictor, Y_test)
    with open(os.path.join(args.output, 'result.txt'), 'w') as f:
        f.write('Test loss: {}\nTest accuracy:{}, Test Gap: {}'.format(loss, acc, gap))

    print('merged model:')
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    print('Test Gap : ', gap)
    
    print(" start test predict: ")
    test_frame_dir = os.path.join(args.output, "test_frame_{}".format(args.depth))
    if not os.path.exists(test_frame_dir):
        pool = ThreadPool(threadsize)
        func_var = [([args.videos, vid3d, test_frame_dir, args.nclass, args.depth,args.color, args.skip, i, div_size], None)for i in range(div_size)]
        requests = makeRequests(loadTestData, func_var)
        [pool.putRequest(req) for req in requests]
        pool.wait()
    fname_npz = os.path.join(test_frame_dir,'dataset_{}_{}_{}_{}_{}.npz'.format(args.nclass, args.depth,
                                              args.skip, 0, div_size))
    loadeddata = np.load(fname_npz)
    Test_X=loadeddata["X"]
    videoId_Y=loadeddata["Y"]
    for i in range(1, div_size):
        fname_npz = os.path.join(test_frame_dir,'dataset_{}_{}_{}_{}_{}.npz'.format(args.nclass, args.depth,
                                              args.skip, i, div_size))
        loadeddata = np.load(fname_npz)
        Test_X=np.append(Test_X, loadeddata["X"], axis=0)
        videoId_Y=np.append(videoId_Y,loadeddata["Y"], axis=0)

    predicte_results_test = model.predict([Test_X] * args.nmodel)
    index_labels, labels_index = get_labels_dic(args.videos)
    
    output_result = {}
    for video_name, result in zip(videoId_Y, predicte_results_test):
        cur_output = {}
        output_result[video_name] = cur_output
        labels, scores = getTop_k(labels_index, result)
        cur_output["result"] = [{"labels": labels, "scores": ["%.2f" % scores[i] for i in range(args.top_k)]}]
    
    output_json = os.path.join(args.output, "submit_{}_{}_{}_{}_{}.json".format(args.nclass, args.depth,
                                              args.skip, i, div_size))
    with open(output_json, 'w', encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent = 4)
    
    print("finished predict!!!")


if __name__ == '__main__':
    main()