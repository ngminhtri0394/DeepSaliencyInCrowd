import glob
from time import time

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from numpy.random import seed
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
from keras. models import Model
from pyemd import emd_samples

from models.models import *
from models.SelfAttentionModule import *
from utils.configs import *
from utils.metric import *
from utils.ultilities import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras.callbacks import LambdaCallback
import sys


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        if net.startswith("ms"):
            yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c),
                   preprocess_images(images[counter:counter + b_s], int(shape_r/2), int(shape_c/2))]
        elif net.startswith("ts"):
            yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c),
                   preprocess_images(images[counter:counter + b_s], int(shape_r / 2), int(shape_c / 2)),
                   preprocess_images(images[counter:counter + b_s], int(shape_r / 4), int(shape_c / 4))]
        else:
            yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c)]
        counter = (counter + b_s) % len(images)

def load_data():
    images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]

    images.sort()
    maps.sort()
    fixs.sort()

    counter = 0
    X_train = []
    Y_train = []
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps_salicon(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        if net.startswith("ms"):
            X = [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_images(images[counter:counter + b_s], int(shape_r / 2), int(shape_c / 2))]
        elif net.startswith("ts"):
            X = [preprocess_images(images[counter:counter + b_s], shape_r, shape_c),
                 preprocess_images(images[counter:counter + b_s], int(shape_r / 2), int(shape_c / 2)),
                 preprocess_images(images[counter:counter + b_s], int(shape_r / 4), int(shape_c / 4))]
        else:
            X = [preprocess_images(images[counter:counter + b_s], shape_r, shape_c)]
        X_train.append(X)
        Y_train.append([Y, Y, Y_fix])

        counter = (counter + b_s) % len(images)
        if counter == 0:
            break
    y_dummy = np.zeros(shape=(len(X_train), 1))
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=1).split(X_train, y_dummy))
    return folds, X_train, Y_train


def batch_generator(X, Y, batch_size = 1):
    while True:
        for x, y in zip(X, Y):
            yield x, y

def batch_generator_test(X):
    while True:
        for x in X:
            yield x


def create_model():
    if net == 'msdensenet':
        print('Compiling multiscale densenet')
        m = Model(inputs=[x, x1], outputs=msdensenet([x, x1]))
    elif net == 'tsdensenet':
        print('Compiling multiscale(3) densenet')
        m = Model(inputs=[x, x1, x2], outputs=tsdensenet([x, x1, x2]))
    elif net == 'msdensenetnon':
        print('Compiling multiscale densenet without dilated block')
        m = Model(input=[x, x1], output=msdensenet_non([x, x1]))
    elif net == 'sdensenet':
        print('Compiling singlescale densenet')
        m = Model(input=[x], output=sdensenet([x]))
    elif net == 'msdensenet_att':
        print('Compiling multiscale densenet dilated with att')
        m = Model(input=[x, x1], output=msdensenet_att([x, x1]))
    elif net == 'dense':
        print('Compiling dense')
        m = Model(input=[x], output=dense([x]))
    else:
        raise NotImplementedError
    return m



def traning_process(path, model, batch_gen_train, nb_train, batch_gen_val, nb_val, fold, weight=None):
    print(weight)
    if weight is not None:
        import tensorflow as tf
        initepochstr = weight[weight.find(".", 9) + 1:weight.find("-")]
        initepoch = int(initepochstr)
        model.load_weights(path + weight)
        del model
        model = load_model(path + weight, custom_objects={"tf": tf, "kl_divergence": kl_divergence,
                                                  "correlation_coefficient": correlation_coefficient,
                                                  "nss": nss, "SelfAttention": SelfAttention})
        print(initepoch)
        print(path+weight)
    else:
        initepoch = 0


    model.fit_generator(batch_gen_train, nb_train,
                        initial_epoch=initepoch,
                        epochs=nb_epoch,
                        validation_data=batch_gen_val,
                        validation_steps=nb_val,
                        callbacks=[tensorboard,
                                   ModelCheckpoint(path + '/weights.'+net+'f' + str(fold) + '.{epoch:02d}-{val_loss:.4f}.h5',
                                                   save_best_only=True)])


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise NotImplementedError
    else:
        print("Version 1.2")
        K.set_image_data_format("channels_first")
        phase = sys.argv[1]
        x = Input((3, shape_r, shape_c))
        x1 = Input((3, shape_r / 2, shape_c / 2))
        x2 = Input((3, shape_r / 4, shape_c / 4))
        x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

        m = 0
        if phase == 'train':
            path = "weight/cv/" + net
            try:
                weight = sys.argv[2]
            except:
                weight = None

            if not os.path.exists(path):
                os.makedirs(path)
            folds, X_train, Y_train = load_data()
            sum_nss = 0
            sum_cc = 0
            sum_kl = 0
            h = 0
            for j, (train_idx, val_idx) in enumerate(folds):
                m = create_model()
                print("Fold: ", j)


                m.output_names = ['output_1', 'output_2', 'output_3']
                tensorboard = TensorBoard(log_dir="logs/{}_{}_{}".format(net, j, time()))

                m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss], metrics={'output_1': kl_divergence, 'output_2': correlation_coefficient,'output_3': nss})

                X_train_cv = [X_train[i] for i in train_idx]
                y_train_cv = [Y_train[i] for i in train_idx]
                X_valid_cv = [X_train[i] for i in val_idx]
                y_valid_cv = [Y_train[i] for i in val_idx]
                print("Number of train image ", len(X_train_cv))
                print("Number of validation image ", len(X_valid_cv))
                traning_process(path, m, batch_generator(X_train_cv, y_train_cv), len(X_train_cv),
                                batch_generator(X_valid_cv, y_valid_cv), len(X_valid_cv), j, weight=weight)
                weight=None

        elif phase == "test":
            # Output Folder Path
            output_folder = "pred/" + net + '/'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            path_test = sys.argv[3]
            file_names = [f for f in os.listdir(path_test) if f.endswith(('.jpg', '.jpeg', '.png'))]
            file_names.sort()
            nb_imgs_test = len(file_names)
            m = create_model()
            if nb_imgs_test % b_s != 0:
                print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
                exit()


            print("Loading weights")
            weight_path = sys.argv[2]
            m.load_weights(weight_path)

            print("Predicting saliency maps for " + path_test)
            predictions = m.predict_generator(generator_test(b_s=b_s, imgs_test_path=path_test), nb_imgs_test)[0]

            for pred, name in zip(predictions, file_names):
                original_image = cv2.imread(path_test + name, 0)
                res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
                cv2.imwrite(output_folder + '%s' % name, res.astype(int))
        elif phase == 'foldcal':
            folds, X_train, Y_train = load_data()
            path = "weight/cv/" + net + "/result"
            f = open('doc/'+net+'_salicon10f.csv', 'a')
            sum_aucjud = 0
            sum_sim = 0
            sum_emd = 0
            sum_aucbor = 0
            sum_sauc = 0
            sum_nss = 0
            sum_cc = 0
            sum_kl = 0
            m = create_model()
            m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
            smap = cv2.imread("data/shuffle_map.png", 0)
            smap = cv2.resize(smap, (640, 480))

            for j, (train_idx, val_idx) in enumerate(folds):
                print("Fold: ", j)

                X_train_cv = [X_train[i] for i in train_idx]
                y_train_cv = [Y_train[i] for i in train_idx]
                X_valid_cv = [X_train[i] for i in val_idx]
                y_valid_cv = [Y_train[i] for i in val_idx]
                nb_val = len(X_valid_cv)
                lastest_file = glob.glob(path + '/weights.' + net + 'f' + str(j) + '*.*')
                if not lastest_file:
                    print("not found")
                    continue
                lastest_file = max(lastest_file, key=os.path.getctime)
                print(lastest_file)
                m.load_weights(lastest_file)

                predictions = m.predict_generator(batch_generator_test(X_valid_cv), nb_val)[0]
                nss_tmp = 0
                cc_tmp = 0
                kl_tmp = 0
                emd_tmp = 0
                aucjud_tmp = 0
                sim_tmp = 0
                aucbor_tmp = 0
                sauc_tmp = 0
                for pred, gt in zip(predictions, y_valid_cv):
                    res = postprocess_predictions(pred[0], shape_r_out, shape_c_out)
                    res = res/255
                    aucjud_tmp += auc_judd(res, gt[2][0, 0])
                    sim_tmp += similarity(res, gt[0][0, 0])
                    aucbor_tmp += auc_borji(res, gt[2][0, 0])
                    nss_tmp += nss_metric(gt[2][0, 0], res)
                    cc_tmp += cc(gt[0][0, 0], res)
                    kl_tmp += kldiv(gt[0][0, 0], res)
                    emdgt = gt[0][0, 0]*255
                    emdres = res*255
                    emd_tmp += emd_samples(emdgt.flatten(), emdres.flatten(), bins=255)
                    sauc_tmp += auc_shuff(res, gt[2][0, 0], smap)
                print(emd_tmp/nb_val)
                sum_nss += nss_tmp / nb_val
                sum_cc += cc_tmp / nb_val
                sum_kl += kl_tmp / nb_val
                sum_emd += emd_tmp / nb_val
                sum_aucjud += aucjud_tmp / nb_val
                sum_sim += sim_tmp / nb_val
                sum_aucbor += aucbor_tmp / nb_val
                sum_sauc += sauc_tmp / nb_val
                f.write("{},{},{},{},{},{},{},{}\n".format(aucjud_tmp / nb_val, sim_tmp / nb_val, emd_tmp / nb_val,
                                                         aucbor_tmp / nb_val, sauc_tmp / nb_val, cc_tmp / nb_val, nss_tmp / nb_val,
                                                         kl_tmp / nb_val))
            f.write("{},{},{},{},{},{},{},{}\n".format(sum_aucjud/10, sum_sim/10, sum_emd/10,
                                                     sum_aucbor/10, sum_sauc/10, sum_cc/10, sum_nss/10, sum_kl/10))
            f.close()

