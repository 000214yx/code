import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import cv2
import glob
import os
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import math
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow import keras
from tensorflow.keras import layers
#from Generator import Generator
from Generator_new2 import Generator_new2
from Discriminator_new2 import Discriminator_new2
from Generator_new1 import Generator_new1
from Discriminator_new1 import Discriminator_new1
from Generator_new4 import Generator_new4
from skimage.metrics import structural_similarity as compare_ssim
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
assert tf.__version__.startswith('2.')

SAVE_PATH_MODELS = './generate_model'

#真为 1
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def celoss_ones(logits):
    # logits的shape = [b, 1]
    # real shape ：[b] => [1,1,1,1...]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    #return loss
    return tf.reduce_mean(loss)
#fake: 0
def celoss_zeros(logits):
    # [b, 0]
    # [b] => [0,0,0,0...]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


#  WGAN-gp
def gradient_penalty(discriminator,batch_x,fake_image):
    batchsz = batch_x.shape[0]
    # [b,h,w,c]
    t = tf.random.uniform([batchsz,1,1,1],0.0,1.0)
    # [b,1,1,1] =>[b,h,w,c]
    #t = tf.broadcast_to(t,batch_x.shape)
    #D(real)
    interplate = t * batch_x +(1 - t) * fake_image

    # min_term = interplate
    Alpha = (prev_interplate + 0.001) / (prev_interplate + prev_d_interplote_logits + 0.001)
    Alpha_ = tf.reduce_mean(Alpha, axis=(1, 2,3), keepdims=True)
    Beta = (prev_d_interplote_logits + 0.001)/ (prev_interplate + prev_d_interplote_logits + 0.001)
    Beta_ = tf.reduce_mean(Beta, axis=(1, 2, 3), keepdims=True)
    min_term = tf.minimum((0.5 + Alpha) * interplate, (0.5 + Beta) * interplate)
    min_term = interplate

    with tf.GradientTape() as tape:
        tape.watch([min_term])
        #D(X)
        d_interplote_logits = discriminator(interplate)
        max_term = tf.minimum(tf.minimum((0.8 + tf.abs(Alpha_)),1.0) * d_interplote_logits, tf.minimum((0.8 + tf.abs(Beta_)),1.0) * d_interplote_logits)
    grads = tape.gradient(max_term ,min_term)
    # grads:[b,h,w,c]
    grads = tf.reshape(grads,[grads.shape[0],-1])
    gp = tf.norm(grads,axis=1) #[b]
    gp = tf.reduce_mean((gp-1)**2)
    prev_interplate.assign(interplate)
    d_interplote_logits = tf.image.resize(d_interplote_logits,[100,100])
    prev_d_interplote_logits.assign(d_interplote_logits)
    return gp


def d_loss_fn(generator,discriminator,batch_z,batch_x,is_training):

    # 1. real image to be real input
    # 2. generater image to be fake input
    fake_image = generator(batch_z,is_training)
    d_fake_logits = discriminator(fake_image,is_training)
    d_real_logits = discriminator(batch_x,is_training)
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    # Alpha = (prev_fake_image + 1e-3) / (prev_fake_image + fake_image)
    # Beta = (prev_batch_x + 1e-3) / (prev_batch_x + batch_x)

    gp = gradient_penalty(discriminator, batch_x, fake_image)
    loss = d_loss_fake + d_loss_real + 0.5 * gp
    #loss = d_loss_fake + d_loss_real
    # prev_batch_x.assign(batch_x)
    # prev_fake_image.assign(fake_image)
    return loss,gp

def g_loss_fn(generator,discriminator,batch_z,is_training):

    fake_image = generator(batch_z,is_training)
    d_fake_logits = discriminator(fake_image,is_training)
    #虽然是fake_image，但是希望它能骗过判别器，所以输出为真（1）
    loss = celoss_ones(d_fake_logits)

    return loss

#load datasets
def make_anime_dataset(img_paths, batch_size, resize=100, drop_remainder=True, shuffle=True, repeat=1):
    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          drop_remainder=drop_remainder,
                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
    img_shape = (resize, resize, 1)
    len_dataset = len(img_paths) // batch_size

    return dataset, img_shape, len_dataset


def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)

    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset


def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    """Batch dataset of memory data.
    Parameters
    ----------
    memory_data : nested structure of tensors/ndarrays/lists
    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
    dataset = batch_dataset(dataset,
                            batch_size,
                            drop_remainder=drop_remainder,
                            n_prefetch_batch=n_prefetch_batch,
                            filter_fn=filter_fn,
                            map_fn=map_fn,
                            n_map_threads=n_map_threads,
                            filter_after_map=filter_after_map,
                            shuffle=shuffle,
                            shuffle_buffer_size=shuffle_buffer_size,
                            repeat=repeat)
    return dataset


def disk_image_batch_dataset(img_paths,
                             batch_size,
                             labels=None,
                             drop_remainder=True,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):
    """Batch dataset of disk image for PNG and JPEG.
    Parameters
    ----------
        img_paths : 1d-tensor/ndarray/list of str
        labels : nested structure of tensors/ndarrays/lists
    """
    if labels is None:
        memory_data = img_paths
    else:
        memory_data = (img_paths, labels)

    def parse_fn(path, *label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, 1)  # fix channels to 3
        return (img,) + label

    if map_fn:  # fuse `map_fn` and `parse_fn`
        def map_fn_(*args):
            return map_fn(*parse_fn(*args))
    else:
        map_fn_ = parse_fn

    dataset = memory_data_batch_dataset(memory_data,
                                        batch_size,
                                        drop_remainder=drop_remainder,
                                        n_prefetch_batch=n_prefetch_batch,
                                        filter_fn=filter_fn,
                                        map_fn=map_fn_,
                                        n_map_threads=n_map_threads,
                                        filter_after_map=filter_after_map,
                                        shuffle=shuffle,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat)
    return dataset



def train():


    #Save G and D loss
    D_loss = []
    G_loss = []
    #Save PSNR,SSIM of each epoch
    test_PSNR = []
    test_SSIM = []

    #Save the mean results each N times epochs
    EPOCH_loss = []
    EPOCH_PSNR = []
    Avg_D_loss = []
    Avg_G_loss = []

    #Save the max results each K times epochs
    PSNR_max = []
    SSIM_max = []

    tf.random.set_seed(1234)
    np.random.seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # img_path = glob.glob('D:\PycharmProjects/noise_code\donoise_DATA/train1\*.jpg')
    img_path = glob.glob('./train_png_GRAY/*.png')
    noise_path = glob.glob('./train_gauss_salt-pepper_GRAY/*.png')

    dataset,img_shape,_ = make_anime_dataset(img_path,batch_size)
    print('dataset,img_shape :',dataset,img_shape)
    sample = next(iter(dataset))
    print('sample.shape: ',sample.shape)
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    #train datasets
    dataset2,noise_shape,_ = make_anime_dataset(noise_path,batch_size)
    print('dataset,img_shape :',dataset2,noise_shape)
    sample = next(iter(dataset2))
    print('sample.shape: ',sample.shape)
    dataset2 = dataset2.repeat()
    db_iter2 = iter(dataset2)

    generator.build(input_shape = (None,100,100,1))
    discriminator.build(input_shape = (None,100,100,1))

    # optimizer
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate_g,beta_1 = 0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate_d,beta_1 = 0.5)

    #inputs
    for epoch in range(epochs):
        # if epoch % 10 == 0:
        #     lr_rate_g = lr_rate_g * 0.98
        #     lr_rate_d = lr_rate_d * 0.98
        #batch_fake_noise = tf.random.uniform([batch_size,z_dim,z_dim],minval=-1.,maxval=1.)
        #new_batch_fake_noise = tf.reshape(batch_fake_noise,(batch_fake_noise.shape[0],batch_fake_noise.shape[1],batch_fake_noise.shape[2],1))
        # batch_z = tf.convert_to_tensor(batch_z)
        #real img
        batch_real_img = next(db_iter)
        #noisy img
        noise_fake_img = next(db_iter2)

        # train D
        with tf.GradientTape() as tape:
            d_loss,gp = d_loss_fn(generator,discriminator,noise_fake_img,batch_real_img,is_training)
            #d_loss = d_loss_fn(generator, discriminator, noise_fake_img, batch_real_img, is_training)
        #tape.gradient
        grads = tape.gradient(d_loss,discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads,discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator,discriminator,noise_fake_img,is_training)
        grads = tape.gradient(g_loss,generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        # if g_loss >= 5:
        #     g_loss = 3
        # if d_loss >= 3:
        #     d_loss = 1
        G_loss.append(g_loss)
        D_loss.append(d_loss)

        avg_sum_num = 80
        if epoch % avg_sum_num == 0:
            avg_G_data = sum(G_loss[:avg_sum_num]) / avg_sum_num
            Avg_G_loss.append(avg_G_data)
            avg_D_data = sum(D_loss[:avg_sum_num]) / avg_sum_num
            Avg_D_loss.append(avg_D_data)
            epoch_now = epoch
            EPOCH_loss.append(epoch_now)
            G_loss = []
            D_loss = []


        if epoch % 20 == 0:
            PSNR,SSIM = test()
            print('PSNR: ',PSNR,'  SSIM: ',SSIM)

            test_PSNR.append(PSNR)
            test_SSIM.append(SSIM)
            if epoch % 100 == 0:
                PSNR_max.append(max(test_PSNR[:10]))
                SSIM_max.append(max(test_SSIM[:10]))
                epoch_now2 = epoch
                EPOCH_PSNR.append(epoch_now2)
                test_PSNR = []
                test_SSIM = []


            now = datetime.now()
            current_time = now.strftime(" %H:%M:%S")
            print(epoch, 'd_loss:', float(d_loss), ' g_loss:', float(g_loss), current_time)
            # print(epoch, 'd_loss:', float(d_loss), ' g_loss:', float(g_loss),'gp:',float(gp))
            if PSNR > 26.:
                print('#################################################################')
                generator.save(os.path.join(SAVE_PATH_MODELS, f'Generator_{datetime.now().strftime("%Y%m%d-%H_%M_%S")}'))

    plt.style.use("default")
    plt.figure()
    plt.plot(EPOCH_loss, Avg_G_loss, color='red', label="G_loss", marker='p', markevery=10)
    #pictures
    plt.plot(EPOCH_loss, Avg_D_loss, color='darkblue', label="D_loss", marker='^', markevery=10)
    #plt.suptitle("Training G&D Loss")
    plt.xlabel("Epoch ")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig('./plt.save/G loss 20.png')
    plt.show()

    plt.style.use("default")
    plt.figure()
    plt.plot(EPOCH_loss, Avg_D_loss, label="D_loss")
    plt.title("Training D Loss")
    plt.xlabel("Epoch ")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig('./plt.save/D loss 7.png')
    plt.show()

    plt.style.use("default")
    plt.figure()
    plt.plot(EPOCH_PSNR, PSNR_max, label="PSNR")
    plt.title("PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("The value of PSNR(dB)")
    plt.legend(loc="best")
    plt.savefig('./plt.save/PSNR 7.png')
    plt.show()

    #plt.style.use("ggplot")
    plt.style.use("default")
    plt.figure()
    plt.plot(EPOCH_PSNR, SSIM_max, label="SSIM")
    plt.title("SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("The value of SSIM")
    plt.legend(loc="best")
    plt.savefig('./plt.save/SSIM 7.png')
    plt.show()

def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def SSIM(img1,img2):
    # calculate the SSIM between the two images
    (score, diff) = compare_ssim(img1, img2, full=True)
    return score

def test():
    test_num = 20
    test_img_path = glob.glob('./test_gauss_salt-pepper_GRAY 2/*.png')
    test_img_label = glob.glob('./test_real_select_png_GRAY 2/*.png')
    batch_size = 20
    z_dim = 100

    # set test imgs
    test_dataset,test_img_shape, _ = make_anime_dataset(test_img_path,batch_size,shuffle=False)
    sample = next(iter(test_dataset))
    test_dataset = test_dataset.repeat()
    test_db_iter = iter(test_dataset)
    batch_noise_img = next(test_db_iter)

    label_dataset,label_img_shape, _ = make_anime_dataset(test_img_label,batch_size,shuffle=False)
    sample = next(iter(label_dataset))
    label_dataset = label_dataset.repeat()
    label_db_iter = iter(label_dataset)
    batch_label_img = next(label_db_iter)

    #make comparison
    for j in range(batch_label_img.shape[0]):
        img_path = './real_test_compare/' + str(j) + '.png'
        fake_image_j = batch_label_img[j, :, :, :]
        new_fake_image_j = np.uint8((fake_image_j.numpy() + 1.0) * 127.5)
        new_fake_image_j = np.squeeze(new_fake_image_j)
        #new_fake_image_j = cv2.cvtColor(new_fake_image_j, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(img_path,new_fake_image_j)

    psnr_after = 0
    SSIM_after = 0

    test_fake_image = generator(batch_noise_img, training=False)

    #save results in ./images_result/
    for j in range(test_fake_image.shape[0]):
        img_path = './images_result/' + str(j) + '.png'
        fake_image_j = test_fake_image[j,:,:,:]
        new_fake_image_j = np.uint8((fake_image_j.numpy()+1.0)*127.5)
        new_fake_image_j = np.squeeze(new_fake_image_j)
        #new_fake_image_j = cv2.cvtColor(new_fake_image_j, cv2.COLOR_BGR2GRAY)
        #print('new:',new_fake_image_j.shape)
        cv2.imwrite(img_path,new_fake_image_j)

    #Compute PSNR SSIM
    test_origin = []
    for i in range(test_num):
        img1 = cv2.imread("./real_test_compare/{}.png".format(i),
                          cv2.IMREAD_GRAYSCALE)
        test_origin.append( img1)

    test_result = []
    for i in range(test_num):
        img1 = cv2.imread("./images_result/{}.png".format(i),
                          cv2.IMREAD_GRAYSCALE)
        test_result.append(img1)

    test_origin = np.array(test_origin)
    test_result = np.array(test_result)


    for i in range(test_num):
        x = test_origin[i]
        # y = test_noisedata[i]
        z = test_result[i]
        # if PSNR(x, y) != float('inf'):
        #     psnr_before += PSNR(x, y)
        if PSNR(x, z) != float('inf'):
            psnr_after += PSNR(x, z)

        SSIM_after += SSIM(x,z)

    #psnr_before = psnr_before / test_num
    psnr_after = psnr_after / test_num
    SSIM_after = SSIM_after /test_num

    #print("After PSNR:", psnr_after)

    return psnr_after,SSIM_after

if __name__ == '__main__':
    batch_size =5
    lr_rate_g = 0.0003
    lr_rate_d = 0.0001
    epochs = 20000
    is_training = True

    prev_interplate = tf.Variable(initial_value=tf.zeros(shape=(batch_size, 100, 100, 1)))
    prev_d_interplote_logits = tf.Variable(initial_value=tf.zeros(shape=(batch_size, 100, 100, 1)))

    discriminator = Discriminator_new1()
    generator = Generator_new1()
    train()
    test()