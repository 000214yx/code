import tensorflow as tf
import glob
#from Generator import Generator
import numpy as np
import cv2
import multiprocessing

#generator = Generator()

def make_anime_dataset(img_paths, batch_size, resize=100, drop_remainder=True, shuffle=False, repeat=1):
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

def test():
    test_img_path = glob.glob('./test_gauss_salt-pepper_GRAY 2/*.png')
    test_num = 20
    z_dim = 100
    test_dataset,test_img_shape, _ = make_anime_dataset(test_img_path,test_num)
    sample = next(iter(test_dataset))
    test_dataset = test_dataset.repeat()
    test_db_iter = iter(test_dataset)


    batch_noise_img = next(test_db_iter)
    print(batch_noise_img.shape)

    test_fake_image = generator(batch_noise_img, training=False)
    for j in range(test_fake_image.shape[0]):
        img_path = './images_result/''test_' + str(j) + '.png'
        fake_image_j = test_fake_image[j,:,:,:]
        new_fake_image_j = np.uint8((fake_image_j.numpy()+1.0)*127.5)
        new_fake_image_j = np.squeeze(new_fake_image_j)
        cv2.imwrite(img_path,new_fake_image_j)

if __name__ == '__main__':
    generator = tf.saved_model.load('./generate_model/new my way Generator_20230523-00_01_31 26.10dB 0.7763')
    #generator = tf.keras.models.load_model('./generate_model/GR_GAN Generator_20230510-14_43_51 23.62dB 0.6639')
    test()


