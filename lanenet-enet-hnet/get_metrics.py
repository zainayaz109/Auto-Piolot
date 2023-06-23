import os
import os.path as ops
import argparse
import time

import tensorflow as tf
import glog as log
import numpy as np
import cv2
import tqdm
import traceback

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from config import global_config
from lanenet_model import hnet_model
from functional import *
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]
color_map = [np.array([255, 0, 0]),
            np.array([0, 255, 0]),
            np.array([0, 0, 255]),
            np.array([125, 125, 0]),
            np.array([0, 125, 125]),
            np.array([125, 0, 125]),
            np.array([50, 100, 50]),
            np.array([100, 50, 100])]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', type=str, help='The image path or the src image save dir')
    parser.add_argument('--lanenet_weights', type=str, help='The lanenet model weights path')
    parser.add_argument('--hnet_weights', type=str, help='The hnet model weights path')
    parser.add_argument('--save_dir', type=str, help='The test output save root dir')


    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def predict_lanenet(gt_image, lanenet_weights):
    """
    :param gt_image:
    :param lanenet_weights:
    :return:
    """
    lanenet_image = gt_image - VGG_MEAN
    # Step1, predict from lanenet
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant(False, tf.bool)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='enet')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    with tf.Session(config=sess_config) as sess:

        saver.restore(sess=sess, save_path=lanenet_weights)
        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: [lanenet_image]})
        t_cost = time.time() - t_start
        log.info('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))

        t_start = time.time()
        mask = np.random.randn(binary_seg_image[0].shape[0], binary_seg_image[0].shape[1]) > 0.5
        bi = binary_seg_image[0] * mask
        mask_image, lane_coordinate, cluster_index, labels = cluster.get_lane_mask(binary_seg_ret=bi,
                                           instance_seg_ret=instance_seg_image[0], gt_image=gt_image)
        t_cost = time.time() - t_start
        log.info('单张图像车道线聚类耗时: {:.5f}s'.format(t_cost))

        print(instance_seg_image.shape)
        for i in range(4):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        # cv2.imwrite('./data/predict_binary.png', binary_seg_image[0] * 255)
        # cv2.imwrite('./data/predict_lanenet.png', mask_image)
        # cv2.imwrite('./data/predict_instance.png', embedding_image)

    sess.close()

    return lane_coordinate, cluster_index, labels, binary_seg_image[0]

def hnet_predict(gt_image, hnet_weights, lanes_pts, image):
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 64, 128, 3], name='input_tensor')
    lane_pts_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='lane_pts')

    net = hnet_model.HNet(is_training=False)
    coef_transform_back, H = net.inference(input_tensor, lane_pts_tensor, name='hnet')

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    with tf.Session(config=sess_config) as sess:
        saver.restore(sess=sess, save_path=hnet_weights)
        t_start = time.time()
        for i in range(len(lanes_pts)):
            pts = np.ones(shape=(len(lanes_pts[i]), 3))
            pts[:, 0:2] = lanes_pts[i]
            pts[:, [0, 1, 2]] = pts[:, [1, 0, 2]]
            coefficient_back, H_matrix = sess.run([coef_transform_back, H], feed_dict={input_tensor: [gt_image], lane_pts_tensor:pts})
            for j in range(coefficient_back.shape[1]):
                color = color_map[i].astype(int)
                color = (int(color[0]),int(color[1]),int(color[2]))
                cv2.circle(image, (int(coefficient_back[0][j]), int(coefficient_back[1][j])), 1, color, 1)
        t_cost = time.time() - t_start
        log.info('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))
        warped = cv2.warpPerspective(image, H_matrix, (512, 256), flags=cv2.INTER_LINEAR)
        # cv2.imwrite('./data/warped.png', warped)

    sess.close()

    return image

def predict(src_dir, lanenet_weights, hnet_weights, save_dir):
    try:
        with open(src_dir,'r') as f:
            image_list = f.readlines()

        MEAN_IOU = 0
        MEAN_PRECISION = 0
        MEAN_RECALL = 0
        MEAN_F1 = 0

        i = 0

        for index, name in tqdm.tqdm(enumerate(image_list)):
            paths = name.split()
            image_path, mask_path = paths[0], paths[1]

            log.info('开始读取图像数据并进行预处理')
            t_start = time.time()
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image_vis = cv2.cvtColor(cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            # cv2.imwrite('./data/origin_image.png', image_vis)

            gt_mask_image = cv2.imread(mask_path, 0)/255.0
            gt_mask_image = cv2.resize(gt_mask_image, (512, 256), interpolation=cv2.INTER_LINEAR).astype(float).reshape((256,512,1))
            gt_mask_image_vis = gt_mask_image
            gt_mask_image = np.expand_dims(gt_mask_image,axis=0)
            gt_mask_image = np.round(gt_mask_image,0)

            image_hnet = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LINEAR)
            log.info('图像读取完毕, 耗时: {:.5f}s'.format(time.time() - t_start))

            # step1: predict from lanenet model
            lane_coordinate, cluster_index, labels, binary_seg_image = predict_lanenet(image_vis, lanenet_weights)
            binary_seg_image_vis = cv2.resize(binary_seg_image, (512, 256), interpolation=cv2.INTER_LINEAR).reshape(256,512,1).astype(float)
            binary_seg_image = np.expand_dims(binary_seg_image_vis, axis=0)


            iou = tf.keras.backend.eval(iou_score(gt_mask_image,binary_seg_image,per_image=True))
            prec = tf.keras.backend.eval(precision(gt_mask_image,binary_seg_image,per_image=True))
            rec = tf.keras.backend.eval(recall(gt_mask_image,binary_seg_image,per_image=True))
            f1 = tf.keras.backend.eval(f_score(gt_mask_image,binary_seg_image,per_image=True))
            
            MEAN_IOU += iou
            MEAN_PRECISION += prec
            MEAN_RECALL += rec
            MEAN_F1 += f1

            plt.figure(figsize=(12,7))
            plt.subplot(1,3,1)
            plt.imshow(image_vis)
            plt.title('src image')
            plt.subplot(1,3,2)
            plt.imshow(gt_mask_image_vis, cmap='gray')
            plt.title('GT mask')
            plt.subplot(1,3,3)
            plt.imshow(binary_seg_image_vis, cmap='gray')
            plt.title('Pr mask')
            plt.suptitle(f'IOU={iou:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, f1={f1:.2f}')
            output_image_path = ops.join(save_dir, image_path.split('/')[-1])
            plt.savefig(output_image_path)

            tf.reset_default_graph()

            # step2: fit from hnet model
            lanes_pts = []
            for i in cluster_index:
                idx = np.where(labels == i)
                coord = lane_coordinate[idx]
                lanes_pts.append(coord)
            mask_image = hnet_predict(image_hnet, hnet_weights, lanes_pts, image_vis)
            i += 1
            # cv2.imwrite('./data/predict_hnet.png', mask_image)
            K.clear_session()
    except:
        print(traceback.print_exc())
        pass
    MEAN_IOU /= i+1
    MEAN_PRECISION /= i+1
    MEAN_RECALL /= i+1
    MEAN_F1 /= i+1

    print(f'IOU={MEAN_IOU:.2f}, Precision={MEAN_PRECISION:.2f}, Recall={MEAN_RECALL:.2f}, f1={MEAN_F1:.2f}')

if __name__ == '__main__':
    # init args
    args = init_args()

    predict(args.txt_file, args.lanenet_weights, args.hnet_weights, args.save_dir)
