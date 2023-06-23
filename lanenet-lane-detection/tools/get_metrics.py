"""
Evaluate lanenet model on tusimple lane dataset
"""
import argparse
import glob
import os
import os.path as ops
import time

import cv2
import numpy as np
import tensorflow as tf
import tqdm
import sys
import matplotlib.pyplot as plt

sys.path.append('/drive/MyDrive/new_projects/lane_detection/lanenet-lane-detection')
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger
from tools.functional import *
sys.path.remove('/drive/MyDrive/new_projects/lane_detection/lanenet-lane-detection')

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_eval')

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', type=str, help='The source tusimple lane test data dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--save_dir', type=str, help='The test output save root dir')

    return parser.parse_args()


def eval_lanenet(src_dir, weights_path, save_dir):
    """

    :param src_dir:
    :param weights_path:
    :param save_dir:
    :return:
    """
    assert ops.exists(src_dir), '{:s} not exist'.format(src_dir)

    os.makedirs(save_dir, exist_ok=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        with open(src_dir,'r') as f:
            image_list = f.readlines()

        # image_list = os.listdir(src_dir+'/gt_image')
        MEAN_IOU = 0
        MEAN_PRECISION = 0
        MEAN_RECALL = 0
        MEAN_F1 = 0

        avg_time_cost = []
        i = 0
        for index, name in tqdm.tqdm(enumerate(image_list)):
            try:
                paths = name.split()
                image_path, mask_path = paths[0], paths[1]
                # image_path = src_dir+'/gt_image/'+name
                # mask_path = src_dir+'/gt_binary_image/'+name

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
                image_vis = image
                image = image / 127.5 - 1.0

                gt_mask_image = cv2.imread(mask_path, 0)/255.0
                gt_mask_image = cv2.resize(gt_mask_image, (512, 256), interpolation=cv2.INTER_LINEAR).astype(float).reshape((256,512,1))
                gt_mask_image_vis = gt_mask_image
                gt_mask_image = np.expand_dims(gt_mask_image,axis=0)
                gt_mask_image = np.round(gt_mask_image,0)

                t_start = time.time()
                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )
                avg_time_cost.append(time.time() - t_start)

                postprocess_result = postprocessor.postprocess(
                    binary_seg_result=binary_seg_image[0],
                    instance_seg_result=instance_seg_image[0],
                    source_image=image_vis
                )

                if index % 100 == 0:
                    LOG.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                    avg_time_cost.clear()

                # input_image_dir = ops.split(image_path.split('clips')[1])[0][1:]
                # input_image_name = ops.split(image_path)[1]
                # output_image_dir = ops.join(save_dir, input_image_dir)
                # os.makedirs(output_image_dir, exist_ok=True)
                output_image_path = ops.join(save_dir, image_path.split('/')[-1])
                # print(image_vis.shape,gt_mask_image.shape,binary_seg_image.reshape((512, 256, 1)).shape)
                # if ops.exists(output_image_path):
                #     continue
                binary_seg_image_vis = binary_seg_image.reshape((256, 512, 1)).astype(float)
                binary_seg_image = np.expand_dims(binary_seg_image_vis,axis=0)
                # print(np.unique(gt_mask_image))
                # print(np.unique(binary_seg_image))

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
                plt.savefig(output_image_path)


                
                # cv2.imwrite(output_image_path, postprocess_result['source_image'])
                i += 1
            except:
                pass

        MEAN_IOU /= i+1
        MEAN_PRECISION /= i+1
        MEAN_RECALL /= i+1
        MEAN_F1 /= i+1

        print(f'IOU={MEAN_IOU:.2f}, Precision={MEAN_PRECISION:.2f}, Recall={MEAN_RECALL:.2f}, f1={MEAN_F1:.2f}')


    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    eval_lanenet(
        src_dir=args.txt_file,
        weights_path=args.weights_path,
        save_dir=args.save_dir
    )