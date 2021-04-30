import os

import cv2
import numpy as np

from datasets.data_io import read_pfm


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """

    thresh = np.maximum((gt / pred), (pred / gt))
    delta = 1.1
    a1 = (thresh < delta).mean()
    a2 = (thresh < delta ** 2).mean()
    a3 = (thresh < delta ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__ == '__main__':
    exp_name = 'unreal_hd1'
    pred_dir = os.path.join('/root/data/outputs', exp_name)
    # gt_dir = '/root/data/datasets/dtu/training/Depths'  # for DTU
    gt_dir = '/root/data/datasets/'   # for UnrealCV

    enable_median_scaling = False
    enable_prob_filter = True

    errors = []
    if enable_median_scaling:
        ratios = []

    for scan in os.listdir(pred_dir):
        print(scan)
        pred_depth_dir = os.path.join(pred_dir, scan, 'depth_est')
        # gt_depth_dir = os.path.join(gt_dir, f'{scan}_train')    # for DTU
        gt_depth_dir = os.path.join(gt_dir, scan, 'Depths')   # for UnrealCV
        # mask_dir = os.path.join(gt_dir, scan, 'masks')

        for pred_depth_filename in os.listdir(pred_depth_dir):
            if pred_depth_filename.endswith('_filtered.pfm') ^ enable_prob_filter:
                continue
            pred_depth = read_pfm(os.path.join(pred_depth_dir, pred_depth_filename))[0]
            gt_depth_filename = 'depth_map_' + pred_depth_filename[4:8] + '.pfm'
            gt_depth = read_pfm(os.path.join(gt_depth_dir, gt_depth_filename))[0]
            gt_depth = np.flip(gt_depth, axis=0)
            gt_depth = cv2.resize(gt_depth, pred_depth.shape[::-1], interpolation=cv2.INTER_NEAREST)

            mask = gt_depth > 0
            # mask_filename = pred_depth_filename[:8] + '.pbm'
            # mask = cv2.imread(os.path.join(mask_dir, mask_filename), cv2.IMREAD_UNCHANGED)
            # mask = np.logical_not(cv2.resize(mask, pred_depth.shape[::-1], interpolation=cv2.INTER_NEAREST))
            # mask = np.logical_and(mask, gt_depth > 0)
            if enable_prob_filter:
                mask = np.logical_and(mask, pred_depth > 0)

            # print(pred_depth_filename)
            # import matplotlib.pyplot as plt
            # plt.subplot(2, 2, 1)
            # plt.imshow(gt_depth)
            # plt.subplot(2, 2, 2)
            # plt.imshow(pred_depth)
            # gt_depth[np.logical_not(mask)] = 0
            # pred_depth[np.logical_not(mask)] = 0
            # plt.subplot(2, 2, 3)
            # plt.imshow(gt_depth)
            # plt.subplot(2, 2, 4)
            # plt.imshow(pred_depth)
            # plt.show()

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            if enable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio
            errors.append(compute_errors(gt_depth, pred_depth))

    if enable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
