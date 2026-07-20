import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections
import matplotlib as mpl
import matplotlib.cm as cm

import sys

print(os.path.abspath(__file__))
sys.path.append(".")

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from data.s3dis import S3DIS, IterableS3DIS
from data.scannet import SCANNET, IterableSCANNET

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
# from util.voxelize import voxelize

from util.helper_ply import read_ply, write_ply

import pdb

random.seed(123)
np.random.seed(123)

# random.seed(456)
# np.random.seed(456)

# random.seed(789)
# np.random.seed(789)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml',
                        help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)

    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.vis_attn:
        writer = SummaryWriter(args.visual_folder)
    else:
        writer = None

    if args.arch == 'slt_seg':
        from model.sltrm_slr import sparse_lowrank_trm_seg as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))

    model = Model(c=args.feat_dim, k=args.classes, lowrank=args.lowrank, sparse=args.sparse, vis_attn=args.vis_attn).cuda()
    logger.info(model)

    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # criterion = nn.CrossEntropyLoss(weight=torch.Tensor(DP.get_class_weights(args.data_name)),
    #                                                     ignore_index=args.ignore_label).cuda()
    # names = [line.rstrip('\n') for line in open(args.names_path)]

    if args.data_name == 's3dis' or args.data_name == 'S3DIS':
        label_to_names = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 5: 'window', 6: 'door',
                          7: 'table', 8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'}
    elif args.data_name == 'scannet' or args.data_name == 'SCANNET':
        label_to_names = {0: 'floor', 1: 'wall', 2: 'cabinet', 3: 'bed', 4: 'chair',
                          5: 'sofa', 6: 'table', 7: 'door', 8: 'window', 9: 'bookshelf',
                          10: 'picture', 11: 'counter', 12: 'desk', 13: 'curtain', 14: 'refrigerator',
                          15: 'bathtub', 16: 'shower curtain', 17: 'toilet', 18: 'sink', 19: 'otherprop'}
    else:
        raise ValueError

    # test(model, criterion, names)
    test(model, label_to_names, writer)


# def data_prepare():
#     if args.data_name == 's3dis':
#         data_list = sorted(os.listdir(args.data_root))
#         data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
#     else:
#         raise Exception('dataset not supported yet'.format(args.data_name))
#     print("Totally {} samples in val set.".format(len(data_list)))
#
#     return data_list


# def data_load(data_name):
#     data_path = os.path.join(args.data_root, data_name + '.npy')
#     data = np.load(data_path)  # xyzrgbl, N*7
#     coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
#
#     idx_data = []       # length equals to the maximum number of points in any voxel
#     if args.voxel_size:
#         coord_min = np.min(coord, 0)
#         coord -= coord_min
#         idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
#         for i in range(count.max()):
#             idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
#             idx_part = idx_sort[idx_select]
#             idx_data.append(idx_part)
#     else:
#         idx_data.append(np.arange(label.shape[0]))
#
#     return coord, feat, label, idx_data


# def input_normalize(coord, feat):
#     coord_min = np.min(coord, 0)
#     coord -= coord_min
#     feat = feat / 255.
#
#     return coord, feat


def test(model, names, writer=None):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    # args.batch_size_test = 10
    model.eval()

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []
    # data_list = data_prepare()

    # for idx, item in enumerate(data_list):
    #     end = time.time()
    #
    #     pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(item, args.epoch))
    #     label_save_path = os.path.join(args.save_folder, '{}_{}_label.npy'.format(item, args.epoch))
    #     if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
    #         logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(data_list), item))
    #         pred, label = np.load(pred_save_path), np.load(label_save_path)
    #     else:
    #         coord, feat, label, idx_data = data_load(item)
    #
    #         pred = torch.zeros((label.size, args.classes)).cuda()
    #         idx_size = len(idx_data)    # length equals to the maximum number of points in any voxel
    #         idx_list, coord_list, feat_list, offset_list = [], [], [], []
    #         for i in range(idx_size):
    #             logger.info(
    #                 '{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
    #             idx_part = idx_data[i]
    #             coord_part, feat_part = coord[idx_part], feat[idx_part]
    #             if args.voxel_max and coord_part.shape[0] > args.voxel_max:
    #                 coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
    #                 while idx_uni.size != idx_part.shape[0]:
    #                     init_idx = np.argmin(coord_p)
    #                     dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
    #                     idx_crop = np.argsort(dist)[:args.voxel_max]
    #                     coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
    #                     dist = dist[idx_crop]
    #                     delta = np.square(1 - dist / np.max(dist))
    #                     coord_p[idx_crop] += delta
    #                     coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
    #                     idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(
    #                         feat_sub), offset_list.append(idx_sub.size)
    #                     idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
    #             else:
    #                 coord_part, feat_part = input_normalize(coord_part, feat_part)
    #                 idx_list.append(idx_part), coord_list.append(coord_part), \
    #                 feat_list.append(feat_part), offset_list.append(idx_part.size)
    #
    #         batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
    #         for i in range(batch_num):
    #             s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
    #             idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], \
    #                                                            feat_list[s_i:e_i], offset_list[s_i:e_i]
    #             idx_part = np.concatenate(idx_part)
    #             coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
    #             feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
    #             offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
    #
    #             with torch.no_grad():
    #                 pred_part = model([coord_part, feat_part, offset_part])  # (n, k)
    #             torch.cuda.empty_cache()
    #             pred[idx_part, :] += pred_part
    #             logger.info(
    #                 'Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i,
    #                                                    len(idx_list), args.voxel_max, idx_part.shape[0]))
    #         # loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
    #         pred = pred.max(1)[1].data.cpu().numpy()
    #
    #     # calculation 1: add per room predictions
    #     intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
    #     intersection_meter.update(intersection)
    #     union_meter.update(union)
    #     target_meter.update(target)
    #
    #     accuracy = sum(intersection) / (sum(target) + 1e-10)
    #     batch_time.update(time.time() - end)
    #     logger.info('Test: [{}/{}]-{} '
    #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
    #                 'Accuracy {accuracy:.4f}.'.format(idx + 1, len(data_list), label.size, batch_time=batch_time,
    #                                                   accuracy=accuracy))
    #     pred_save.append(pred)
    #     label_save.append(label)
    #     np.save(pred_save_path, pred)
    #     np.save(label_save_path, label)

    if args.data_name == 's3dis' or args.data_name == 'S3DIS':
        dataset = S3DIS(path=args.data_root, sub_grid_size=args.voxel_size,
                        weak_label_ratio=args.weak_label_ratio, test_area_idx=args.test_area)

        val_data = IterableS3DIS(dataset, num_points=args.point_max, loop=args.loop, noise_init=args.noise_init,
                                 split='validation')
    elif args.data_name == 'scannet' or args.data_name == 'SCANNET':
        dataset = SCANNET(path=args.data_root, sub_grid_size=args.voxel_size,
                          weak_label_ratio=args.weak_label_ratio)
        val_data = IterableSCANNET(dataset, num_points=args.point_max, loop=args.loop, noise_init=args.noise_init,
                                   split='validation')

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_test, shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True, sampler=None, drop_last=False)

    # store predictions for all points in sub-clouds
    input_labels = val_data.dataset.input_labels['validation']
    input_names = val_data.dataset.input_names['validation']

    num_clouds = len(input_labels)
    test_probs = [np.zeros(shape=[l.shape[0], args.classes], dtype=np.float32) for l in input_labels]
    test_labels = [np.zeros(shape=[l.shape[0]], dtype=np.float32) for l in input_labels]
    test_proj_probs = [np.zeros(shape=[l.shape[0], args.classes], dtype=np.float32) for l in val_data.dataset.val_proj]
    test_proj_labels = [np.zeros(shape=[l.shape[0]], dtype=np.float32) for l in val_data.dataset.val_labels]

    # smoothing parameter for votes
    test_smooth = 0.95  # smoothness for prediction PROBABILITIES, (not categories)

    # pred_save_path, label_save_path = None, None

    # for idx, item in enumerate(data_list):
    for idx, (coord, feat, target, mask, queried_idx, cloud_idx) in enumerate(val_loader):

        # todo: reload prediction
        # # pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(item, args.epoch))
        # # label_save_path = os.path.join(args.save_folder, '{}_{}_label.npy'.format(item, args.epoch))
        # pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(cloud_idx, args.epoch))
        # label_save_path = os.path.join(args.save_folder, '{}_{}_label.npy'.format(cloud_idx, args.epoch))
        #
        # if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
        #     # logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(data_list), item))
        #     logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, num_clouds, cloud_idx))
        #     pred, label = np.load(pred_save_path), np.load(label_save_path)
        #     test_probs[cloud_idx] = pred
        #     test_labels[cloud_idx] = label
        # else:

        coord, feat, target, mask = \
            coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), \
            target.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        offset = torch.arange(1, coord.shape[0] + 1, device=coord.device).int() * args.point_max  # (b,)
        coord = coord.view(-1, coord.shape[-1])  # (n, 3)
        feat = feat.view(-1, feat.shape[-1])  # (n, c)
        target = target.view(-1)  # (n,)
        mask = mask.view(-1)  # (n,)

        # coord_list = torch.split(coord, args.point_max, dim=0)
        # feat_list = torch.split(feat, args.point_max, dim=0)
        # mask_list = torch.split(mask, args.point_max, dim=0)
        # pred_part_list = []
        # with torch.no_grad():
        #     for coord, feat, mask in zip(coord_list, feat_list, mask_list):
        #         offset = torch.Tensor([coord.shape[0]]).to(coord.device).int()
        #         pred_part = model([coord, feat, mask, offset])  # (n, k)
        #         pred_part_list.append(pred_part)
        # pred_part = torch.cat(pred_part_list, dim=0)

        with torch.no_grad():
            pred_part = model([coord, feat, mask, offset])  # (n, k)

        torch.cuda.empty_cache()

        pred_part = torch.split(pred_part, args.point_max, dim=0)
        # pred_part = pred_part.unsqueeze(0)  # (b=1, n, n_classes)
        for j in range(queried_idx.shape[0]):
            c_i = cloud_idx[j].item()
            p_idx = queried_idx[j].cpu().numpy()
            test_probs[c_i][p_idx] = test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * pred_part[j].cpu().numpy()
            test_labels[c_i][p_idx] = val_data.dataset.input_labels['validation'][c_i][p_idx]

            # # raw points projection
            # val_proj = torch.LongTensor(val_data.dataset.val_proj[c_i]).to(pred_part[j].device)
            # pred_all = torch.index_select(pred_part[j].squeeze(0), dim=0, index=val_proj)
            #
            # pred_all = torch.zeros(val_proj.shape[0], args.classes, device=pred_part[j].device)
            # idx = queried_idx[j].unsqueeze(-1).to(pred_all.device).repeat(1, args.classes)
            # pred_all_tmp = torch.scatter(pred_all, dim=0, index=idx, src=pred_part[j])
            # pred_all = torch.index_select(pred_all_tmp, index=val_proj, dim=0)
            #
            # test_proj_probs[c_i] = test_smooth * test_proj_probs[c_i] + (1 - test_smooth) * pred_all[j].cpu().numpy()
            # test_proj_labels[c_i]= val_data.dataset.val_labels[c_i]

        # visualize attention weights
        if args.vis_attn:
            for attr, _ in model._modules.items():
                if 'enc' in attr:
                    attn = model._modules[attr][1].transformer.attn
                    for i, w_info in enumerate(attn):
                        norm = mpl.colors.Normalize(vmin=w_info['attn'].min(), vmax=w_info['attn'].max())
                        cmap = cm.rainbow(norm(w_info['attn'].squeeze(0))).transpose(2, 0, 1)  # * 255
                        name = 'label_' + str(target[w_info['pt_idx']].item()) + '_block_id_' + str(w_info['block_id'])
                        writer.add_image(name, cmap, dataformats='CHW')

    for cloud_idx in range(len(val_data.dataset.val_proj)):
        # raw points projection
        val_proj = torch.LongTensor(val_data.dataset.val_proj[cloud_idx])
        test_proj_probs[cloud_idx] = torch.index_select(torch.Tensor(test_probs[cloud_idx]), dim=0,
                                                        index=val_proj).numpy()
        test_proj_labels[cloud_idx] = val_data.dataset.val_labels[cloud_idx]

    # calculation 1: add per room predictions
    # for idx, (prob, label) in enumerate(zip(test_probs, test_labels)):
    for idx, (prob, label) in enumerate(zip(test_proj_probs, test_proj_labels)):
        pred_save_path = os.path.join(args.save_folder, '{}_pred.npy'.format(input_names[idx]))
        label_save_path = os.path.join(args.save_folder, '{}_label.npy'.format(input_names[idx]))

        # pred = prob.max(1)[1].data.cpu().numpy()
        pred = prob.argmax(axis=1)

        intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        logger.info(
            'Test: [{}/{}]-{} Accuracy {accuracy:.4f}.'.format(idx + 1, num_clouds, label.size, accuracy=accuracy))
        pred_save.append(pred)
        label_save.append(label)

        # save prediction
        np.save(pred_save_path, pred)
        np.save(label_save_path, label)

        # write_ply(os.path.join(args.save_folder, input_names[idx] + '.ply'), [pred, label], ['pred', 'label'])
        #
        # with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        #     pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        #     pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save),
                                                       np.concatenate(label_save), args.classes,
                                                       args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(
            i, iou_class[i], accuracy_class[i], names[i]))

    if writer is not None:
        writer.close()

    # ========================================================================================================

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for idx, (prob, label) in enumerate(zip(test_probs, test_labels)):
        # pred_save_path = os.path.join(args.save_folder, '{}_pred.npy'.format(input_names[idx]))
        # label_save_path = os.path.join(args.save_folder, '{}_label.npy'.format(input_names[idx]))

        # pred = prob.max(1)[1].data.cpu().numpy()
        pred = prob.argmax(axis=1)

        intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        logger.info(
            'Test: [{}/{}]-{} Accuracy {accuracy:.4f}.'.format(idx + 1, num_clouds, label.size, accuracy=accuracy))
        pred_save.append(pred)
        label_save.append(label)

        # # save prediction
        # np.save(pred_save_path, pred)
        # np.save(label_save_path, label)

        # write_ply(os.path.join(args.save_folder, input_names[idx] + '.ply'), [pred, label], ['pred', 'label'])
        #
        # with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        #     pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        #     pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save),
                                                       np.concatenate(label_save), args.classes,
                                                       args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(
            i, iou_class[i], accuracy_class[i], names[i]))

    if writer is not None:
        writer.close()
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    #         coord, feat, label, idx_data = data_load(item)
    #
    #         pred = torch.zeros((label.size, args.classes)).cuda()
    #         idx_size = len(idx_data)    # length equals to the maximum number of points in any voxel
    #         idx_list, coord_list, feat_list, offset_list = [], [], [], []
    #         for i in range(idx_size):
    #             logger.info(
    #                 '{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
    #             idx_part = idx_data[i]
    #             coord_part, feat_part = coord[idx_part], feat[idx_part]
    #             if args.voxel_max and coord_part.shape[0] > args.voxel_max:
    #                 coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
    #                 while idx_uni.size != idx_part.shape[0]:
    #                     init_idx = np.argmin(coord_p)
    #                     dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
    #                     idx_crop = np.argsort(dist)[:args.voxel_max]
    #                     coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
    #                     dist = dist[idx_crop]
    #                     delta = np.square(1 - dist / np.max(dist))
    #                     coord_p[idx_crop] += delta
    #                     coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
    #                     idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(
    #                         feat_sub), offset_list.append(idx_sub.size)
    #                     idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
    #             else:
    #                 coord_part, feat_part = input_normalize(coord_part, feat_part)
    #                 idx_list.append(idx_part), coord_list.append(coord_part), \
    #                 feat_list.append(feat_part), offset_list.append(idx_part.size)
    #
    #         batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
    #         for i in range(batch_num):
    #             s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
    #             idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], \
    #                                                            feat_list[s_i:e_i], offset_list[s_i:e_i]
    #             idx_part = np.concatenate(idx_part)
    #             coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
    #             feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
    #             offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
    #
    #             with torch.no_grad():
    #                 pred_part = model([coord_part, feat_part, offset_part])  # (n, k)
    #             torch.cuda.empty_cache()
    #             pred[idx_part, :] += pred_part
    #             logger.info(
    #                 'Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i,
    #                                                    len(idx_list), args.voxel_max, idx_part.shape[0]))
    #         # loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
    #         pred = pred.max(1)[1].data.cpu().numpy()
    #
    #     # calculation 1: add per room predictions
    #     intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
    #     intersection_meter.update(intersection)
    #     union_meter.update(union)
    #     target_meter.update(target)
    #
    #     accuracy = sum(intersection) / (sum(target) + 1e-10)
    #     batch_time.update(time.time() - end)
    #     logger.info('Test: [{}/{}]-{} '
    #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
    #                 'Accuracy {accuracy:.4f}.'.format(idx + 1, len(data_list), label.size, batch_time=batch_time,
    #                                                   accuracy=accuracy))
    #     pred_save.append(pred)
    #     label_save.append(label)
    #     np.save(pred_save_path, pred)
    #     np.save(label_save_path, label)
    #
    # with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
    #     pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
    #     pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # # calculation 1
    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU1 = np.mean(iou_class)
    # mAcc1 = np.mean(accuracy_class)
    # allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    #
    # # calculation 2
    # intersection, union, target = intersectionAndUnion(np.concatenate(pred_save),
    #                                                    np.concatenate(label_save), args.classes, args.ignore_label)
    # iou_class = intersection / (union + 1e-10)
    # accuracy_class = intersection / (target + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection) / (sum(target) + 1e-10)
    # logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    # logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))
    #
    # for i in range(args.classes):
    #     logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i],
    #                                                                                 names[i]))
    # logger.info('<<<<<<<<<<<<<<<<


if __name__ == '__main__':
    main()
