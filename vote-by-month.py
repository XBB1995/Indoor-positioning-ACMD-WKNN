import csv
import math
from datetime import datetime
from collections import OrderedDict
from decimal import Decimal
# import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import timeit
import time
import os
import logging
from tqdm import trange
from collections import Counter


def timestamp2datetime(timeStamp):
    try:
        d = datetime.fromtimestamp(int(timeStamp))
        str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
        # 2015-08-28 16:43:37.283000'
        return str1
    except Exception as e:
        print(e)
        return ''


def clock(func):
    def clocked(*args):
        t0 = timeit.default_timer()
        result = func(*args)
        elapsed = timeit.default_timer() - t0
        logging.debug('Time_cost:[%0.8fs]' % elapsed)
        return result

    return clocked


def n2s(num):
    if num <= 9:
        return '0' + str(num)
    else:
        return str(num)


# 绘图函数
# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_plot_rp(myList, Title, Xlabel, Ylabel):
    y1 = myList
    x1 = range(0, 96, 1)
    plt.plot(x1, y1, label='Error', linewidth=1, color='r', marker='o', markerfacecolor='blue', markersize=3)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.legend()
    plt.show()


# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_plot_month(myList, Title, Xlabel, Ylabel, No):
    y1 = myList
    x1 = range(1, 16, 1)
    plt.figure(int(No))
    plt.plot(x1, y1, label='Error WKNN', linewidth=1, color='r', marker='o', markerfacecolor='green', markersize=2)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.savefig('WKNN-Error-75 of month No.%s.png' % No)
    logging.info('Fig saved!')
    plt.legend()
    plt.ion()
    plt.pause(1)
    plt.close()


def draw_error_acc(error_file, Title, Xlabel, Ylabel):
    lines = []
    with open(error_file, 'r') as f:
        lines = f.read().split('\n')

    dataSets = []
    for line in lines:
        # print(line)
        try:
            dataSets.append(line.split(','))
        except:
            print("Error: Exception Happened... \nPlease Check Your Data Format... ")

    temp = []
    for set in dataSets:
        temp2 = []
        for item in set:
            if item != '':
                temp2.append(float(item))
        temp2.sort()
        temp.append(temp2)
    dataSets = temp

    for set in dataSets:
        plotDataset = [[], []]
        count = len(set)
        for i in range(count):
            plotDataset[0].append(float(set[i]))
            plotDataset[1].append((i + 1) / count)
        # print(plotDataset)
        plt.plot(plotDataset[0], plotDataset[1], '-', linewidth=2)

    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    # plt.savefig(error_file[:-4] + '.png')
    # logging.info('error CDF saved!')
    plt.show()


# # 按照参考点生成指纹库 每个参考点对应一个指纹 原方法
def rss_crd(tra_filename):
    """
    :param tra_filename: 训练集文件名
    :return: 指纹库
    """
    # get raw data from .csv file

    # get rss

    fp_coor = {}

    with open(tra_filename) as f:
        reader = list(csv.reader(f))
        fp_len = len(reader[0])
        for i in range(len(reader)):
            if i % 6 == 0:
                continue
            if i % 6 == 1:
                fp = fp_len * [0]
            for j in range(fp_len):
                if reader[i][j] == '100':
                    fp[j] = fp[j] - 105
                else:
                    fp[j] = fp[j] + int(reader[i][j])
            if i % 6 == 5:
                for j in range(fp_len):
                    fp[j] = fp[j] // 5
                    # if fp[j] == -100:
                    #    fp[j] = 100
                fp_coor['rp' + str(i // 6)] = fp

    # get crd match fp

    crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
    with open(crd_filename) as crd:
        coor = list(csv.reader(crd))
        crd_len = len(coor)
        for i in range(0, crd_len, 6):
            fp_coor['rp' + str(i // 6)] = fp_coor['rp' + str(i // 6)] + coor[i]

    return fp_coor,


# 按照参考点生成指纹库 每个参考点对应一个指纹 最大值替换平均值
def rss_crd_max(tra_filename):
    """
    :param tra_filename: 训练集文件名
    :return: 指纹库
    """
    # get raw data from .csv file

    # get rss

    fp_coor = {}

    with open(tra_filename) as f:
        reader = list(csv.reader(f))
        r_len = int(len(reader))
        fp_len = len(reader[0])
        rss = np.array(list(map(int, reader[0])))
        for i in range(1, r_len):
            rss = np.vstack((rss, list(map(int, reader[i]))))
        rss[rss == 100] = -105

    max_rss = [0] * r_len
    for i in range(0, r_len, 6):
        _max_rss = np.max(rss[i:i + 6, :], 0)
        if i == 0:
            max_rss = _max_rss
        else:
            max_rss = np.vstack((max_rss, _max_rss))

    for i in range(0, r_len, 6):
        fp_coor['rp' + str(i // 6)] = list(map(str, max_rss[i // 6, :]))

    # get crd match fp

    crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
    with open(crd_filename) as crd:
        coor = list(csv.reader(crd))
        crd_len = len(coor)
        for i in range(0, crd_len, 6):
            fp_coor['rp' + str(i // 6)] = fp_coor['rp' + str(i // 6)] + coor[i]

    return fp_coor, max_rss


# 实现聚类
def k_means(f_rss):
    max_itor = 50

    pass

# 高斯滤波预处理 RSS属于(u-o，u+o) 效果不佳

# 按照参考点生成指纹库 每个参考点对应一个指纹 原方法
# def rss_crd(tra_filename):
#     """
#     :param tra_filename: 训练集文件名
#     :return: 指纹库
#     """
#     # get raw data from .csv file
#
#     # get rss
#
#     fp_coor = {}
#
#     with open(tra_filename) as f:
#         reader = list(csv.reader(f))
#         fp_len = len(reader[0])
#         for i in range(len(reader)):
#             if i % 6 == 0:
#                 fp = fp_len * [0]
#             for j in range(fp_len):
#                 if reader[i][j] == '100':
#                     fp[j] = fp[j] - 105
#                 else:
#                     fp[j] = fp[j] + int(reader[i][j])
#             if i % 6 == 5:
#                 for j in range(fp_len):
#                     fp[j] = fp[j] // 6
#                     # if fp[j] == -100:
#                     #    fp[j] = 100
#                 fp_coor['rp' + str(i // 6)] = fp
#
#     # get crd match fp
#
#     crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
#     with open(crd_filename) as crd:
#         coor = list(csv.reader(crd))
#         crd_len = len(coor)
#         for i in range(0, crd_len, 6):
#             fp_coor['rp' + str(i // 6)] = fp_coor['rp' + str(i // 6)] + coor[i]
#
#     return fp_coor


# 引入RMSE 剔除粗大误差 取平均 效果不佳

# 原方法RMSE
# def rss_crd_rmse(tra_filename):
#     """
#     :param tra_filename: 训练集文件名
#     :return: 指纹库
#     """
#     # get raw data from .csv file
#
#     # get rss
#
#     fp_coor = {}
#
#     with open(tra_filename) as f:
#         reader = list(csv.reader(f))
#         r_len = int(len(reader) / 2)
#         fp_len = len(reader[0])
#         rss = np.array(list(map(int, reader[0])))
#         for i in range(1, r_len):
#             rss = np.vstack((rss, list(map(int, reader[i]))))
#         rss[rss == 100] = -105
#
#     mean_rss = [0] * r_len
#     for i in range(0, r_len, 6):
#         _mean_rss = np.mean(rss[i:i + 6, :], 0)
#         if i == 0:
#             mean_rss = _mean_rss
#         else:
#             mean_rss = np.vstack((mean_rss, _mean_rss))
#
#     # RMSE 均方根误差
#     # v 残差矩阵
#     v = [0] * fp_len
#     for i in range(r_len):
#         _v = (rss[i] - mean_rss[i // 6]) ** 2
#         if i == 0:
#             v = _v
#         else:
#             v = np.vstack((v, _v))
#     # sigma 均方根误差矩阵
#     sigma = [0] * r_len
#     for i in range(0, r_len, 6):
#         _sigma = np.sqrt(0.2 * np.sum(v[i:i + 6, :], 0))
#         if i == 0:
#             sigma = _sigma
#         else:
#             sigma = np.vstack((sigma, _sigma))
#     three_sigma = 3 * sigma
#
#     v = np.sqrt(v)
#
#     fv = [0] * r_len
#     for i in range(r_len):
#         _v = v[i] - three_sigma[i // 6]
#         if i == 0:
#             fv = _v
#         else:
#             fv = np.vstack((fv, _v))
#     # v < 3 * simga 保留 存为1
#     fv[fv >= 0] = 0
#     fv[fv < 0] = 1
#     # 对RSS进行过滤
#     f_rss = fv * rss
#
#     # 求均值 形成最终的RSS指纹
#     final_rss = [0] * r_len
#     count = [0] * r_len
#     for i in range(0, r_len, 6):
#         _f_rss = np.sum(f_rss[i:i + 6, :], 0)
#         _count = np.sum(fv[i:i + 6, :], 0)
#         if i == 0:
#             final_rss = _f_rss
#             count = _count
#         else:
#             final_rss = np.vstack((final_rss, _f_rss))
#             count = np.vstack((count, _count))
#     final_rss = final_rss / count
#
#     where_are_nan = np.isnan(final_rss)
#     final_rss[where_are_nan] = -105
#
#     for i in range(0, r_len, 6):
#         fp_coor['rp' + str(i // 6)] = list(map(str, final_rss[i // 6, :]))
#
#     crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
#     with open(crd_filename) as crd:
#         coor = list(csv.reader(crd))
#         crd_len = int(len(coor) / 2)
#         for i in range(0, crd_len, 6):
#             fp_coor['rp' + str(i // 6)] = fp_coor['rp' + str(i // 6)] + coor[i]
#
#     return fp_coor, final_rss

# 引入RMSE 剔除粗大误差 取平均 by.MAT
def rss_crd_rmse(tra_filename):
    """
    :param tra_filename: 训练集文件名
    :return: 指纹库 字典形式fp_coor + 矩阵形式final_fp
    """
    # get raw data from .csv file

    # get rss

    fp_coor = {}

    with open(tra_filename) as f:
        reader = list(csv.reader(f))
        r_len = int(len(reader) / 2)
        fp_len = len(reader[0])
        rss = np.array(list(map(int, reader[0])))
        for i in range(1, r_len):
            rss = np.vstack((rss, list(map(int, reader[i]))))
        rss[rss == 100] = -105

    # query = rss[0, :]
    # tp_rss = np.tile(query, (r_len, 1))

    # RMSE预处理
    mean_rss = [0] * r_len
    for i in range(0, r_len, 6):
        _mean_rss = np.mean(rss[i:i + 6, :], 0)
        if i == 0:
            mean_rss = _mean_rss
        else:
            mean_rss = np.vstack((mean_rss, _mean_rss))

    # v 残差矩阵
    v = [0] * fp_len
    for i in range(r_len):
        _v = (rss[i] - mean_rss[i // 6]) ** 2
        if i == 0:
            v = _v
        else:
            v = np.vstack((v, _v))
    # sigma 均方根误差矩阵
    sigma = [0] * r_len
    for i in range(0, r_len, 6):
        _sigma = np.sqrt(0.2 * np.sum(v[i:i + 6, :], 0))
        if i == 0:
            sigma = _sigma
        else:
            sigma = np.vstack((sigma, _sigma))
    three_sigma = 3 * sigma

    v = np.sqrt(v)

    fv = [0] * r_len
    for i in range(r_len):
        _v = v[i] - three_sigma[i // 6]
        if i == 0:
            fv = _v
        else:
            fv = np.vstack((fv, _v))
    # v < 3 * simga 保留 存为1
    fv[fv >= 0] = 0
    fv[fv < 0] = 1
    # 对RSS进行过滤
    f_rss = fv * rss

    # 求均值 形成最终的RSS指纹
    final_rss = [0] * r_len
    count = [0] * r_len
    for i in range(0, r_len, 6):
        _f_rss = np.sum(f_rss[i:i + 6, :], 0)
        _count = np.sum(fv[i:i + 6, :], 0)
        if i == 0:
            final_rss = _f_rss
            count = _count
        else:
            final_rss = np.vstack((final_rss, _f_rss))
            count = np.vstack((count, _count))
    final_rss = final_rss / count

    where_are_nan = np.isnan(final_rss)
    final_rss[where_are_nan] = -105

    # # powed RSS
    # min_t = np.min(final_rss)
    # # -105dBm
    # final_rss = final_rss - min_t
    # final_rss = (final_rss / (-min_t)) ** 2.71828

    for i in range(0, r_len, 6):
        fp_coor['rp' + str(i // 6)] = list(map(str, final_rss[i // 6, :]))

    crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
    with open(crd_filename) as crd:
        coor = list(csv.reader(crd))
        crd_len = int(len(coor) / 2)

        # 初始化第一行
        crd = np.array(list(map(float, coor[0])))
        fp_coor['rp' + str(0)] = fp_coor['rp' + str(0)] + coor[0]
        # 循环
        for i in range(6, crd_len, 6):
            crd = np.vstack((crd, list(map(float, coor[i]))))
            fp_coor['rp' + str(i // 6)] = fp_coor['rp' + str(i // 6)] + coor[i]

    # 拼接RSS与CRD
    final_fp = np.hstack((final_rss, crd))

    return fp_coor, final_fp


# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range


# 引入strong AP 概念 RSS + P 效果较好(Fisher准则 效果不佳)
def AP_r_get(tra_filename):
    """
    :param tra_filename: 训练集文件名
    :return: AP可靠性字典
    """
    # get raw data from .csv file

    # get rss

    fp_coor = {}

    with open(tra_filename) as f:
        reader = list(csv.reader(f))
        r_len = int(len(reader) / 2)
        fp_len = len(reader[0])
        rss = np.array(list(map(int, reader[0])))
        for i in range(1, r_len):
            rss = np.vstack((rss, list(map(int, reader[i]))))
        rss[rss == 100] = -105

    # ①信号强度大小 最大值

    mean = rss.copy()
    sum = np.max(mean, 0)

    p_rss = rss.copy()
    p_rss[p_rss != -105] = 1
    p_rss[p_rss == -105] = 0

    p_sum = np.sum(p_rss, 0)

    # AP_rss = {}

    # ①字典
    # for i in range(fp_len):
    #     # AP_rss["AP" + str(i)] = 1
    #     AP_rss["AP" + str(i)] = float(Decimal((sum[i] + 105) / 105).quantize(Decimal('0.000')))

    # ②矩阵
    # ap_rss = (sum + 105) / 105

    # 分级修正宽容度 权重
    ap_rss = -105 / sum - 1
    # ap_rss = normalization(ap_rss)
    # sum_m2t = np.reshape(np.sum(m2t[:, :448], axis=1), (np.shape(m2t)[0], 1))
    # s_m2t = np.tile(sum_m2t, (1, np.shape(m2t)[1]))
    # ap_rss = np.sqrt(m2t)

    # sort 升序排序
    # AP_rss = OrderedDict(sorted(AP_rss.items(), key=lambda d: d[1], reverse=True))

    # ②信号出现频率
    p_sum = p_sum / 288
    p_sum[p_sum <= 0.05] = 0

    # ①字典
    # AP_p = {}
    # for i in range(fp_len):
    #     if p_sum[i] == 0:
    #         AP_p["AP" + str(i)] = 0
    #     elif p_sum[i] == 1:
    #         AP_p["AP" + str(i)] = 100
    #     else:
    #         AP_p["AP" + str(i)] = float(Decimal(1 / (1 - p_sum[i])).quantize(Decimal('0.000')))

    # ②矩阵
    p = p_sum
    p[p == 1] = 101
    p = 1 / (1 - p)
    p[p == 1] = 0
    p[p == -0.01] = 100
    ap_p = p
    # ap_p = normalization(ap_p)
    ap_p = np.log2(ap_p + 1)
    # sort 升序排序
    # AP_p = OrderedDict(sorted(AP_p.items(), key=lambda d: d[1], reverse=True))

    # ③Fisher-AP
    # mean_ap = np.mean(rss, 0)
    # var_rss = [0] * r_len
    # mean_rss = [0] * r_len
    # for i in range(0, r_len, 6):
    #     _var_rss = np.var(rss[i:i + 6, :], 0)
    #     _mean_rss = np.mean(rss[i:i + 6, :], 0)
    #     if i == 0:
    #         var_rss = _var_rss
    #         mean_rss = _mean_rss
    #     else:
    #         var_rss = np.vstack((var_rss, _var_rss))
    #         mean_rss = np.vstack((mean_rss, _mean_rss))
    #
    # # Fisher 准则
    # up_sum = 0
    # down_sum = 0
    # for i in range(int(r_len / 6)):
    #     temp = (mean_rss[i] - mean_ap) ** 2
    #     up_sum = up_sum + (mean_rss[i] - mean_ap) ** 2
    #     down_sum = down_sum + var_rss[i] ** 2
    # fisher_ap = up_sum / down_sum
    #
    # where_are_nan = np.isnan(fisher_ap)
    # fisher_ap[where_are_nan] = 0

    # fisher_ap = np.log2(fisher_ap + 1)

    # AP = {}
    # for i in range(fp_len):
    #     AP["AP" + str(i)] = fisher_ap[i]
        # AP["AP" + str(i)] = 1
    # sort 升序排序
    # AP = OrderedDict(sorted(AP.items(), key=lambda d: d[1], reverse=True))

    # 综合三种指标 形成AP的可靠性参数

    # ①字典
    # AP_r = {}
    # for i in range(fp_len):
    #     AP_r["AP" + str(i)] = AP["AP" + str(i)] * AP_p["AP" + str(i)] * AP_rss["AP" + str(i)]
    # ②矩阵

    ap_mat = ap_rss * ap_p

    AP_r = {}
    for i in range(fp_len):
        AP_r["AP" + str(i)] = ap_mat[i]
    # sort 升序排序
    AP_r = OrderedDict(sorted(AP_r.items(), key=lambda d: d[1], reverse=True))

    return AP_r, ap_mat


# 按照每个测量样本生成指纹库 每个参考点处的每次测量对应的一个指纹
def rss_crd_row(tra_filename):
    """
    :param tra_filename: 训练集文件名
    :return:指纹库
    """
    # get raw data from .csv file
    # fp_coor per row

    fp_coor = {}

    with open(tra_filename) as f:
        reader = list(csv.reader(f))
        fp_len = len(reader[0])
        for i in range(len(reader)):
            fp = [None] * fp_len
            if i % 6 == 0:
                fp_coor['row' + str(i)] = fp
            else:
                for j in range(fp_len):
                    if reader[i][j] == '100':
                        fp[j] = -105
                    else:
                        fp[j] = int(reader[i][j])
                fp_coor['row' + str(i)] = fp

    # get crd match fp

    crd_filename = tra_filename.split('.')[0][:-3] + 'crd.csv'
    with open(crd_filename) as crd:
        coor = list(csv.reader(crd))
        crd_len = len(coor)
        for i in range(crd_len):
            if i % 6 != 0:
                fp_coor['row' + str(i)] = fp_coor['row' + str(i)] + coor[i]
    return fp_coor


# 根据楼层过滤数据
def floor_filter(tra, floor):
    f_c_tra = {}
    for rp, fp in tra.items():
        if fp[-1] == floor:
            f_c_tra[rp] = fp
    return f_c_tra


# 获取训练集对应的r参数字典
def radius_get(f_c_tra):
    """
    :param f_c_tra:指纹库
    :return: radius_dict:对应每个参考点的r参数字典 rp：r
    """
    # radius_dict = {}
    euclid_dis = {}
    r_mat = [0] * np.shape(f_c_tra)[0]
    # AP_list, ap_mat_list = r2list(AP_r)

    # 矩阵方法
    for k, query in enumerate(f_c_tra):
        # 初始化 创建副本矩阵
        temp_dis = np.zeros((np.shape(f_c_tra)))
        out_tra = np.tile(query, (np.shape(temp_dis)[0], 1))
        # ①计算欧氏距离
        temp_dis = (f_c_tra - out_tra) ** 2
        rp_points = np.reshape(np.sqrt(np.sum(temp_dis[:, :-3], axis=1)), (np.shape(temp_dis)[0], 1))
        # ②计算曼哈顿距离
        # temp_dis = np.abs(f_c_tra - out_tra)
        # rp_points = np.reshape(np.sum(temp_dis[:, :-3], axis=1), (np.shape(temp_dis)[0], 1))
        # 拼接得分与坐标
        crd = f_c_tra[:, -2:]
        final_points = np.hstack((rp_points, f_c_tra[:, -2:]))
        # 形成字典（偷懒。。）
        for i, dis in enumerate(final_points):
            euclid_dis["rp" + str(i)] = [dis[0], dis[1], dis[2]]
        # 排序 选出最小值
        euclid_dis = OrderedDict(sorted(euclid_dis.items(), key=lambda d: d[1]))
        min_dis = list(euclid_dis.values())
        for i in range(len(min_dis)):
            if min_dis[i][1] != min_dis[0][1]:
                # radius_dict["rp" + str(k)] = min_dis[i][0]
                r_mat[k] = min_dis[i][0]
                break
    return r_mat


    # 字典方法
    # for out_rp in f_c_tra.keys():
    #     for in_rp, fp in f_c_tra.items():
    #         fp_lens = len(fp) - 3
    #         temp_dis = 0
    #
    #         # 欧氏距离
    #         # for i in AP_list:
    #         #     temp_dis = temp_dis + (float(fp[int(i[2:])]) - float(f_c_tra[out_rp][int(i[2:])])) ** 2
    #         # euclid_dis[in_rp] = [float(Decimal(math.sqrt(temp_dis)).quantize(Decimal('0.000'))),
    #         #                      [f_c_tra[in_rp][-3], f_c_tra[in_rp][-2]]]
    #
    #         # 曼哈顿距离
    #         # for i in AP_list:
    #         #     diff = np.abs(float(fp[int(i[2:])]) - float(f_c_tra[out_rp][int(i[2:])]))
    #         #     temp_dis = temp_dis + diff
    #         # euclid_dis[in_rp] = [float(Decimal(temp_dis).quantize(Decimal('0.000'))),
    #         #                      [f_c_tra[in_rp][-3], f_c_tra[in_rp][-2]]]
    #
    #         # 无AP优选 欧氏距离
    #         for i in range(fp_lens):
    #             temp_dis = temp_dis + (float(fp[i]) - float(f_c_tra[out_rp][i])) ** 2
    #         euclid_dis[in_rp] = [float(Decimal(math.sqrt(temp_dis)).quantize(Decimal('0.000'))),
    #                              [f_c_tra[in_rp][-3], f_c_tra[in_rp][-2]]]
    #
    #         # 无AP优选 曼哈顿距离
    #         # for i in range(fp_lens):
    #         #     diff = np.abs(float(fp[i]) - float(f_c_tra[out_rp][i]))
    #         #     temp_dis = temp_dis + diff
    #         # euclid_dis[in_rp] = [float(Decimal(temp_dis).quantize(Decimal('0.000'))),
    #         #                      [f_c_tra[in_rp][-3], f_c_tra[in_rp][-2]]]
    #
    #     euclid_dis = OrderedDict(sorted(euclid_dis.items(), key=lambda d: d[1]))
    #     min_dis = list(euclid_dis.values())
    #     for i in range(len(min_dis)):
    #         if min_dis[i][1] != min_dis[0][1]:
    #             radius_dict[out_rp] = min_dis[i][0]
    #             break
    # return radius_dict


# 寻找对应dict中前k个rp点较多rp对应的类
def class_get(dis_dict, k, tra_data):
    tag = 1
    class_list = []
    class_dict = {}
    for key in dis_dict.keys():
        if tag > k:
            break
        else:
            tag = tag + 1
            class_list.append(tra_data[key][-3] + "-"+ tra_data[key][-2])
    len_list = len(class_list)
    for i in range(len_list):
        class_dict[i] = class_list.count(class_list[i])
    class_dict = OrderedDict(sorted(class_dict.items(), key=lambda d: d[1], reverse=True))
    for key in class_dict.keys():
        return class_list[int(key)]


def get_k(dis, rp_mat):
    """
    :param dis:
    :param rp_mat:
    :return:
    """
    # 小于1的则全选 大于1排除
    # t = 0
    # for key, value in dis.items():
    #     if value <= 1:
    #         t = t + 1
    #     else:
    #         break
    # if t != 0:
    #     return t

    # 原方法 动态K

    # ②矩阵（待完成）
    # rp_mat[rp_mat > 2 * np.min(rp_mat)] = -1


    # ①字典
    temp = {}
    k_list = []
    mini_dis = 0
    tag = 0
    for k, v in dis.items():
        tag = tag + 1
        if tag == 1:
            mini_dis = v
            temp[k] = v
        else:
            if v <= 2 * mini_dis:
                temp[k] = v

    sum = 0
    out_k = 1
    extra = {}
    tag1 = 0
    mini = 0
    for k, v in temp.items():
        tag1 = tag1 + 1
        if tag1 == 1:
            mini = temp[k]
            k_list.append(int(k[2:]))
            continue
        extra[k] = temp[k] - mini
        sum = sum + extra[k]
    if tag1 == 1:
        return out_k, k_list
    else:
        avg = sum / (tag1 - 1)

    for k in extra.keys():
        if extra[k] <= avg:
            out_k = out_k + 1
            k_list.append(int(k[2:]))
    # out_k = 3 if out_k > 3 else out_k
    return out_k, k_list


# 通过AP_r获取AP_list
def r2list(AP_r):
    """
    :param AP_r:
    :return:
    """
    tag = 0
    ap_list = []
    ap_mat_list = []
    for key in AP_r.keys():
        tag = tag + 1
        if tag > 50:
            break
        else:
            ap_list.append(key)
            ap_mat_list.append(int(key[2:]))
    return ap_list, ap_mat_list


# 波动上限系数w2t
def get_w2t(rss, a):
    u = -105
    # 根据信号强度f_c_tra得到对应波动上限T的系数
    m2t = u / rss - 1
    max_m2t = np.max(m2t)
    m2t = m2t / max_m2t
    # sum_m2t = np.reshape(np.sum(m2t[:, :448], axis=1), (np.shape(m2t)[0], 1))
    # s_m2t = np.tile(sum_m2t, (1, np.shape(m2t)[1]))
    w2t = 1 / (a ** m2t)
    return w2t


# 数据归一化
# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range


# 指纹匹配方法
def tst_rss_crd(f_c_tra, f_c_tst, tst_rp, AP_r, r_mat, ap_mat):
    """
    :param f_c_tra: 训练集指纹
    :param f_c_tst: 测试集指纹
    :param tst_rp: 待测试参考点
    :param radius_dict: r参数字典
    :return: error_dis 误差
    """
    # rps get votes
    rp_vote = {}
    # r_vote = {}
    # 0 -> VOTE
    # 1 -> VOTE_POINT
    statue = 1
    # 阈值默认值
    threshold = 6
    u = -105
    a = 1.05
    # b = 1.02
    # a = 0.08
    # b = 0.92
    # ei = 2.71828
    # dk = 1.01
    # threshold = dk ** threshold
    # 可靠AP列表获取
    AP_list, ap_mat_list = r2list(AP_r)

    # online & offline
    # ap_tra = AP_r
    # ap = ap_tra * ap_tst
    # AP_r = {}
    # for i in range(448):
    #     AP_r["AP" + str(i)] = ap[i]
    # # sort 升序排序
    # AP_r = OrderedDict(sorted(AP_r.items(), key=lambda d: d[1], reverse=True))
    # AP_list, ap_mat_list = r2list(AP_r)

    # 指纹相似度比较 by MAT
    temp_dis = np.zeros((np.shape(f_c_tra)))
    # 初始化存储空间
    query = f_c_tst[int(tst_rp[2:])]
    f_tst = np.tile(query, (np.shape(temp_dis)[0], 1))
    # 根据测试样本扩张

    # 根据信号强度f_c_tra和系数a得到对应波动上限T的系数
    w2t = get_w2t(f_c_tra, a)
    # m2t = u / f_c_tra - 1
    # max_m2t = np.max(m2t)
    # m2t = m2t / max_m2t
    # # sum_m2t = np.reshape(np.sum(m2t[:, :448], axis=1), (np.shape(m2t)[0], 1))
    # # s_m2t = np.tile(sum_m2t, (1, np.shape(m2t)[1]))
    # w2t = 1 / (a ** m2t)
    # re_w2t = (1 - m2t)*0.1+0.9
    # r_w2t = b ** m2t
    # 具体化波动上限
    gt = threshold * w2t

    # 将at写入csv文件
    # output_file = 'E:\\db\\cdf2019\\GT\\fisher-add.csv'
    # with open(output_file, 'a+', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for g in gt:
    #         writer.writerow(g)

    # 将f_c_tra写入csv文件
    # output_file = 'E:\\db\\cdf2019\\GT\\f_c_tra.csv'
    # with open(output_file, 'a+', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for f in f_c_tra:
    #         writer.writerow(f)

    # 匹配
    temp_dis = np.abs(f_c_tra - f_tst)
    # 修正可信度权重
    # diff = temp_dis - gt
    # cr = a * np.abs((np.exp(diff) - 1) / (np.exp(diff) + 1)) + b
    # gt = gt * cr
    temp_dis = temp_dis - gt
    # temp_dis = temp_dis - gt
    temp_dis[temp_dis <= 0] = 0
    # temp_dis = r_w2t * temp_dis
    rp_points = np.sum(temp_dis[:, ap_mat_list], axis=1) / r_mat

    # 归一化距离 (结果：误差增大, 原因：最小值取0会导致误差放大)
    # rp_points = normalization(rp_points)

    for i, dis in enumerate(rp_points):
        rp_vote["rp" + str(i)] = dis

    # 原方法 by 字典
    # for rp, fp in f_c_tra.items():
    #     rp_vote[rp] = 0
    #     fp_lens = len(fp) - 3
    #     # 接收不到的信号不计入
    #
    #     # AP选择
    #     # for i in range(448):
    #     #     AP_r["AP" + str(i)] = AP_r["AP" + str(i)] * AP_r_tst["AP" + str(i)]
    #     #
    #     # AP_r = OrderedDict(sorted(AP_r.items(), key=lambda d: d[1], reverse=True))
    #
    #     for i in AP_list:
    #         if float(fp[int(i[2:])]) == -105 and float(f_c_tst[tst_rp][int(i[2:])]) == -105:
    #             continue
    #         else:
    #             fp_diff = abs((float(fp[int(i[2:])])) - float(f_c_tst[tst_rp][int(i[2:])]))
    #
    #     # for i in range(fp_lens):
    #     #     if float(fp[i]) == -105 and float(f_c_tst[tst_rp][i]) == -105:
    #     #         continue
    #     #     else:
    #     #         fp_diff = abs(float(fp[i]) - float(f_c_tst[tst_rp][i]))
    #
    #         # 阈值的确定!!!
    #
    #         # VOTES
    #         if statue == 0:
    #             if fp_diff >= threshold:
    #                 rp_vote[rp] = rp_vote[rp] + 1
    #         # Vote -> Point 得分量化
    #         # elif statue == 1:
    #         #     if fp_diff <= threshold:
    #         #         rp_vote[rp] = rp_vote[rp] + (threshold - fp_diff) / threshold
    #
    #         # VOTEP 误差累积
    #         elif statue == 1:
    #             if fp_diff >= threshold:
    #                 rp_vote[rp] = rp_vote[rp] + fp_diff

    # rp_vote = OrderedDict(sorted(rp_vote.items(), key=lambda d: d[1]))
    # kv = get_k(rp_vote)

    # sorted
    # for rp, dis in rp_vote.items():
    #     rp_vote[rp] = dis / radius_dict[rp]
    rp_vote = OrderedDict(sorted(rp_vote.items(), key=lambda d: d[1]))
    # for rp in rp_vote.keys():
    #     selected_rp = rp
    #     break
    # print(selected_rp)
    k, k_list = get_k(rp_vote, rp_points)
    # k = 4
    # kr = get_k(r_vote)
    #
    # if kr <= kv:
    #     rp_vote = r_vote
    #     k = kr
    # else:
    #     k = kv

    # k_file = r'E:\db\votep-imp-kfile.csv'
    # with open(k_file, 'a+', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(str(k))

    # print(rp_vote)
    # 形成weight = w + coor
    # Z-score normalize
    # mu = np.mean(rp_points)
    # sigma = np.std(rp_points)
    # rp_points = (rp_points - mu) / sigma
    # w = (1 / np.sqrt(2*3.1415926)) * np.exp(rp_points ** 2 / 2)
    w = 1 / (rp_points + 1e-8)
    w = np.reshape(w, (np.shape(w)[0], 1))
    weight = np.hstack((w, f_c_tra[:, -2:]))[k_list, :]
    weight[:, 0] = weight[:, 0] / np.sum(weight[:, 0], 0)
    # 预测坐标
    [xcoor, ycoor] = [np.dot(weight[:, 0], weight[:, 1]),
                     np.dot(weight[:, 0], weight[:, 2])]
    # 真实坐标
    [realx, realy] = f_c_tst[int(tst_rp[2:]), [-2, -1]]

    # weight = {}
    # for rp, dis in rp_vote.items():
    #     weight[rp] = 1 / (dis + 1e-8)
    #
    # weight = OrderedDict(weight)
    # print(weight)
    # tag = 1
    # xcoor, ycoor, sum_weight = 0, 0, 0
    # k = 7
    # for rp, w in weight.items():
    #     if tag > k:
    #         break
    #     else:
    #         tag = tag + 1
    #         rpx, rpy = float(f_c_tra[int(rp[2:]), -2]), float(f_c_tra[int(rp[2:]), -1])
    #         xcoor = xcoor + rpx * w
    #         ycoor = ycoor + rpy * w
    #         sum_weight = sum_weight + w
    #
    # xcoor = float(Decimal(xcoor / sum_weight).quantize(Decimal('0.00')))
    # ycoor = float(Decimal(ycoor / sum_weight).quantize(Decimal('0.00')))
    # knn
    # not using r 不使用r参数 使用KNN方法估计位置坐标

    # if radius_dict == {}:
    #     tag = 1
    #     xcoor, ycoor = 0, 0
    #
    #     for rp in euclid_dis.keys():
    #         if tag > k:
    #             break
    #         else:
    #             tag = tag + 1
    #             rpx, rpy = float(f_c_tra[rp][-3]), float(f_c_tra[rp][-2])
    #             xcoor = xcoor + rpx
    #             ycoor = ycoor + rpy
    #
    #     xcoor = float(Decimal(xcoor / k).quantize(Decimal('0.000')))
    #     ycoor = float(Decimal(ycoor / k).quantize(Decimal('0.000')))

    # using-r 使用r参数

    # elif radius_dict != {}:
    #     new_dis = {}
    #
    #     for rp, value in euclid_dis.items():
    #         if value <= 1:
    #             new_dis[rp] = value
    #         else:
    #             break
    #
    #     # 如果值均大于1，则寻找前k个rp中出现次数最多的类当做分类结果
    #     if new_dis == {}:
    #         class_1 = class_get(euclid_dis, k, f_c_tra)
    #         xcoor = class_1.split("-")[0]
    #         ycoor = class_1.split("-")[1]
    #     else:
    #         len_new_dis = len(new_dis)
    #         class_2 = class_get(new_dis, len_new_dis, f_c_tra)
    #         xcoor = class_2.split("-")[0]
    #         ycoor = class_2.split("-")[1]
    # 用于定位结果的可视化
    # vis_temp = [tst_rp, realx, realy, xcoor, ycoor]
    # visible_file = r'E:\db\3rdv.csv'
    # with open(visible_file, 'a+', newline='') as vis:
    #     writer = csv.writer(vis)
    #     writer.writerow(vis_temp)

    error_dis = (float(xcoor) - float(realx)) ** 2 + (float(ycoor) - float(realy)) ** 2
    error_dis = float(Decimal(math.sqrt(error_dis)).quantize(Decimal('0.000')))

    # print('%s target coordinates : (%s, %s) , error = %s' % (tst_rp, xcoor, ycoor, error_dis))

    return error_dis


# 实现仿真定位部分
def select_tra_tst(month, test_no, fp_coor_tra, floor, row_state, AP_r, r_mat, ap_mat):
    logging.debug('Reading train_data and test_data.')

    # 生成待读取测试集文件名
    tst_filename = "E:\\db\\" + n2s(month) + "\\tst" + n2s(test_no) + "rss.csv"

    # 设定w_k参数
    # w_k = 3

    # 获取指纹训练样本集，按照r参数分开处理
    if row_state == 1:
        fp_coor_tst = floor_filter(rss_crd_row(tst_filename), floor)
    else:
        fp_coor, fp_mat = rss_crd_rmse(tst_filename)
        # AP_r_tst, ap_tst = AP_r_get(tst_filename)
        # fp_coor_tst = floor_filter(fp_coor, floor)
        fp_coor_tst = np.delete(fp_mat, -1, axis=1)

    error_s = []
    # error_s 数组保存定位误差，用于绘制CDF图
    for rp in range(np.shape(fp_coor_tst)[0]):
        # 按照row参数的取值，分为两种情况进行定位误差估计
        # 上面的rp是指从测试集中选取出的参考点序号
        if row_state == 0:
            e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, 'rp' + str(rp), AP_r, r_mat, ap_mat)
            # e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, 'rp' + str(rp), radius_d)
            error_s = error_s + [e_dis]
        elif rp % 6 != 0 and row_state == 1:
            e_dis = tst_rss_crd(fp_coor_tra, fp_coor_tst, 'row' + str(rp))
            error_s = error_s + [e_dis]

    # 75%的定位误差
    err_75 = np.percentile(np.array(error_s), 75)
    err_75 = float(Decimal(err_75).quantize(Decimal('0.000')))

    # 将定位误差写入csv文件
    output_file = 'E:\\db\\cdf2019\\GT\\6.11.csv'
    with open(output_file, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(error_s)

    # 记录日志
    logging.debug('error_list -> .csv file: OK!')
    logging.debug('max error = %s, min error = %s' % (max(error_s), min(error_s)))

    # 画直方图
    # draw_plot_rp(error_s, 'Error Distance Plot', 'RP', 'error/m')  # 直方图展示
    # draw_error_acc(output_file, 'Error CDF Graph', 'error/m', 'percentage')  # 累计误差分布图

    return err_75


def main():
    # 配置日志设置
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    logging.info('Start test!')
    np.seterr(divide='ignore', invalid='ignore')
    # 测试序列
    # output_merge_file = 'E:\\db\\wknn_error\\wknn-m2-ori.csv'

    # param
    test_m_end = 15
    test_m_start = 1
    # 训练集序号
    test_tra_no = 1
    floor = "3"
    # 是否引入r参数
    r_state = 1
    # 是否使用全部测量样本
    row_state = 0

    # match by month 根据月份来进行实验
    for month in range(test_m_start, test_m_end + 1):

        # read trn data 生成训练集文件名
        filename = "E:\\db\\" + n2s(month) + "\\trn" + n2s(test_tra_no) + "rss.csv"

        # using r_statue row_state
        if r_state == 0 and row_state == 0:
            fp_coor, final_rss = rss_crd_rmse(filename)
            AP_r = AP_r_get(filename)
            tra_data = floor_filter(fp_coor, floor)

            # 方案一 奇数RP用于测试
            # f_odd = {}
            # for k, v in tra_data.items():
            #     if int(k[2:]) % 2 == 0:
            #         f_odd[k] = v
            # tra_data = f_odd

            # # 方案三
            # f_odd = {}
            # for k, v in tra_data.items():
            #     if int(k[2:]) % 6 != 0 and int(k[2:]) % 6 != 1:
            #         f_odd[k] = v
            # tra_data = f_odd

            radius = {}
        elif r_state == 0 and row_state == 1:
            tra_data = floor_filter(rss_crd_row(filename), floor)

            # 选择奇数RP用于测试
            # f_odd = {}
            # for k, v in tra_data.items():
            #     if (int(k[3:]) // 6) % 2 == 0:
            #         f_odd[k] = v
            # tra_data = f_odd

            # 方案三
            # f_odd = {}
            # for k, v in tra_data.items():
            #     new_rp = int(k[3:]) // 6
            #     if new_rp % 6 != 0 and new_rp % 6 != 1:
            #         f_odd[k] = v
            # tra_data = f_odd

            radius = {}
        elif r_state == 1 and row_state == 0:
            fp_coor, fp_mat = rss_crd_rmse(filename)
            AP_r, ap_mat = AP_r_get(filename)
            # tra_data = floor_filter(fp_coor, floor)
            tra_data = np.delete(fp_mat, -1, axis=1)

            # 选择奇数RP用于测试
            # f_odd = {}
            # for k, v in tra_data.items():
            #     if int(k[2:]) % 2 == 0:
            #         f_odd[k] = v
            # tra_data = f_odd

            # 方案三
            # f_odd = {}
            # for k, v in tra_data.items():
            #     if int(k[2:]) % 6 != 0 and int(k[2:]) % 6 != 1:
            #         f_odd[k] = v
            # tra_data = f_odd

            r_mat = radius_get(tra_data)
        else:
            tra_data = floor_filter(rss_crd_row(filename), floor)

            # 选择奇数RP用于测试
            # f_odd = {}
            # for k, v in tra_data.items():
            #     if (int(k[3:]) // 6) % 2 == 0:
            #         f_odd[k] = v
            # tra_data = f_odd

            # 方案三
            # f_odd = {}
            # for k, v in tra_data.items():
            #     new_rp = int(k[3:]) // 6
            #     if new_rp % 6 != 0 and new_rp % 6 != 1:
            #         f_odd[k] = v
            # tra_data = f_odd
            #
            r_mat = radius_get(tra_data)

        # match test set error_list用于绘图
        error_list = []
        for test_no in trange(1, 6):
            error_75 = select_tra_tst(month, test_no, tra_data, floor, row_state, AP_r, r_mat, ap_mat)
            error_list = error_list + [error_75]
        # print(error_list)

    logging.info('Test end!')

    # for test_tst_no in range(1, 6):
    #     error_list = month_error_75_func(test_m_start, test_m_end, test_tra_no, test_tst_no)
    #     # draw_plot_month(error_list, 'Error Distance Plot TEST NO.%s' % test_tst_no, 'Month', 'error/m', test_tst_no)
    #     with open(output_merge_file, 'a+', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         # writer.writerow([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    #         writer.writerow(error_list)


if __name__ == "__main__":
    main()
