import math
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from PIL import Image
import prettytable

import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

def read_file(address):
    data = pd.read_excel(address)
    sheet_nrows = data.shape[0]
    sheet_ncols = data.shape[1] - 1
    object_name = list(data.values[:,0])
    face_name = list(data.columns.values)
    face_name.pop(0)
    U = np.zeros((sheet_nrows, sheet_ncols))
    nro = 0
    while nro < sheet_nrows:
        col = 0
        while col < sheet_ncols:
            U[nro, col] = data.values[nro , col + 1]
            col = col + 1
        nro = nro + 1
    object_name = np.array(object_name)
    return object_name, U, sheet_nrows, sheet_ncols, face_name
def conversion(U, sheet_nrows, sheet_ncols):
    a = np.sum(U, axis=0)/sheet_nrows
    b = np.std(U, axis=0)
    nro = 0
    while nro < sheet_nrows:
        col = 0
        while col < sheet_ncols:
            U[nro, col] = (U[nro, col] - a[col])/b[col]
            col = col+1
        nro = nro+1
    U_max = np.max(U, axis=0)
    U_min = np.min(U, axis=0)
    nro = 0
    while nro < sheet_nrows:
        col = 0
        while col < sheet_ncols:
            U[nro, col] = (U[nro, col] - U_min[col])/(U_max[col]-U_min[col])
            col = col+1
        nro = nro+1
    return U
def dot_product(sheetnrows, sheetncols, uu):
    matrix_max = sheetnrows
    rr = np.eye(matrix_max, matrix_max)
    nro1 = 0
    while nro1 < matrix_max:
        nro2 = 0
        while nro2 < matrix_max:
            if nro2 != nro1:
                nco = 0
                while nco < sheetncols:
                    rr[nro1, nro2] = rr[nro1, nro2] + uu[nro1, nco] * uu[nro2, nco]
                    nco = nco+1
            nro2 = nro2 + 1
        nro1 = nro1+1
    mm = rr.max()
    nro1 = 0
    while nro1 < matrix_max:
        nro2 = 0
        while nro2 < matrix_max:
            if nro2 != nro1:
                rr[nro1, nro2] = rr[nro1, nro2] / mm
            nro2 = nro2 + 1
        nro1 = nro1 + 1
    return rr
def distance_weighted_absolute_value(sheetnrows, sheetncols, uu, ak):
    matrix_max = sheetnrows
    rr = np.zeros((matrix_max, matrix_max))
    nro1 = 0
    while nro1 < matrix_max:
        nro2 = 0
        while nro2 < matrix_max:
            nco = 0
            while nco < sheetncols:
                rr[nro1, nro2] = rr[nro1, nro2] + ak[nco]*abs(uu[nro1, nco] - uu[nro2, nco])
                nco = nco+1
            rr[nro1, nro2] = 1 - rr[nro1, nro2]
            nro2 = nro2 + 1
        nro1 = nro1+1
    return rr
def angle_cos(sheetnrows, sheetncols, uu):
    matrix_max = sheetnrows
    rr = np.zeros((matrix_max, matrix_max))
    nro1 = 0
    while nro1 < matrix_max:
        nro2 = 0
        while nro2 < matrix_max:
            nco = 0
            while nco < sheetncols:
                rr[nro1, nro2] = rr[nro1, nro2] + uu[nro1, nco] * uu[nro2, nco]
                nco = nco+1
            no = 0
            mm1 = 0
            mm2 = 0
            while no < sheetncols:
                mm1 = mm1 + uu[nro1, no]*uu[nro1, no]
                mm2 = mm2 + uu[nro2, no]*uu[nro2, no]
                no = no+1
            rr[nro1, nro2] = rr[nro1, nro2]/(pow(mm1, .5)*pow(mm2, .5))
            nro2 = nro2 + 1
        nro1 = nro1+1
    return rr
def correlation_coefficient_way(sheetnrows, sheetncols, uu):
    matrix_max = sheetnrows
    rr = np.zeros((matrix_max, matrix_max))
    nro1 = 0
    while nro1 < matrix_max:
        nro2 = 0
        while nro2 < matrix_max:
            xx_ = 0
            yy_ = 0
            for i in range(sheetncols):
                xx_ = xx_ + uu[nro1, i]
                yy_ = yy_ + uu[nro2, i]
            xx_ = xx_ / sheetncols
            yy_ = yy_ / sheetncols
            nco = 0
            while nco < sheetncols:  # 开始nco（k）循环
                rr[nro1, nro2] = rr[nro1, nro2] + abs(uu[nro1, nco] - xx_) * abs(uu[nro2, nco] - yy_)
                nco = nco+1
            no = 0
            mm1 = 0
            mm2 = 0
            while no < sheetncols:
                mm1 = mm1 + (uu[nro1, no] - xx_)*(uu[nro1, no] - xx_)
                mm2 = mm2 + (uu[nro2, no] - yy_)*(uu[nro2, no] - yy_)
                no = no+1
            if mm1 == 0 or mm2 == 0:
                rr[nro1, nro2] = 'nan'
            else:
                rr[nro1, nro2] = rr[nro1, nro2]/(pow(mm1, .5) * pow(mm2, .5))
            nro2 = nro2 + 1
        nro1 = nro1+1
    return rr
def exponential_similarity_coefficient(sheetnrows, sheetncols, uu):
    matrix_max = sheetnrows
    rr = np.zeros((matrix_max, matrix_max))
    x_ = np.sum(uu, axis=0)/sheetnrows
    x_std = np.std(uu, axis=0)
    nro1 = 0
    while nro1 < matrix_max:
        nro2 = 0
        while nro2 < matrix_max:
            nco = 0
            while nco < sheetncols:
                rr[nro1, nro2] = rr[nro1, nro2] + math.exp(-3*pow(uu[nro1, nco]-uu[nro2, nco], 2)/4/pow(x_std[nco], 4))
                nco = nco+1
            rr[nro1, nro2] = rr[nro1, nro2] / sheetncols
            nro2 = nro2 + 1
        nro1 = nro1+1
    return rr
def max_min_way(sheetnrows, sheetncols, uu):
    matrix_max = sheetnrows
    rr = np.zeros((matrix_max, matrix_max))
    nro1 = 0
    while nro1 < matrix_max:
        nro2 = 0
        while nro2 < matrix_max:
            nco = 0
            max00 = 0
            min00 = 0
            while nco < sheetncols:
                min00 = min00 + min(uu[nro1, nco], uu[nro2, nco])
                max00 = max00 + max(uu[nro1, nco], uu[nro2, nco])
                nco = nco+1
            rr[nro1, nro2] = min00 / max00
            nro2 = nro2 + 1
        nro1 = nro1+1
    return rr
def Arithmetic_mean_minimum_way(sheetnrows, sheetncols, uu):
    matrix_max = sheetnrows
    rr = np.zeros((matrix_max, matrix_max))
    nro1 = 0
    while nro1 < matrix_max:
        nro2 = 0
        while nro2 < matrix_max:
            nco = 0
            max00 = 0
            min00 = 0
            while nco < sheetncols:
                min00 = min00 + min(uu[nro1, nco], uu[nro2, nco])
                max00 = max00 + uu[nro1, nco]+uu[nro2, nco]
                nco = nco+1
            rr[nro1, nro2] = 2*min00 / max00
            nro2 = nro2 + 1
        nro1 = nro1+1
    return rr
def geometric_mean_minimum_way(sheetnrows, sheetncols, uu):
    matrix_max = sheetnrows
    rr = np.zeros((matrix_max, matrix_max))
    nro1 = 0
    while nro1 < matrix_max:
        nro2 = 0
        while nro2 < matrix_max:
            nco = 0
            max00 = 0
            min00 = 0
            while nco < sheetncols:
                min00 = min00 + min(uu[nro1, nco], uu[nro2, nco])
                max00 = max00 + math.sqrt(uu[nro1, nco]*uu[nro2, nco])
                nco = nco+1
            if max00 != 0:
                rr[nro1, nro2] = min00 / max00
            nro2 = nro2 + 1
        nro1 = nro1+1
    return rr
def test(R):
    label = np.isnan(R)
    if True in label:
        return 'False'
def equal(matrix1, matrix2):
    flag = 0
    for i in range(len(matrix1)):
        k = matrix1[i]
        kt = matrix2[i]
        for j in range(len(k)):
            if k[j] != kt[j]:
                flag = 1
    if flag == 0:
        return 'True'
    else:
        return 'False'
def multi(matrix1, matrix2):
    matrix00 = []
    for i in range(len(matrix1)):
        matrix01 = []
        for j in range(len(matrix2[0])):
            r = []
            for k in range(len(matrix1[0])):
                r.append(min(matrix1[i, k], matrix2[k, j]))
            g = max(r)
            matrix01.append(g)
        matrix00.append(matrix01)
    matrix00 = np.array(matrix00)
    return matrix00
def transitive_closure(R):
    K = R
    KT = multi(R, R)
    while equal(K, KT) != 'True':
        K = KT
        KT = multi(K, K)
    for i in range(0, len(K)):
        for j in range(0, len(K[i])):
            K[i][j] = round(K[i][j], 8)
    return K

def get_class(temp_location):
    lists = []
    for it1 in temp_location:
        temp_list = []
        for it2 in temp_location:
            if it1[0] == it2[1]:
                temp_list.append(it2[0])
        if temp_list not in lists:
            lists.append(sorted(list(set(temp_list))))
    return lists
def class_all(K, lambdas):
    classification_result = []
    for lams in lambdas:
        if lams == lambdas[0]:
            classification_result.append([[x] for x in range(len(K))])
        elif lams == lambdas[-1]:
            classification_result.append([x for x in range(len(K))])
        else:
            location = np.argwhere(K >= lams)
            classification_result.append(get_class(location))
    classification_result[-1] = [classification_result[-1]]
    return classification_result
def list_nozeros_number(matrix):
    number = 0
    for i in range(len(matrix)):
        if matrix[i] != 0:
            number = number + 1
    return number
def judge_same(classes_matrix, contrast_list, length, y_length):
    flag00 = []
    max_list = []
    a = length
    flag_b = 0
    for i in range(y_length, len(contrast_list)):
        if len(list(set(classes_matrix) & set(contrast_list[i]))) != 0:
            if list(set(classes_matrix)) == list(set(contrast_list[i])):
                flag_b = length + 1
                b1 = contrast_list.index(sorted(list(set(classes_matrix))))
                c1 = len(list(set(classes_matrix)))
            else:
                flag00.append(list(set(classes_matrix) - set(contrast_list[i])))
    for j in range(len(flag00)):

        if 0 < len(flag00[j]) < a:
            a = len(flag00[j])
            max_list = flag00[j]
    if a != length and flag_b == 0:
        b0 = contrast_list.index(sorted(list(set(classes_matrix) - set(max_list))))
        c = len(list(set(classes_matrix)-set(max_list)))
    if a == length and flag_b == 0:
        return 'False', 0, 0
    if flag_b == length + 1:
        return 'same', b1, c1
    else:
        return sorted(max_list), b0, c
def change_matrix(K, lambdas, classification_result,):
    Z = []
    Y = [[x] for x in range(len(K))]
    for i in range(1, len(lambdas)):
        for j in range(len(classification_result[i])):
            if classification_result[i][j] not in Y:
                if len(classification_result[i][j]) == 2:
                    mid = [classification_result[i][j][0], classification_result[i][j][1], 1 - lambdas[i], 2]
                    Z.append(mid)
                    Y.append(classification_result[i][j])
                if len(classification_result[i][j]) != 2:
                    location00_T = [0] * len(classification_result[i][j])
                    number_T = [0] * len(classification_result[i][j])
                    complementary_set, location00_T[0], number_T[0] = judge_same(classification_result[i][j], Y, len(K),len(K))

                    flag_new = 0
                    element_new = complementary_set
                    while classification_result[i][j] not in Y:
                        if complementary_set == 'False' and flag_new == 0:
                            for k in range(1, len(classification_result[i][j])):
                                if k == 1:
                                    mid = [classification_result[i][j][0], classification_result[i][j][1], 1 - lambdas[i], 2]
                                    Z.append(mid)
                                    Y.append(classification_result[i][j][0:2])
                                else:
                                    mid = [classification_result[i][j][k], Y.index(classification_result[i][j][0:k]), 1 - lambdas[i],
                                           len(classification_result[i][j][0:k]) + 1]
                                    Z.append(mid)
                                    Y.append(classification_result[i][j][0:k+1])
                            break
                        if complementary_set == 'False' and flag_new != 0:
                            flag_p = 0
                            location00_etc = location00_T[flag_p]
                            number_etc = number_T[flag_p]
                            if list_nozeros_number(location00_T) != 1:
                                while location00_T[flag_p + 1] != 0:
                                    flag_p = flag_p + 1
                                    number_etc = number_etc + number_T[flag_p]
                                    if flag_p == 1:
                                        mid = [location00_etc, location00_T[flag_p], 1 - lambdas[i], number_etc]
                                    else:
                                        mid = [location00_T[flag_p], location00_etc, 1 - lambdas[i], number_etc]
                                    Z.append(mid)
                                    Y.append(sorted(list(set(Y[location00_etc]) | set(Y[location00_T[flag_p]]))))
                                    location00_etc = len(Y) - 1
                            if list_nozeros_number(location00_T) == 1:
                                for new in range(len(element_new)):
                                    mid = [element_new[new], location00_etc, 1 - lambdas[i], len(Y[location00_etc]) + 1]
                                    Z.append(mid)
                                    Y.append(sorted(list(set([element_new[new]]) | set(Y[location00_etc]))))
                                    location00_etc = len(Y) - 1
                            else:
                                for new in range(len(element_new)):
                                    mid = [element_new[new], location00_etc, 1 - lambdas[i], len(Y[-1]) + 1]
                                    Z.append(mid)
                                    Y.append(sorted(list(set([element_new[new]]) | set(Y[-1]))))
                                    location00_etc = len(Y) - 1
                            break
                        if complementary_set == 'same':
                            flag_p = 0
                            location00_etc = location00_T[flag_p]
                            number_etc = number_T[flag_p]
                            while location00_T[flag_p + 1] != 0:
                                flag_p = flag_p + 1
                                number_etc = number_etc + number_T[flag_p]
                                if flag_p == 1:
                                    mid = [location00_etc, location00_T[flag_p], 1 - lambdas[i], number_etc]
                                else:
                                    mid = [location00_T[flag_p], location00_etc, 1 - lambdas[i], number_etc]
                                Z.append(mid)
                                Y.append(sorted(list(set(Y[location00_etc]) | set(Y[location00_T[flag_p]]))))
                                location00_etc = len(Y) - 1
                            break
                        else:
                            flag_new = flag_new + 1
                            element_new = complementary_set
                            complementary_set, location00_T[flag_new], number_T[flag_new] = judge_same(
                                complementary_set, Y, len(K), len(K))
    Z = np.array(Z)
    return Z
def best_lambdas_F(original_U, classification_result, lambdas):
    original_lambadas = np.zeros([len(lambdas), 2])
    for i in range(1 , len(lambdas)-1):
        original_lambadas[i, 0] = lambdas[i]
        original_lambadas[i, 1] = len(classification_result[i])
    ave_1 = np.sum(original_U, axis=0) / original_U.shape[0]
    best_lam = lambdas[0]
    flag_best = 0
    for i in range(len(lambdas)):
        distance_x_above = 0
        distance_x_below = 0
        for j in range(int(original_lambadas[i, 1])):
            distance_x_x = 0
            distance_x_j = 0
            for k in range(original_U.shape[1]):
                sum_glass = 0
                for nu in range(len(classification_result[i][j])):
                    sum_glass = sum_glass + original_U[classification_result[i][j][nu], k]
                ave_2 = sum_glass / len(classification_result[i][j])
                for nu in range(len(classification_result[i][j])):
                    distance_x_j = distance_x_j +pow(original_U[classification_result[i][j][nu], k]-ave_2 , 2)
                distance_x_below = distance_x_below + distance_x_j
                distance_x_x = distance_x_x + pow((ave_2-ave_1[k]) , 2)
            distance_x_above = distance_x_above + len(classification_result[i][j])*distance_x_x
        distance_x_above = distance_x_above/(original_lambadas[i, 1]-1)
        distance_x_below = distance_x_below/(original_U.shape[0]-original_lambadas[i, 1])
        if distance_x_below != 0:
            F_statistics = distance_x_above/distance_x_below
            if F_statistics > flag_best:
                flag_best = F_statistics
                best_lam = lambdas[i]
    return best_lam
def alter_x(lambdas):
    index_x = [str(lambdas[0])]
    section = round((lambdas[0] - lambdas[-1])/8, 5)
    flagg = lambdas[0]
    rangee = [0]
    flagg_r = 0
    for i in range(8):
        flagg = round(flagg -section, 4)
        index_x.append(str(flagg))
        flagg_r = round(flagg_r + section, 4)
        rangee.append(flagg_r)
    return index_x, section, rangee
def draw_picture(Z, object_name, lambdas, best_lambda):
    Z = np.array(Z)
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['lines.color'] = 'black'
    fig = plt.figure(figsize=(8.61, 5.41), dpi=100)
    plt.title("Fuzzy cluster analysis visualization", fontsize=18)
    plt.xlabel("λ", fontsize=16)
    dendrogram(Z, orientation="right", labels=np.array(object_name), leaf_font_size = 5, color_threshold=0, count_sort='descending', distance_sort='ascending',above_threshold_color='black')
    index_x, section, rangee = alter_x(lambdas)
    plt.xticks(rangee, index_x, fontsize=9)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.axvline(x=1 - best_lambda, ls='--')
    plt.text(1 - best_lambda, 5, 'λ = ' + str(round(best_lambda,4)), verticalalignment='bottom', horizontalalignment='left',
             fontsize=12)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format='jpg', dpi = 400)
    buffer_.seek(0)
    dataPIL = Image.open(buffer_)
    dataPIL = np.asarray(dataPIL)
    buffer_.close()
    return dataPIL
def class_lambda_subjective(class_number, classification_result, lambdas):
    best_lambdas_Sub = 0
    for i in range(len(classification_result)):
        if len(classification_result[i]) == class_number:
            best_lambdas_Sub = lambdas[i]
    return best_lambdas_Sub
def all_class_show(classification_result, object_name, lambdas):
    tb1 = prettytable.PrettyTable()
    tbname = ['阈值']
    for i in range(1, len(object_name) + 1):
        a = ' 类别' + str(i)
        tbname.append(a)
    tb1.field_names = tbname
    for i in range(len(classification_result)):
        list_t = ["{:.8f}".format(lambdas[i])]
        for j in range(len(classification_result[i])):
            list_mid = ''
            for k in range(len(classification_result[i][j])):
                list_mid = list_mid + object_name[classification_result[i][j][k]] + ';'
            list_t.append(list_mid)
        while len(list_t) != len(tbname):
            list_t.append('None')
        tb1.add_row(list_t)
    return tb1
def best_class_show(classification_result, object_name, lambdas, best_lambdas):
    tb2 = prettytable.PrettyTable()
    tbname = ['最佳分类结果']
    tbname.append('对象')
    tb2.field_names =tbname
    search_falg = 0
    while best_lambdas != lambdas[search_falg]:
        search_falg+=1
    class_good = classification_result[search_falg]
    for i in range(len(class_good)):
        list_content = ['类别' + str(i)]
        list__ = ''
        for j in range(len(class_good[i])):
            list__ = list__  + object_name[class_good[i][j]] + ';'
        list_content.append(list__)
        tb2.add_row(list_content)
        tb2.align = 'c'
    return tb2
def plot_fuzzy_equiv_heatmap_with_dendrogram(
        K: np.ndarray,
        object_name: list,
        figsize: tuple = (10, 8),
        dendrogram_ratio: float = 0.1,
) -> Figure:
    if K.shape[0] != K.shape[1]:
        raise ValueError("模糊等价矩阵K必须是方阵")

    if K.shape[0] != len(object_name):
        raise ValueError(f"矩阵维度({K.shape[0]})与标签数量({len(object_name)})不匹配")
    colors = ["#F5DB91", "#E8B33C", "#DB8A2A", "#CE5E19", "#BF3308", "#A12B10", "#9E2512"]
    custom_cmap = ListedColormap(colors)
    df = pd.DataFrame(K, index=object_name, columns=object_name)
    distance_matrix = 1 - df.values
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='single')
    try:
        g = sns.clustermap(
            df,
            row_linkage=linkage_matrix,
            col_linkage=linkage_matrix,
            cmap=custom_cmap,
            figsize=figsize,
            dendrogram_ratio=dendrogram_ratio,
            linewidths=0,
            linecolor='none',
            metric='precomputed',
            method='single'
        )
        g.cax.set_visible(False)
        ax = g.ax_heatmap
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        cbar = g.ax_heatmap.collections[0].colorbar
        cbar_ax = cbar.ax
        cbar_ax.set_position([0.9, 0.15, 0.03, 0.3])  # zuo,xia,kuan,gaodu更精确调整位置
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha='center', fontsize=8, fontfamily='Times New Roman')
        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8, fontfamily='Times New Roman')
        return g.fig

    except Exception as e:
        print(f"使用seaborn clustermap失败: {e}")
        return None


def plot_colorbar_only(
        K: np.ndarray,
        cmap_colors: list = None,
        figsize: tuple = (2, 8),
) -> Figure:
    if cmap_colors is None:
        cmap_colors = ["#F5DB91", "#E8B33C", "#DB8A2A", "#CE5E19", "#BF3308", "#A12B10", "#9E2512"]
    custom_cmap = ListedColormap(cmap_colors)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    norm = plt.Normalize(vmin=K.min(), vmax=K.max())
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_ticks(np.linspace(K.min(), K.max(), 5))
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig




