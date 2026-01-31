import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import Qt_childwindows
import Qt_childwindows2
import Qt_mainwindows
from function_lib import *
import prettytable
from PIL import Image
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class run_main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Qt_mainwindows.Ui_mainWindow()
        self.ui.setupUi(self)
        self.init_ui()
        self.dataPIL = ""
        self.file_data = ""
        self.K = None
        self.object_name = []
    def click_3esc(self):
        app.quit()
    def click_load(self):
        self.ui.textBrowser_input.setLineWrapMode(QtWidgets.QTextBrowser.NoWrap)
        self.ui.textBrowser_class.setLineWrapMode(QtWidgets.QTextBrowser.NoWrap)
        filename = QtWidgets.QFileDialog.getOpenFileNames(self, '打开文件', os.getcwd(), "Xlsx Files(*xlsx);;Xls Files(*xls)")
        if filename != ([], ""):
            adress=str(filename[0][0])
            self.file_data = read_file(adress)
            object_name, original_U, sheet_nrows, sheet_ncols, face_name = self.file_data
            flag_label = 'True'
            self.ui.textBrowser_input.append('对象：')
            self.ui.textBrowser_input.append(str(object_name))
            self.ui.textBrowser_input.append('性状数据:')
            tb0 = prettytable.PrettyTable()
            tb0.field_names = face_name
            for i in range(len(original_U)):
                tb0.add_row(original_U[i])
            self.ui.textBrowser_input.append(str(tb0))
    def click_start(self):
        if self.file_data != "":
            object_name, original_U, sheet_nrows, sheet_ncols, face_name = self.file_data
            U = conversion(original_U, sheet_nrows, sheet_ncols)
            way_falg, R = self.way_select(U, sheet_nrows, sheet_ncols)
            if way_falg == 1:
                K = transitive_closure(R)
                self.K = K
                self.object_name = object_name
                lambdas = np.sort(np.unique(K).reshape(-1))[::-1]
                classification_result = class_all(K, lambdas)
                best_lambda = self.lambda_select(original_U, classification_result, lambdas)
                if best_lambda != 0:
                    self.tb1 = all_class_show(classification_result, object_name, lambdas)
                    self.ui.textBrowser_class.append('分类情况')
                    self.ui.textBrowser_class.append(str(self.tb1))
                    self.tb2 = best_class_show(classification_result, object_name, lambdas, best_lambda)
                    self.ui.textBrowser_input_2.append('最终分类结果')
                    self.ui.textBrowser_input_2.append(str(self.tb2))
                    self.setAcceptDrops(True)
                    Z = change_matrix(K, lambdas, classification_result)
                    self.dataPIL = draw_picture(Z, object_name, lambdas, best_lambda)
                    image_jpg = Image.fromarray(self.dataPIL)
                    image_jpg = image_jpg.convert("RGBA")
                    data00 = image_jpg.tobytes("raw", "RGBA")
                    img = QtGui.QImage(data00, image_jpg.size[0], image_jpg.size[1], QtGui.QImage.Format_RGBA8888)
                    pix = QtGui.QPixmap.fromImage(img)
                    self.ui.label_truegraph.setPixmap(pix)
                    self.ui.label_truegraph.setScaledContents(True)
    def heat(self):
        if self.K is None or len(self.object_name) == 0:
            self.open_child(3)
            return
        print(f"K的形状: {self.K.shape}, object_name长度: {len(self.object_name)}")  # 调试信息
        self.open_child2(self.K, self.object_name)
    def way_select(self, U, sheet_nrows, sheet_ncols):
        falg_way =0
        R = 0
        if self.ui.comboBox_way.currentIndex() == 0:
            R = angle_cos(sheet_nrows, sheet_ncols, U)
            falg_way = 1
        elif self.ui.comboBox_way.currentIndex() == 1:

            R = dot_product(sheet_nrows, sheet_ncols, U)
            falg_way = 1
        elif self.ui.comboBox_way.currentIndex() == 2:
            if self.ui.lineEdit.text() == "":
                ak = [0]
            else:
                rx = QtCore.QRegExp(r"(0|[1-9]\d*)(,0|,[1-9]\d*)*")
                self.ui.lineEdit.setValidator(QtGui.QRegExpValidator(rx))
                ak = list(np.asarray(self.ui.lineEdit.text().split('，'), dtype=np.float64, order='C'))
            if len(ak) == sheet_ncols:
                R = distance_weighted_absolute_value(sheet_nrows, sheet_ncols, U, ak)
                falg_way = 1
            else:
                R = 0
                self.open_child(1)
        elif self.ui.comboBox_way.currentIndex() == 3:
            R = correlation_coefficient_way(sheet_nrows, sheet_ncols, U)
            if test(R) == 'False':
                self.open_child(0)
            else:
                falg_way = 1
        elif self.ui.comboBox_way.currentIndex() == 4:
            R = exponential_similarity_coefficient(sheet_nrows, sheet_ncols, U)
            falg_way = 1
        elif self.ui.comboBox_way.currentIndex() == 5:
            R = max_min_way(sheet_nrows, sheet_ncols, U)
            falg_way = 1
        elif self.ui.comboBox_way.currentIndex() == 6:
            R = Arithmetic_mean_minimum_way(sheet_nrows, sheet_ncols, U)
            falg_way = 1
        elif self.ui.comboBox_way.currentIndex() == 7:
            R = geometric_mean_minimum_way(sheet_nrows, sheet_ncols, U)
            falg_way = 1
        return falg_way,R
    def lambda_select(self, original_U, classification_result, lambdas):
        best_lambda = 0
        if self.ui.comboBox_way_2.currentIndex() == 1:
            best_lambda = best_lambdas_F(original_U, classification_result, lambdas)
        if self.ui.comboBox_way_2.currentIndex() ==0:
            if self.ui.lineEdit_2.text() == "" or int(self.ui.lineEdit_2.text()) >= len(original_U):
                self.open_child(2)
                best_lambda = 0
            else:
                best_lambda = class_lambda_subjective(int(self.ui.lineEdit_2.text()), classification_result, lambdas)
        return best_lambda
    def click_esc(self):
        self.ui.textBrowser_input.clear()
        self.ui.textBrowser_input_2.clear()
        self.ui.textBrowser_class.clear()
        self.ui.textBrowser_graph.clear()
        self.ui.lineEdit.clear()
        self.ui.label_truegraph.clear()
    def click_clear_show(self):
        self.ui.textBrowser_input_2.clear()
        self.ui.textBrowser_class.clear()
        self.ui.textBrowser_graph.clear()
        self.ui.label_truegraph.clear()
    def save_img(self):
        if self.dataPIL.any():
            path_save = os.path.abspath('.')
            path_save = path_save + "\\output\\"
            now = datetime.datetime.now()
            other_StyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
            other_StyleTime = "".join(filter(str.isdigit, other_StyleTime))
            path_save = path_save + other_StyleTime
            Image.fromarray(self.dataPIL).save(path_save + '.jpg')
            with open(path_save + '.txt', 'w') as file_op:
                file_op.write("1、不同阈值下的聚类实现 \n %s \n 2、最佳分类结果 \n %s" % (str(self.tb1), str(self.tb2)))
                file_op.close()
    def other_save_img(self):
        QString = QtWidgets.QFileDialog.getSaveFileName(self,'选择另存为路径',".",'JPEG Files(*.jpg);;PNG Files(*.png)')
        Image.fromarray(self.dataPIL).save(QString[0])

    def soft(self):
        self.open_child(99)
    def soft_author(self):
        self.open_child(999)
    def init_ui(self):
        self.ui.pushButton_3.clicked.connect(self.click_3esc)
        self.ui.pushButton_esc.clicked.connect(self.click_esc)
        self.ui.pushButton_clear_show.clicked.connect(self.click_clear_show)
        self.ui.action_1.triggered.connect(self.click_load)
        self.ui.pushButton_start.clicked.connect(self.click_start)
        self.ui.action_2.triggered.connect(self.save_img)
        self.ui.action_3.triggered.connect(self.other_save_img)
        self.ui.action_4.triggered.connect(self.soft)
        self.ui.action_5.triggered.connect(self.soft_author)
        self.ui.pushButton_heat.clicked.connect(self.heat)
        self.show()
    def open_child(self,erro_flag):
        self.run_child1_go = run_child1(erro_flag)
        self.run_child1_go.show()
    def open_child2(self, K, object_name):
        try:
            self.run_child2_go = run_child2(self.K, self.object_name)
            self.run_child2_go.show()
        except Exception as e:
            print(f"打开子窗口出错: {str(e)}")
            self.open_child(3)

class run_child1(QtWidgets.QWidget):
    def __init__(self, erro_flag):
        super().__init__()
        self.runui = Qt_childwindows.Ui_select_content()
        self.runui.setupUi(self)
        self.erro_flag = erro_flag
        self.init_runui()
        self.erro_()
    def erro_(self):
        if self.erro_flag == 0:
            self.runui.textBrowser.append('方差为0，该组不适合用相关系数法!')
        elif self.erro_flag == 1:
            self.runui.textBrowser.append('需要输入权重!未输入权重，或者权重输入不正确，请在权重行矩阵输入框输出逗号隔开，元素数为性状数，元素整体和为1的权重矩阵')
        elif self.erro_flag == 2:
            self.runui.textBrowser.append('专家评判需分类数，未输入分类数,或者分类数输入格式不正确，分类数格式要求为不超过对象数的正整数!')
        elif self.erro_flag == 3:  # 新增的错误代码
            self.runui.textBrowser.append('没有可用的模糊等价矩阵数据，请先执行分类计算!')
        elif self.erro_flag == 99:
            self.runui.textBrowser.append('本程序为模糊聚类分析程序，目前评判指标代码还没有补充完整,建议使用专家分类数，相关功能还在继续完善中。')
        elif self.erro_flag == 999:
            self.runui.textBrowser.append('本程序为复旦大学现代物理研究所张斌课题组所有 \n 邮箱：binzhang@fudan.edu.cn')
    def init_runui(self):
        self.runui.pushButton.clicked.connect(self.close)

class run_child2(QtWidgets.QWidget):
    def __init__(self, K, object_name):
        super().__init__()
        self.K = K
        self.object_name = object_name
        self.runui = Qt_childwindows2.Ui_select_content()
        self.runui.setupUi(self)
        self.init_runui()
        self.show_heatmap()

    def show_heatmap(self):
        try:
            fig = plot_fuzzy_equiv_heatmap_with_dendrogram(
                K=self.K,
                object_name=self.object_name,
                figsize=(10, 10),
            )
            if fig is None:
                self.runui.textBrowser.append("热图绘制失败，请检查数据。")
                return
            fig_cbar = plot_colorbar_only(
                K=self.K,
                cmap_colors=["#F5DB91", "#E8B33C", "#DB8A2A", "#CE5E19", "#BF3308", "#A12B10", "#9E2512"],
                figsize=(1, 8),
            )

            if fig_cbar is None:
                self.runui.textBrowser.append("警告：颜色条绘制失败！")
            self.display_figure_in_label(fig, self.runui.label_heat)
            self.display_figure_in_label(fig_cbar, self.runui.label_cbar)
        except Exception as e:
            self.runui.textBrowser.append(f"热图绘制错误: {str(e)}")

    def display_figure_in_label(self, fig, label):
        if label.layout():
            old_layout = label.layout()
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            old_layout.deleteLater()
        layout = QtWidgets.QVBoxLayout(label)
        layout.setContentsMargins(0, 0, 0, 0)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        canvas.draw()
    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

    def init_runui(self):
        self.runui.pushButton.clicked.connect(self.close)


if __name__ == '__main__':
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    #np.warnings.filterwarnings('ignore', category=FutureWarning)
    app = QtWidgets.QApplication(sys.argv)
    run_main_go = run_main()
    run_main_go.show()
    sys.exit(app.exec_())  

