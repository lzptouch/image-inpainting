from os.path import basename
import random
import wx
import os
import time
import sys
from imageio import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import opt
from test import test
from utils import resize, load_flist
import numpy as np
import cv2


def crop():
    image = imread(r"input_image.jpg")
    image = resize(image, 256, 256)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("input_image.jpg", image)


def errase(f):
    # 关联到网络的随机擦除函数
    mask1 = mask2 = None

    if f == "random" or f =="":
        mask_data = load_flist(opt.TEST_MASK_FLIST)
        mask_index = random.randint(0, len(mask_data) - 1)
        mask1 = imread(mask_data[mask_index])
        # print(mask_index)
        mask1 = resize(mask1, 256, 256)
        mask_ = mask1
    else:
        mask2 = np.zeros((256,256))
        mask2[64:192,64:192]=1.0
        mask_ = mask2

    mask_ = (mask_ > 0).astype(np.uint8) #* 255  # threshold due to interpolation
    mask_ = np.array([mask_, mask_, mask_]).swapaxes(0, 1).swapaxes(1, 2)
    image = imread(r"input_image.jpg")
    image = resize(image, 256, 256)
    images_masked = (image * (1 - mask_)) + mask_


    images_masked = cv2.cvtColor(images_masked, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor((image).astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite("input_image.jpg", image)
    cv2.imwrite("errased_image.jpg", images_masked)

    if mask1 is not None:
        cv2.imwrite("mask.jpg", mask1)
    else:
        cv2.imwrite("mask.jpg", mask2*255)




def inpainting():
    # 关联到网络
    out = test()
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite("restored.jpg", out)



def evauate():
    img_pred = (imread("restored.jpg") / 255.0).astype(np.float32)

    # img_gt =imread(path_true + '/'+dir+ '/' + basename(str(fn)))
    img_gt = (imread("input_image.jpg") / 255.0).astype(np.float32)

    img_gt = rgb2gray(img_gt)  # 去掉维度通道，将三通道变为一通道
    img_pred = rgb2gray(img_pred)

    psnr = compare_psnr(img_gt, img_pred, data_range=1)
    ssim = compare_ssim(img_gt, img_pred, data_range=1, win_size=51)
    return round(psnr,4), round(ssim,4)









# 窗口大小
x = 1220
y = 900
image_length = 256
controlbar_length = 350
controlbar_high = 200
inputbar_length = 256

# 菜单栏
ID_EXIT = 200
ID_ABOUT = 201
Version = "V1.0"
ReleaseDate = "2022-10-22"
# 背景
frame_title = '基于深度学习算法的图像修复软件'


# 主框架




class MainFrame(wx.Frame):
    def __init__(self, parent, id, title):
        # 设置窗口不能最大化
        wx.Frame.__init__(self, parent, id, title, size=(x, y),
                          style=wx.DEFAULT_FRAME_STYLE ^ wx.MAXIMIZE_BOX)
        # 图像名称背景颜色
        self.SetBackgroundColour(wx.Colour(235, 235, 235))
        # 调用UI
        self.initUI()
        # 调用状态栏
        self.setupStatusBar()
        # 调用菜单栏
        self.setupMenuBar()
        # 调用图标
        self.setupIcon()
        # 初始化图像板
        self.init_panel()
        # 窗口关闭事件
        self.Bind(wx.EVT_CLOSE, self.exit_sys)

    # 创建画板
    def init_panel(self):

        # 图片1
        self.input_bmp = wx.Panel(self, pos=(100, 110), size=(256, 256))
        self.crupted_bmp = wx.Panel(self, pos=(460, 110), size=(256, 256))
        self.restored_bmp = wx.Panel(self, pos=(100, 460), size=(256, 256))
        self.evaluate_bmp = wx.Panel(self, pos=(460, 460), size=(256, 256))

        # 设置面板颜色
        self.input_bmp.SetBackgroundColour(wx.Colour(210, 210, 210))
        self.crupted_bmp.SetBackgroundColour(wx.Colour(210, 210, 210))
        self.restored_bmp.SetBackgroundColour(wx.Colour(210, 210, 210))
        self.evaluate_bmp.SetBackgroundColour(wx.Colour(224, 224, 224))

        # 控制板
        self.controlbar_bmp = wx.Panel(self, pos=(800, 110), size=(330, 280))
        self.controlbar_bmp.SetBackgroundColour(wx.Colour(242, 242, 242))

        # 输入栏
        self.inputBar_bmp = wx.Panel(self, pos=(760, 460), size=(360, 250))
        self.inputBar_bmp.SetBackgroundColour(wx.Colour(242, 242, 242))

    # 创建UI
    def initUI(self):

        self.input_image = None
        self.crupted_image = None
        self.restored_image = None

        # 按键字体
        fontButton = wx.Font(12, wx.SWISS, wx.ROMAN, wx.BOLD)

        # 设置按键大小 位置
        self.image_input = wx.Button(self, -1, u"input", (830, 180), (120, 50))
        self.random_erase = wx.Button(self, -1, u"erase", (830, 240), (120, 50))
        self.inpainting = wx.Button(self, -1, u"restore", (830, 300), (120, 50))
        self.evaluate = wx.Button(self, -1, u"evaluate", (980, 180), (120, 50))
        self.image_clear = wx.Button(self, -1, u"clear", (980, 240), (120, 50))
        self.quit = wx.Button(self, -1, u"quit", (980, 300), (120, 50))

        # 每个按键配置字体
        self.image_input.SetFont(fontButton)
        self.random_erase.SetFont(fontButton)
        self.inpainting.SetFont(fontButton)
        self.evaluate.SetFont(fontButton)
        self.image_clear.SetFont(fontButton)
        self.quit.SetFont(fontButton)

        # 按键颜色
        self.image_input.SetBackgroundColour(wx.Colour(200, 200, 200))
        self.random_erase.SetBackgroundColour(wx.Colour(224, 224, 224))
        self.inpainting.SetBackgroundColour(wx.Colour(255, 204, 229))
        self.evaluate.SetBackgroundColour(wx.Colour(229, 255, 204))
        self.image_clear.SetBackgroundColour(wx.Colour(204, 204, 255))
        self.quit.SetBackgroundColour(wx.Colour(255, 102, 102))


        # 按键事件绑定
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.image_input)
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.random_erase)
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.inpainting)
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.evaluate)
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.image_clear)
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.quit)



        # 图片名字2，设置格式居中
        self.input_bmp_FileName = wx.StaticText(self, pos=(100, 380),
                                                size=(image_length, 25),
                                                style=wx.ALIGN_CENTRE_HORIZONTAL)
        self.crupted_bmp_FileName = wx.StaticText(self, pos=(460, 380),
                                                  size=(image_length, 25),
                                                  style=wx.ALIGN_CENTRE_HORIZONTAL)
        self.restored_bmp_FileName = wx.StaticText(self, pos=(100, 730),
                                                   size=(image_length, 25),
                                                   style=wx.ALIGN_CENTRE_HORIZONTAL)
        self.evaluate_bmp_FileName = wx.StaticText(self, pos=(460, 730),
                                                   size=(image_length, 25),
                                                   style=wx.ALIGN_CENTRE_HORIZONTAL)

        self.input_bmp_FileName.SetBackgroundColour(wx.Colour(230,230,230))
        self.crupted_bmp_FileName.SetBackgroundColour(wx.Colour(230,230,230))
        self.restored_bmp_FileName.SetBackgroundColour(wx.Colour(230,230,230))
        self.evaluate_bmp_FileName.SetBackgroundColour(wx.Colour(230,230,230))

        # 图片2，按键按下之前 定义好三个框内图片的名字 并显示
        self.input_bmp_FileName.SetLabelText('input image')
        self.crupted_bmp_FileName.SetLabelText('corruped image')
        self.restored_bmp_FileName.SetLabelText('restored image')
        self.evaluate_bmp_FileName.SetLabelText('图片修复质量')

        self.input_bmp_FileName.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))
        self.crupted_bmp_FileName.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))
        self.restored_bmp_FileName.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))
        self.evaluate_bmp_FileName.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))

        # 标题，设置格式居中
        self.title_name = wx.StaticText(self, pos=(150, 15), size=(900, 40),
                                        style=wx.ALIGN_CENTRE_HORIZONTAL)
        self.title_name.SetLabelText(frame_title)
        self.title_name.SetBackgroundColour(wx.Colour(255, 255, 255))
        # 字体大小
        self.title_name.SetFont(wx.Font(30, wx.SWISS, wx.NORMAL, wx.NORMAL))

        # 控制面板标题+边界
        self.controlbar_title = wx.StaticText(self, pos=(810, 110), size=(140, 40),
                                              style=wx.ALIGN_CENTRE_HORIZONTAL)
        self.controlbar_title.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.controlbar_title.SetLabelText('控制面板')
        self.controlbar_title.SetFont(wx.Font(20, wx.SWISS, wx.NORMAL, wx.NORMAL))

        # 擦除位置输入面板标题
        self.erasebar_title = wx.StaticText(self, pos=(780, 460), size=(260, 25),
                                            style=wx.ALIGN_CENTRE_HORIZONTAL)
        self.erasebar_title.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.erasebar_title.SetLabelText('图像擦除方式')
        self.erasebar_title.SetFont(wx.Font(20, wx.SWISS, wx.NORMAL, wx.NORMAL))

        # 输入面板静态文本
        self.erase_tip = wx.StaticText(self, pos=(810, 550), size=(200, 20),
                                       style=wx.ALIGN_CENTRE_HORIZONTAL)
        self.erase_tip.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.erase_tip.SetLabelText('随机random/中央center： ')
        self.erase_tip.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))

        # 输入面板控制文本
        self.erase_input = wx.TextCtrl(self, pos=(810, 630), size=(260, 25))
        self.erase_input.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.erase_input.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL))

        # PSNR/SSIM面板内容，4个静态文本
        # PSNR_Bar1
        self.PSNR_Bar1 = wx.StaticText(self, pos=(480, 540),
                                       size=(70, 25),
                                       style=wx.TE_RIGHT)
        self.PSNR_Bar1.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.PSNR_Bar1.SetLabelText('PSNR:')
        self.PSNR_Bar1.SetFont(wx.Font(13, wx.SWISS, wx.NORMAL, wx.NORMAL))

        # PSNR_Bar2
        self.PSNR_Bar2 = wx.StaticText(self, pos=(550, 540),
                                       size=(140, 25),
                                       style=wx.TE_LEFT)
        self.PSNR_Bar2.SetBackgroundColour(wx.Colour(245, 245, 245))
        self.PSNR_Bar2.SetFont(wx.Font(13, wx.SWISS, wx.NORMAL, wx.NORMAL))

        # PSNR_Bar3
        self.PSNR_Bar3 = wx.StaticText(self, pos=(480, 620),
                                       size=(70, 25),
                                       style=wx.TE_RIGHT)
        self.PSNR_Bar3.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.PSNR_Bar3.SetLabelText('SSIM:')
        self.PSNR_Bar3.SetFont(wx.Font(13, wx.SWISS, wx.NORMAL, wx.NORMAL))

        # PSNR_Bar4
        self.PSNR_Bar4 = wx.StaticText(self, pos=(550, 620),
                                       size=(140, 25),
                                       style=wx.TE_LEFT)
        self.PSNR_Bar4.SetBackgroundColour(wx.Colour(245, 245, 245))
        self.PSNR_Bar4.SetFont(wx.Font(13, wx.SWISS, wx.NORMAL, wx.NORMAL))

    def OnClick(self, event):
        if event.GetEventObject() == self.image_input:
            # 导入图片并显示在一个框内
            wildcard = 'All files(*.*)|*.*'

            dialog = wx.FileDialog(None, 'select', os.getcwd(), '', wildcard, wx.FD_OPEN)
            if dialog.ShowModal() == wx.ID_OK:
                image_path = dialog.GetPath()
                image_name = basename(image_path)
                self.input_bmp_FileName.SetLabelText('input image: ' + image_name)
                # 等比压缩
                self.input_image = wx.Image(image_path, wx.BITMAP_TYPE_ANY)
                self.input_image.SaveFile("input_image.jpg", wx.BITMAP_TYPE_BMP)
                crop()
                self.input_image = wx.Image('input_image.jpg', wx.BITMAP_TYPE_ANY)
                self.show1 = wx.StaticBitmap(self, -1, self.input_image.ConvertToBitmap(),
                                             pos=(100, 110),
                                             size=(256, 256)
                                             )


        elif event.GetEventObject() == self.random_erase:
            erase_input_abc = self.erase_input.GetValue()
            if erase_input_abc == "random" or erase_input_abc =="":
                errase(erase_input_abc)
                self.errased_image = wx.Image("errased_image.jpg", wx.BITMAP_TYPE_ANY).ConvertToBitmap()

                self.show2 = wx.StaticBitmap(self, -1, self.errased_image,
                                             pos=(460, 110),
                                             size=(256, 256)
                                             )

            elif erase_input_abc == "center":
                errase(erase_input_abc)
                self.errased_image = wx.Image("errased_image.jpg", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
                self.show2 = wx.StaticBitmap(self, -1, self.errased_image,
                                             pos=(460, 110),
                                             size=(256, 256)
                                             )

            else:
                self.errorEvent(self)

        elif event.GetEventObject() == self.inpainting:
            inpainting()
            self.restored_image = wx.Image("restored.jpg", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.show3 = wx.StaticBitmap(self, -1, self.restored_image,
                                         pos=(100, 460),
                                         size=(256, 256)
                                         )
        elif event.GetEventObject() == self.evaluate:
            psnr,ssim = evauate()
            self.PSNR_Bar2.SetLabelText("%.4f" % round(psnr, 4))
            self.PSNR_Bar4.SetLabelText("%.4f" % round(ssim, 4))

        elif event.GetEventObject() == self.image_clear:
            self.show1.Destroy()
            self.input_bmp_FileName.SetLabelText('input image')
            self.show2.Destroy()
            self.show3.Destroy()

            self.PSNR_Bar2.SetLabelText("")
            self.PSNR_Bar4.SetLabelText("")

            self.erase_input.SetValue("")



        elif event.GetEventObject() == self.quit:
            self.Close()

    # 创建状态栏
    def setupStatusBar(self):
        # 状态栏的创建
        self.CreateStatusBar(2)
        self.SetStatusWidths([-1, -1])
        # 状态栏1
        self.SetStatusText("ready", 0)
        # 状态栏2，timer
        self.timer = wx.PyTimer(self.Notify)
        self.timer.Start(1000, wx.TIMER_CONTINUOUS)
        self.Notify()

    def Notify(self):
        t = time.localtime(time.time())
        st = time.strftime('%Y-%m-%d   %H:%M:%S', t)
        self.SetStatusText(st, 1)

    # 创建菜单栏
    def setupMenuBar(self):
        # 创建菜单栏
        # 主菜单
        menubar = wx.MenuBar()
        # 子菜单f ：退出(Quit)
        fmenu = wx.Menu()
        quit_menu = wx.MenuItem(fmenu, ID_EXIT, u'Quit(&Q)', 'Terminate the program')
        # 添加一个图标
        quit_menu.SetBitmap(wx.Bitmap('./pit/quit.jpg'))
        # 将Quit添加到File中
        fmenu.AppendItem(quit_menu)
        # 将File添加到菜单栏中
        menubar.Append(fmenu, u'File(&F)')
        # 帮助菜单
        hmenu = wx.Menu()
        about_menu = wx.MenuItem(fmenu, ID_ABOUT, u'关于(&A)', 'More information about this program')
        # 添加一个图标
        about_menu.SetBitmap(wx.Bitmap('./pit/about.jpg'))
        # 将About添加到Help中
        hmenu.AppendItem(about_menu)
        # 将Help添加到菜单栏中
        menubar.Append(hmenu, u'Help(&H)')
        self.SetMenuBar(menubar)
        # 菜单中子菜单，事件行为的绑定即实现
        wx.EVT_MENU(self, ID_EXIT, self.OnMenuExit)
        wx.EVT_MENU(self, ID_ABOUT, self.OnMenuAbout)

    def OnMenuExit(self, event):
        self.Close()

    def OnMenuAbout(self, event):
        dlg = AboutDialog(None, -1)
        dlg.ShowModal()

    # 创建图标
    def setupIcon(self):
        # 图标的实现
        self.img_path = os.path.abspath("pit/diannao.png")
        icon = wx.Icon(self.img_path, type=wx.BITMAP_TYPE_PNG)
        self.SetIcon(icon)

    # 窗口关闭事件
    def exit_sys(self, event):
        toastone = wx.MessageDialog(None, "确定要退出系统吗？", "确认信息提示",
                                    wx.YES_NO | wx.ICON_QUESTION)
        if toastone.ShowModal() == wx.ID_YES:  # 如果点击了提示框的确定按钮
            toastone.Destroy()
            sys.exit(1)

    def errorEvent(self, event):
        dlg2 = errorDialog(None, -1)
        dlg2.ShowModal()


# 菜单栏对话框
class AboutDialog(wx.Dialog):
    def __init__(self, parent, id):
        wx.Dialog.__init__(self, parent, id, 'About Me', size=(250, 200))

        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"基于特征生成的图像修复软件"),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=20)
        self.sizer1.Add(wx.StaticText(self, -1, "Version %s , %s" % (Version, ReleaseDate)),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=10)
        self.sizer1.Add(wx.StaticText(self, -1, u"Authors:ZhiPeng Li ,DongMin Chen\n"),
                        0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=10)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER | wx.BOTTOM)
        self.SetSizer(self.sizer1)


# 输入错误框
class errorDialog(wx.Dialog):
    def __init__(self, parent, id):
        wx.Dialog.__init__(self, parent, id, '输入错误', size=(250, 200))

        self.SetBackgroundColour(wx.Colour(255,255,255))
        self.sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.sizer2.Add(wx.StaticText(self, 0, u"   输入错误，请重新输入正确参数\n"),
                        0, wx.ALIGN_CENTRE_HORIZONTAL | wx.TOP, border=40)
        self.sizer2.Add(wx.Button(self, wx.ID_OK), 0,
                        wx.ALIGN_CENTER | wx.BOTTOM, border=10)

        self.SetSizer(self.sizer2)


class App(wx.App):
    def __init__(self):
        super(self.__class__, self).__init__()

    def OnInit(self):
        # 初始化
        self.version = u" " + Version
        self.title = frame_title + self.version
        frame = MainFrame(None, -1, self.title)
        frame.Show(True)
        return True


if __name__ == "__main__":
    app = App()
    app.MainLoop()
