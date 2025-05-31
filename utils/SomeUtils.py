import math
import os
from traceback import print_last
import numpy as np
import matplotlib.pyplot as plt

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def learning_rate(init, epoch):
    optim_factor = 0
    if epoch > 160:
        optim_factor = 3
    elif epoch > 120:
        optim_factor = 2
    elif epoch > 60:
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)

def learning_rate_2(epoch, warmup_steps, warmup_start_lr, max_iter, lr0, power=0.5):
    warmup_factor = (lr0/warmup_start_lr)**(1/warmup_steps)
    if epoch <= warmup_steps:
        lr = warmup_start_lr*(warmup_factor**epoch)
    else: # epoch <= max_iter/4*3:
        factor = (1-(epoch-warmup_steps)/(max_iter-warmup_steps))**power
        lr = lr0*factor
    # else:
    #     lr = learning_rate(5e-4, epoch-max_iter/4*3)
    return lr

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def draw_fig(args, data_list, name):
    import matplotlib
    matplotlib.use('Agg')  # FIX: tkinter.TclError: couldn't connect to display "localhost:11.0" 
    import matplotlib.pyplot as plt
    x1 = range(args.val_delta, args.epochs+1, args.val_delta)
    y1 = data_list

    plt.cla()
    plt.title(name.split('_')[-1]+' vs. epoch', fontsize=15)
    # plt.plot(x1, y1, '.-')
    plt.plot(x1, y1[args.val_delta-1::args.val_delta])
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel(name.split('_')[-1], fontsize=15)
    plt.grid()
    plt.savefig(args.save_dir+'/'+name+".png", dpi=900)

    # plt.show()

def draw_train_test(args, train_list, test_list, name):
    plt.figure(figsize=(10, 5))
    
    epochs = range(args.val_delta, args.epochs+1, args.val_delta)
    plt.plot(epochs, train_list[args.val_delta-1::args.val_delta], 'b-', label='Train Loss')
    plt.plot(epochs, test_list[args.val_delta-1::args.val_delta], 'r-', label='Test Loss')
    
    # 美化图形
    plt.title('Training & Test Loss')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend()
    plt.grid(True)
    
    # 自动调整y轴范围
    max_loss = max(max(train_list), max(test_list))
    plt.ylim(0, max_loss * 1.1)
    
    # 保存图片
    plt.savefig(args.save_dir+'/'+name+".png", dpi=900)
    plt.close()  # 关闭图形避免内存泄漏
    return 0

def try_make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def drawAnomaliesBySSC_A(points_2D, lower_limit, upper_limit):
    color = points_2D.T[:,0].copy()
    color_str = points_2D.T[:,0].copy().astype('str')
    
    color_str[(lower_limit<color)*(color<upper_limit)] = 'blue'
    color_str[upper_limit<points_2D.T[:,0]] = 'red'
    color_str[points_2D.T[:,0]<lower_limit] = 'red'
    
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(16, 8))
    points_2D = points_2D.T

    # 绘制散点图
    scatter = ax.scatter([x+1 for x in range(points_2D.shape[0])], points_2D[:, 0], c=color_str, s=10)

    # 设置标题和坐标轴标签，并调整字体大小和粗体显示
    # ax.set_title('Scatter Plot with Color Gradient', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=16, fontweight='bold')
    ax.set_ylabel('SSC-A', fontsize=16, fontweight='bold')

    # 更新刻度标签字体大小
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontsize(10)
        tick.set_fontweight('bold')

    # 移除背景颜色和网格线，并设置为白色背景
    ax.set_facecolor('white')  # 设置绘图区域背景颜色为白色
    fig.patch.set_facecolor('white')  # 设置整个绘图纸的背景颜色为白色
    ax.grid(False)  # 关闭网格线

    # 保存图像为高 DPI 格式的文件
    plt.savefig('/home/ljw/Code/AML/PreprocessResults/Time-SSCA.png', dpi=600, bbox_inches='tight')
    plt.close()
    return 0

def findAnomaliesBySSC_A(points_array, cut_off=2, draw_fig=False):  # 'SSC-A', 'FSC-A', 'FSC-H'  (15, 500000)
    new_array = []
    discard_array = []
    std = np.std(points_array, 1)[0]
    mean = np.mean(points_array, 1)[0]
    cut_off *= std
    lower_limit = mean-cut_off
    upper_limit = mean+cut_off

    for item in points_array.T:
        if lower_limit<item[0]<upper_limit:
            new_array.append(item)
        else:
            discard_array.append(item)

    if draw_fig:
        drawAnomaliesBySSC_A(points_array, lower_limit, upper_limit)
    return np.array(new_array).T

def drawAnomaliesByFSC_AH(points_2D, lower_limit, upper_limit):
    slopes = points_2D[2, :] / (points_2D[1, :]+0.001)
    color = slopes.copy()
    color_str = points_2D.T[:,0].copy().astype('str')
    
    color_str[(lower_limit<color)*(color<upper_limit)] = 'blue'
    color_str[upper_limit<slopes] = 'red'
    color_str[slopes<lower_limit] = 'red'
    
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(16, 8))
    points_2D = points_2D.T

    # 绘制散点图
    scatter = ax.scatter(x=points_2D[:, 1], y=points_2D[:, 2], c=color_str, s=10)

    # 设置标题和坐标轴标签，并调整字体大小和粗体显示
    # ax.set_title('Scatter Plot with Color Gradient', fontsize=16, fontweight='bold')
    ax.set_xlabel('FSC-A', fontsize=16, fontweight='bold')
    ax.set_ylabel('FSC-H', fontsize=16, fontweight='bold')

    # 更新刻度标签字体大小
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontsize(10)
        tick.set_fontweight('bold')

    # 移除背景颜色和网格线，并设置为白色背景
    ax.set_facecolor('white')  # 设置绘图区域背景颜色为白色
    fig.patch.set_facecolor('white')  # 设置整个绘图纸的背景颜色为白色
    ax.grid(False)  # 关闭网格线

    # 保存图像为高 DPI 格式的文件
    plt.savefig('/home/ljw/Code/AML/PreprocessResults/FSC-A-FSC-H.png', dpi=600, bbox_inches='tight')
    plt.close()
    return 0

def findAnomaliesByFSC_AH(points_array, cut_off=1, draw_fig=False):  # 'SSC-A', 'FSC-A', 'FSC-H'  (15, 500000)
    new_array = []
    discard_array = []

    # 求斜率数组，以FSC-A为x轴，FSC-H为y轴
    slope_array = points_array[2, :] / (points_array[1, :]+0.001)

    std = np.std(slope_array)
    mean = np.mean(slope_array)
    cut_off *= std
    lower_limit = mean-cut_off
    upper_limit = np.inf

    for i, item in enumerate(slope_array):
        if lower_limit<item<upper_limit:
            new_array.append(points_array.T[i, :])
        else:
            discard_array.append(points_array.T[i, :])

    if draw_fig:
        drawAnomaliesByFSC_AH(points_array, lower_limit, upper_limit)
    return np.array(new_array).T

def drawPoints(points, lower_limit, upper_limit):
    import plotly.express as px
    color = points.T[:, 1].copy()
    color_str = points.T[:, 1].copy().astype('str')
    
    color_str[(lower_limit<color)*(color<upper_limit)] = 'blue'
    color_str[upper_limit<=points.T[:, 1]] = 'red'
    color_str[points.T[:, 1]<=lower_limit] = 'red'

    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(16, 8))

    # 绘制散点图
    scatter = ax.scatter(x=points.T[:, 1], y=points.T[:, 0], c=color_str, s=10)

    # 设置标题和坐标轴标签，并调整字体大小和粗体显示
    # ax.set_title('Scatter Plot with Color Gradient', fontsize=16, fontweight='bold')
    ax.set_xlabel('FSC-A', fontsize=16, fontweight='bold')
    ax.set_ylabel('SSC-A', fontsize=16, fontweight='bold')
    ax.set_ylim(bottom=0, top=1000)

    # 更新刻度标签字体大小
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontsize(10)
        tick.set_fontweight('bold')

    # 移除背景颜色和网格线，并设置为白色背景
    ax.set_facecolor('white')  # 设置绘图区域背景颜色为白色
    fig.patch.set_facecolor('white')  # 设置整个绘图纸的背景颜色为白色
    ax.grid(False)  # 关闭网格线

    # 保存图像为高 DPI 格式的文件
    plt.savefig('/home/ljw/Code/AML/PreprocessResults/SFSC.png', dpi=600, bbox_inches='tight')
    plt.close()
    return 0

if __name__ == '__main__':
    # import numpy as np
    # a = np.ones((2, 4))*10.
    # b = np.random.random((2,4))
    # print(a*b)
    # print(b)
    
    
    # # 画learning rate曲线图
    # max_iter = 200
    # epochs = [i for i in range(max_iter)]
    # lr_list  = []
    # for epoch in epochs:
    #     lr_list.append(learning_rate_2(epoch, 10, 1e-5, max_iter, 1e-3, 2))
    #     # lr_list.append(learning_rate(1e-3, epoch))
    
    # import matplotlib
    # matplotlib.use('Agg')  # FIX: tkinter.TclError: couldn't connect to display "localhost:11.0" 
    # import matplotlib.pyplot as plt
    # x1 = range(1, max_iter+1)
    # y1 = lr_list

    # plt.cla()
    # plt.title('lr test'+' vs. epoch', fontsize=15)
    # # plt.plot(x1, y1, '.-')
    # plt.plot(x1, y1)
    # plt.xlabel('epoch', fontsize=15)
    # plt.ylabel('lr_test', fontsize=15)
    # plt.grid()
    # plt.savefig('test.png')
    
    import torch.nn.functional as F
    import torch
    print(torch.arange(0, 5) % 3)
    print(F.one_hot(torch.arange(0, 5) % 3, num_classes=5))
    print(F.one_hot(torch.tensor([1,2,0]), num_classes=5))