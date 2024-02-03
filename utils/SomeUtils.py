import math
import os
from traceback import print_last
import numpy as np

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
    x1 = range(1, args.epochs+1)
    y1 = data_list

    plt.cla()
    plt.title(name.split('_')[-1]+' vs. epoch', fontsize=15)
    # plt.plot(x1, y1, '.-')
    plt.plot(x1, y1)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel(name.split('_')[-1], fontsize=15)
    plt.grid()
    plt.savefig(args.save_dir+'/'+name+".png")

    # plt.show()

def try_make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def drawAnomaliesBySSC_A(points_2D, lower_limit, upper_limit):
    import plotly.express as px
    color = points_2D.T[:,0].copy()
    
    color[(lower_limit<color)*(color<upper_limit)] = 0
    color[upper_limit<points_2D.T[:,0]] = 1
    color[points_2D.T[:,0]<lower_limit] = 1
    
    fig = px.scatter(x=[x+1 for x in range(points_2D.shape[1])], y=points_2D.T[:,0], color=color, title='Scatter Plot with Color Gradient', labels=dict(x="Time", y="SSC-A"))
    fig.show()
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
    import plotly.express as px
    slopes = points_2D[2, :] / (points_2D[1, :]+0.001)
    color = slopes.copy()
    
    color[(lower_limit<color)*(color<upper_limit)] = 0
    color[upper_limit<slopes] = 1
    color[slopes<lower_limit] = 1
    
    fig = px.scatter(x=points_2D[1, :], y=points_2D[2, :], color=color, title='Scatter Plot with Color Gradient', labels=dict(x="FSC-A", y="FSC-H"))
    fig.show()
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

def drawPoints(points):
    import plotly.express as px
    fig = px.scatter(x=points[1, :], y=points[2, :], title='Scatter Plot with Color Gradient', labels=dict(x="FSC-A", y="SSC-A"))
    fig.show()
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