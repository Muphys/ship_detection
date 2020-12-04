import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

def get_data(str, tag):
    tmp = str.split(f"{tag}: ")[1].split(" ")[0]
    return float(tmp)

if __name__ == '__main__':
    it = []	
    total_loss = []
    loss_cls = []
    loss_box_reg = []
    loss_mask = []
    loss_rpn_cls = []
    loss_rpn_loc =[]
    lr=[]

    log_file = '../train/output_r50_70000_non-ROI_augment/r50_70000_non-ROI_augment.log'

    with open(log_file) as f:
        for line in f:
            if "eta" not in line:
                continue
            cit = get_data(line, 'iter')
            if cit>=20000: 
                continue
            it.append(cit)
            total_loss.append(get_data(line, 'total_loss'))
            loss_cls.append(get_data(line, 'loss_cls'))
            loss_box_reg.append(get_data(line, 'loss_box_reg'))
            loss_mask.append(get_data(line, 'loss_mask'))
            loss_rpn_cls.append(get_data(line, 'loss_rpn_cls'))
            loss_rpn_loc.append(get_data(line, 'loss_rpn_loc'))
            lr.append(get_data(line, 'lr'))

    fig = plt.figure(figsize=(8,6))
    ax2 = fig.add_subplot(111)
    ax1 = ax2.twinx()


    ax1.plot(it, total_loss, color='red', label='total_loss')
    ax1.plot(it, loss_cls, color='purple', label='loss_cls')
    ax1.plot(it, loss_box_reg, color='blue', label='loss_box_reg')
    ax1.plot(it, loss_mask, color='orange', label='loss_mask')
    #ax1.plot(it, loss_rpn_cls, color='olive', label='loss_rpn_cls')
    #ax1.plot(it, loss_rpn_loc, color='green', label='loss_rpn_loc')
    #设置坐标轴范围
    ax1.set_xlim((0,it[-1]))
    ax1.set_ylim((0,1))

    # 设置坐标轴、图片名称
    ax1.set_xlabel('iters')
    log_name = log_file.split('.log')[0].split('/')[-1]
    cfg = log_name.split('_')
    ax1.set_title(f'{cfg[-2]}_{cfg[-1]}')


    ax2.plot(it, lr, color='black', label='lr')

    ax2.set_ylim([0, max(lr)*1.1])
    ax1.legend(loc='upper right')
    ax2.legend(loc='center right')


    ax1.set_ylabel('loss')
    ax2.set_ylabel('learning rate')

    plt.savefig(log_file[:-4]+'_0-20000.png')
    plt.show()
