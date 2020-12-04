import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

def get_data(str, tag):
    tmp = str.split(f"{tag}: ")[1].split(" ")[0]
    return float(tmp)

def mask_loss_analysis(log_path, max_it, gap, exp, val, it_ana):
    it = []
    loss_mask = []
    with open(log_path) as f:
        for line in f:
            if "eta" not in line:
                continue
            cit = get_data(line, 'iter')
            if cit > max_it:
                continue
            it.append(cit)
            loss_mask.append(get_data(line, 'loss_mask'))
    index = range(0, len(it), gap)
    for i in range(len(index)):
        if i==0: continue
        idx = index[i]
        win = loss_mask[idx-gap:idx]
        e = sum(win)/gap
        pows = [pow(x-e,2) for x in win]
        exp.append(e)
        val.append(pow(sum(pows)/gap, 0.5))
        it_ana.append(it[idx-gap])
    exp.append(exp[-1])
    val.append(val[-1])
    it_ana.append(it[-1])

if __name__ == '__main__':
    max_it = 180000
    gap = 50
    log1 = '../train/output_r50_200000_ROI512_augment/r50_200000_ROI512_augment.log'
    exp1, val1, it1 = [], [], []
    log2 = '../train/output_x101_180000_ROI512_non-augment/x101_180000_ROI512_non-augment.log'
    exp2, val2, it2 = [], [], []

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    mask_loss_analysis(log1, max_it, gap, exp1, val1, it1)
    mask_loss_analysis(log2, max_it, gap, exp2, val2, it2)
    ax.set_xlim((0,max_it))

    ax.plot(it1, exp1, label=f"exp_{log1.split('_')[-1].split('.')[0]}")
    ax.plot(it2, exp2, label=f"exp_{log2.split('_')[-1].split('.')[0]}")
    ax.plot(it1, val1, label=f"val_{log1.split('_')[-1].split('.')[0]}")
    ax.plot(it2, val2, label=f"val_{log2.split('_')[-1].split('.')[0]}")

    # 设置坐标轴、图片名称
    ax.set_xlabel('iters')
    ax.set_title('mask loss analysis')
    ax.legend(loc='upper right')
    ax.set_ylabel('mask loss')

    plt.savefig('./charts/mask_loss_variance_analysis.png')
    plt.show()
