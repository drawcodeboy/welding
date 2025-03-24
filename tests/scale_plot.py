import os, sys
sys.path.append(os.getcwd())

from datasets import WeldingDataset

import seaborn as sns
import matplotlib.pyplot as plt

def main():
    ds_1 = WeldingDataset(mode='train',
                          train_num=400,
                          test_num=0,
                          scale=True)
    
    ds_2 = WeldingDataset(mode='train',
                          train_num=400,
                          test_num=0,
                          scale=False)
    
    select = 1
    
    norm_li, unnorm_li = [[] for i in range(0, 7)], [[] for i in range(0, 7)]
    for i in range(0, len(ds_1)):
        sample_1, _ = ds_1[i]
        sample_2, _ = ds_2[i]
        for j in range(0, 7):
            if j == 0 or j == 3 or j == 6: continue
            norm_li[j].append(sample_1[j].item())
            unnorm_li[j].append(sample_2[j].item())
    
    if select == 0:
        for idx, samples in enumerate(norm_li):
            plt.xlim(-0.1, 1.1)
            sns.kdeplot(samples)
    else:
        for samples in unnorm_li:
            sns.kdeplot(samples)
    
    plt.legend()
    plt.savefig('scale_plot_test.png')

if __name__ == '__main__':
    main()