import os, sys
sys.path.append(os.getcwd())

import pandas as pd

def get_samples(df):
    # ['TIG-알루미늄(BOP)', 'TIG-알루미늄(T-Fillet)', 'TIG-알루미늄(V-Groove)', 'TIG-알루미늄(I-Groove)', 'TIG-연강(BOP', 'TIG-연강(T-Fillet)', 'TIG-연강(V-Groove)', 'TIG-연강(I-Groove)']
    sheet_names = list(df.keys())
    
    sample_li = []
    
    for sheet_name in sheet_names:
        sample_li.extend([list(df[sheet_name].iloc[i]) for i in range(3, 28)])
        sample_li.extend([list(df[sheet_name].iloc[i]) for i in range(37, 62)])
    
    return sample_li

def main():
    df = pd.read_excel('data/welding/data.xlsx', sheet_name=None)
    
    sample_li = get_samples(df)
    
    for sample in sample_li:
        print(sample)
    print(f"Number of samples: {len(sample_li)}")
    
    
if __name__ == '__main__':
    main()