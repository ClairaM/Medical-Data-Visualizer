import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - read in csv file
df = pd.read_csv('medical_examination.csv')

# 2 - checks to see if patient is overweight and adds column
df['overweight'] = ((df['weight'] / (df['height'] * .01) ** 2) > 25).astype(int)

# 3 - updates columns to have 0 for good cholesterol and glucose and 1 for bad cholestrol and glucose
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5 - create a dataframe for the cat plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], var_name='variable', value_name='value')

    # 6 - group and reformat the data in df_cat to split by cardio
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')    

    # 7 - create graph
    graph = sns.catplot(data=df_cat, x='variable', y='count', col='cardio', kind='bar', col_wrap=2, hue='value')
    graph.set_axis_labels('variable', 'total')

    # 8
    fig = graph


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11 - clean up data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))]
    
    # 12 - create correlation matrix
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype = bool))

    # 14
    fig, ax = plt.subplots()

    # 15
    
    ax = sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', vmax=0.4, center=0, square=True)

    # 16
    fig.savefig('heatmap.png')
    return fig
