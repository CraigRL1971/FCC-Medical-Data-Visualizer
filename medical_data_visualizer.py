import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def weight_calc(weight, height):
    if weight / ((height / 100) ** 2) > 25:
        return 1
    else:
        return 0

def normalise(value):
    if value == 1:
        return 0
    else:
        return 1
    
# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = [weight_calc(x, y) for x, y in zip(df['weight'], df['height'])]

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = [normalise(x) for x in df['cholesterol']]
df['gluc'] = [normalise(x) for x in df['gluc']]

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.
    grouped_df_cat = df_cat.groupby(['cardio', 'variable']).agg({'value' : ['value_counts']}) 
    grouped_df_cat.columns = ['total']
    grouped_df_cat = grouped_df_cat.reset_index(level=['cardio', 'variable', 'value'])

    # Draw the catplot with 'sns.catplot( )'
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", kind="bar", data=grouped_df_cat).fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr().round(1)
 
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig = plt.subplots(1,1,figsize=(11, 9))
    
    # Draw the heatmap with 'sns.heatmap()'
    fig = sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, vmax=0.24, vmin=0.08, linewidths=0.5).figure

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
