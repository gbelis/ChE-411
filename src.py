import cobra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import itertools


def build_BOF(model, df, name):
    reaction = cobra.Reaction(id = name, name = name)
    data = df.loc[name][:-2].reset_index()
    idx = data[data['index'] == '->'].index[0]
    data = pd.concat([data[:idx], data[idx+1:]])
    data[name] = data[name].astype(float)
    data.loc[:idx, name] = data.loc[:idx, name] * -1
    # Add metabolites and their stoichiometric coefficients
    reaction.add_metabolites({model.metabolites.get_by_id(met.replace('[', '_').replace(']', '')): stoch for stoch, met in zip(data[name], data['index'])})
    return reaction

# Creating conditions
def BOF_test(model, objective, BOF, BOF_bounds = (0,1000)):
    model.objective = objective
    model.reactions.get_by_id(BOF).bounds = BOF_bounds
    ammonium = -np.linspace(0,10,21)
    glucose = -np.linspace(0,10,21)
    df = pd.DataFrame(columns=['Glucose', 'Ammonium', 'Growth', 'Acetate', 'RQ'])
    #rates = np.empty((glucose.shape[0], ammonium.shape[0]))
    for am, gl in itertools.product(glucose,ammonium):
        model.reactions.get_by_id('EX_glc__D_e').bounds = (gl, 10)
        model.reactions.get_by_id('EX_nh4_e').bounds = (am, 10)
        sol = model.optimize('maximize')
        if sol.fluxes['EX_o2_e'] != 0:
            RQ = sol.fluxes['EX_co2_e'] / sol.fluxes['EX_o2_e']
        else: RQ = 0
        df.loc[len(df.index)] = [-gl, -am, sol.fluxes[BOF], sol.fluxes['EX_ac_e'], -RQ]
    return df
        # rates[i,j] = model.objective_value

def FBA(model, objective):
    # Step 1: Maximize growth (BOF)
    model.objective = objective
    sol = model.optimize('maximize')
    
    # If optimization is successful, append the growth value
    if sol.status == 'optimal':
        return sol.objective_value
    else:
        return 0


def pFBA_rq(model, BOF):
    model.objective = BOF
    try:
        sol = cobra.flux_analysis.pfba(model)
        if sol.status == 'optimal' and sol.fluxes['EX_o2_e'] != 0:
            rq = - sol.fluxes['EX_co2_e'] / sol.fluxes['EX_o2_e']
            return rq
        else:
            return rq
    except Exception as e:
        print(f"Error in pFBA: {e}")
        return 0
    


# Setting the bounds of all biomass reactions to zero initially

def fig_1_row(model,  BOF, res = 10, tol = 1e-10):
    model.reactions.get_by_id(BOF).bounds = (0, 0)

    # Define glucose and ammonium uptake ranges
    ammonium = -np.linspace(0,20, res)
    glucose = -np.linspace(0, 20, res)

    # Dataframe to store results
    df = pd.DataFrame(columns=['BOF','Glucose', 'Ammonium', 'Growth', 'Acetate', 'RQ'])

    # Loop over glucose and ammonium uptake combinations
    for am, gl in itertools.product(glucose, ammonium):
        # Set the bounds for glucose and ammonium uptake
        model.reactions.get_by_id('EX_glc__D_e').bounds = (gl, 10)
        model.reactions.get_by_id('EX_nh4_e').bounds = (am, 10)
        model.reactions.get_by_id('EX_ac_e').bounds = (0,1000)
        if isinstance(BOF, str):
            model.reactions.get_by_id(BOF).bounds = (0,1000)
        elif isinstance(BOF, dict):
            for reaction in BOF.keys():
                reaction.bounds = (0,1000)
        row = [BOF, -gl, -am]

        # Step 1: Maximize growth (BOF)
        model.objective = BOF
        sol = model.optimize('maximize')
        
        # If optimization is successful, append the growth value
        if sol.status == 'optimal':
            growth_flux = sol.objective_value
            if growth_flux>tol:
                row.append(sol.objective_value)
            else: 
                row += [-100,-100,-100]
                continue
        else:
            # print(f'no feasible solution for\nGlucose:\t{gl}\nAmmonium:\t{am}')
            row += [-100,-100,-100]
            df.loc[len(df.index)] = row
            continue  # Skip the rest if growth is 0 (no feasible solution)

        # Step 2: Fix the growth rate to the optimized value
        if isinstance(BOF, str):
            model.reactions.get_by_id(BOF).bounds = (growth_flux, growth_flux)
        elif isinstance(BOF, dict):
            for reaction, w in BOF.items():
                reaction.bounds = (w*growth_flux,w*growth_flux)


        # Step 3: Maximize acetate secretion
        model.objective = 'EX_ac_e'
        sol = model.optimize('maximize')
        
        # If optimization is successful, append the acetate secretion value
        if sol.status == 'optimal':
            acetate_flux = sol.fluxes['EX_ac_e']
            row.append(acetate_flux)
        else:
            row.append(0)
            acetate_flux = 0  # Just for fixing bounds later

        # Step 4: Fix acetate secretion rate
        model.reactions.get_by_id('EX_ac_e').bounds = (acetate_flux, acetate_flux)

        # Step 5: Perform parsimonious FBA (pFBA)
        model.objective = BOF
        try:
            sol = cobra.flux_analysis.pfba(model)
            if sol.status == 'optimal' and sol.fluxes['EX_o2_e'] != 0:
                rq = - sol.fluxes['EX_co2_e'] / sol.fluxes['EX_o2_e']
                row.append(rq)
            else:
                row.append(0)
        except Exception as e:
            print(f"Error in pFBA: {e}")
            row.append(0)
        # Append row to dataframe
        df.loc[len(df.index)] = row
    
    # Reset BOF bounds after the loop
    if isinstance(BOF, str):
            model.reactions.get_by_id(BOF).bounds = (0,1000)
    elif isinstance(BOF, dict):
        for reaction in BOF.keys():
            reaction.bounds = (0,1000)
    return df
   


def fig_1(model, df = None):
    dfs = []
    BOFs = ['WT', 'NitStarv', 'CarbStarv']
    for bof, in BOFs:
        df = fig_1_row(model, bof)
        dfs.append(df.copy())
    return pd.concat(dfs,axis=0)


# Plotting functions
def plot_heatmap(data, ax, metric, vmax = None):
    # Reshape data for heatmap
    heatmap_data = data.pivot(index='Glucose', columns='Ammonium', values=metric)

    
    # Mask -100 values by treating them as NaN temporarily
    # heatmap_data[heatmap_data<1e-10] Does not work
    mask_special_value = heatmap_data == -100
    heatmap_data = heatmap_data.replace(-100, np.nan)

    # Define the base colormap for the heatmap (excluding -100 values)
    vmin = 0
    if not vmax: vmax = 0, heatmap_data.max()
    cmap = plt.cm.inferno

    # Plot the heatmap, masking NaNs (including original -100 values)
    sns.heatmap(
        heatmap_data, cmap=cmap,vmin = vmin, vmax=vmax, mask=mask_special_value, annot=False, fmt=".1f", ax=ax,
        cbar_kws={'label': 'Values'}
    )

    # Overlay special values (-100) as light grey
    sns.heatmap(
        heatmap_data.replace(np.nan, -100), cmap=ListedColormap(['lightgrey']), mask=~mask_special_value, 
        annot=False, cbar=False, ax=ax
    )

    # Customize axes and labels
    ax.invert_yaxis()
    ax.set_ylabel('Glucose uptake')
    ax.set_xlabel('Ammonium uptake') 


def fig_1_plot(df):
    BOFs = ['WT', 'NitStarv', 'CarbStarv']
    metrics = ['Growth', 'Acetate', 'RQ']
    fig, axes = plt.subplots(3, 3, figsize=(18, 8), sharey=True, sharex=True)
    if isinstance(df, pd.core.frame.DataFrame): 
        for bof, ax_row in zip(BOFs, axes):
            d = df[df['BOF']==bof]
            for metric, ax in zip(metrics, ax_row):
                plot_heatmap(d, ax, metric, vmax=df[metric].max())


    axes[0,0].set_title('Growth'),9
    axes[0,1].set_title('Acetate Secretion')
    axes[0,2].set_title('RQ')

    row_titles = ["UL", "NL", "CL"]
    for i, title in enumerate(row_titles):
        fig.text(0.09, 0.8 - i * 0.3, title, va='center', ha='center', rotation='vertical', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0.1, 0, 1, 0.95])
    plt.show()
    return df