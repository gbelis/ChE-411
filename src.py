import cobra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,TwoSlopeNorm, LinearSegmentedColormap,Normalize
from scipy.interpolate import griddata

import numpy as np
import itertools


def build_BOF(model, df, name, reaction_name):
    reaction = cobra.Reaction(id = reaction_name, name = reaction_name)
    data = df.loc[name][:-2].reset_index()
    idx = data[data['index'] == '->'].index[0]
    data = pd.concat([data[:idx], data[idx+1:]])
    data[name] = data[name].astype(float)
    data.loc[:idx, name] = data.loc[:idx, name] * -1
    # Add metabolites and their stoichiometric coefficients
    reaction.add_metabolites({model.metabolites.get_by_id(met.replace('[', '_').replace(']', '')): stoch for stoch, met in zip(data[name], data['index'])})
    return reaction

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
        #print(f"Error in pFBA: {e}")
        return 0

def fig_1_row(model,  BOF, res = 10, tol = 1e-10):
    for biomass in ['UL', 'NL', 'CL', 'BIOMASS_Ec_iML1515_core_75p37M', 'BIOMASS_Ec_iML1515_WT_75p37M']:
        model.reactions.get_by_id(biomass).bounds = (0, 0)

    # Define glucose and ammonium uptake ranges
    ammonium = -np.linspace(0,20, res)
    glucose = -np.linspace(0,20, res)

    # Dataframe to store results
    df = pd.DataFrame(columns=['BOF','Glucose', 'Ammonium', 'Growth', 'Acetate', 'RQ'])
   
    best_growth = 0
    # Loop over glucose and ammonium uptake combinations
    for am, gl in itertools.product(glucose, ammonium):
        # Set the bounds for glucose and ammonium uptake
        model.reactions.get_by_id('EX_glc__D_e').bounds = (gl, gl)
        model.reactions.get_by_id('EX_nh4_e').bounds = (am, am)
        model.reactions.get_by_id('EX_ac_e').bounds = (0,1000)
        model.reactions.get_by_id(BOF).bounds = (0,1000)
        row = [BOF, -gl, -am]

        # Step 1: Maximize growth (BOF)
        model.objective = BOF
        sol = model.optimize('maximize')
        growth_flux = FBA(model, BOF)
        if growth_flux>tol and best_growth<growth_flux:
            row.append(sol.objective_value)
        else: 
            row += [0,0,0]
            df.loc[len(df.index)] = row
            continue

        # Step 2: Fix the growth rate to the optimized value
        if isinstance(BOF, str):
            model.reactions.get_by_id(BOF).bounds = (growth_flux, growth_flux)
        elif isinstance(BOF, dict):
            for reaction, w in BOF.items():
                reaction.bounds = (w*growth_flux,w*growth_flux)


        # Step 3: Maximize acetate secretion
        acetate_flux = FBA(model, 'EX_ac_e')
        row.append(acetate_flux)

        # Step 4: Fix acetate secretion rate
        model.reactions.get_by_id('EX_ac_e').bounds = (acetate_flux, acetate_flux)

        # Step 5: Perform parsimonious FBA (pFBA)
        row.append(pFBA_rq(model, BOF))
        
        # Append row to dataframe
        df.loc[len(df.index)] = row
    
    # Reset BOF bounds after the loop
    model.reactions.get_by_id(BOF).bounds = (0,1000)

    return df

def BTW(model,  BOFs, res = 10, tol = 1e-10):
    for biomass in BOFs + ['BIOMASS_Ec_iML1515_core_75p37M', 'BIOMASS_Ec_iML1515_WT_75p37M']:
        model.reactions.get_by_id(biomass).bounds = (0, 0)

    # Define glucose and ammonium uptake ranges
    ammonium = -np.linspace(0,20, res)
    glucose = -np.linspace(0,20, res)

    # Dataframe to store results
    df = pd.DataFrame(columns=['BOF','Glucose', 'Ammonium', 'Growth', 'Acetate', 'RQ'])
   
    best_growth = 0
    # Loop over glucose and ammonium uptake combinations
    for am, gl in itertools.product(glucose, ammonium):
        for w in zip(np.linspace(0, 1, 11), 1-np.linspace(0, 1, 11)):
            # Set the bounds for glucose and ammonium uptake
            model.reactions.get_by_id('EX_glc__D_e').bounds = (gl, gl)
            model.reactions.get_by_id('EX_nh4_e').bounds = (am, am)
            model.reactions.get_by_id('EX_ac_e').bounds = (0,1000)
            BOF={model.reactions.get_by_id(bof): w for bof, w in zip(BOFs, w)}
            if isinstance(BOF, str):
                model.reactions.get_by_id(BOF).bounds = (0,1000)
            elif isinstance(BOF, dict):
                for reaction in BOF.keys():
                    reaction.bounds = (0,1000)
            row = [{reaction.name: w for reaction,w in BOF.items()}, -gl, -am]

            # Step 1: Maximize growth (BOF)
            model.objective = BOF
            sol = model.optimize('maximize')
            growth_flux = FBA(model, BOF)
            if growth_flux>tol and best_growth<growth_flux:
                row.append(sol.objective_value)
            else: 
                row += [0,0,0]
                df.loc[len(df.index)] = row
                continue

            # Step 2: Fix the growth rate to the optimized value
            if isinstance(BOF, str):
                model.reactions.get_by_id(BOF).bounds = (growth_flux, growth_flux)
            elif isinstance(BOF, dict):
                for reaction, w in BOF.items():
                    reaction.bounds = (w*growth_flux,w*growth_flux)


            # Step 3: Maximize acetate secretion
            acetate_flux = FBA(model, 'EX_ac_e')
            row.append(acetate_flux)

            # Step 4: Fix acetate secretion rate
            model.reactions.get_by_id('EX_ac_e').bounds = (acetate_flux, acetate_flux)

            # Step 5: Perform parsimonious FBA (pFBA)
            row.append(pFBA_rq(model, BOF))
            
            # Append row to dataframe
            df.loc[len(df.index)] = row
    
    # Reset BOF bounds after the loop
    if isinstance(BOF, str):
            model.reactions.get_by_id(BOF).bounds = (0,1000)
    elif isinstance(BOF, dict):
        for reaction in BOF.keys():
            reaction.bounds = (0,1000)
    return df

def fig_1(model, df = None, res=10):
    dfs = []
    BOFs = ['UL', 'NL', 'CL']
    for bof in BOFs:
        df = fig_1_row(model, bof, res=res)
        dfs.append(df.copy())
    return pd.concat(dfs,axis=0)

# Plotting functions

def plot_heatmap(data, ax, metric, vmax = None, mask=None, cmap='rocket', norm=None):
    # Reshape data for heatmap
    heatmap_data = data.pivot(index='Glucose', columns='Ammonium', values=metric)
    if vmax == None:
        vmax=heatmap_data.max(axis=None)
    if norm is None:
        norm=Normalize(vmin=0,vmax=vmax)

    # Plot the heatmap, masking NaNs (including original -100 values)
    sns.heatmap(
        heatmap_data, vmax=vmax,vmin=0, annot=False, fmt=".1f", ax=ax,
        cbar_kws={'label': 'Values'}, cmap=cmap, norm=norm
    )

    # Grey area mask
    if not mask is None:
        sns.heatmap(
            data=mask, cmap=ListedColormap(['lightgrey']), 
            annot=False, cbar=False, ax=ax, mask=~mask
        )

    # Customize axes and labels
    ax.invert_yaxis()
    ax.set_ylabel('Glucose uptake')
    ax.set_xlabel('Ammonium uptake') 

def fig_1_plot(df):
    BOFs = ['UL', 'NL', 'CL']
    metrics = ['Growth', 'Acetate', 'RQ']
    fig, axes = plt.subplots(3, 3, figsize=(18, 8), sharey=True, sharex=True)
    if isinstance(df, pd.core.frame.DataFrame): 
        for bof, ax_row in zip(BOFs, axes):
            d = df[df['BOF']==bof]
            mask = d.pivot(index='Glucose', columns='Ammonium', values='Growth') < 1e-10
            for metric, ax in zip(metrics, ax_row):
                if metric=='RQ': cmap,norm=get_unity_colormap(vmax=df[metric].max())
                else:  
                    cmap='rocket'
                    norm=None
                plot_heatmap(d, ax, metric, vmax=df[metric].max(), mask=mask, cmap=cmap, norm=norm)


    axes[0,0].set_title('Growth')
    axes[0,1].set_title('Acetate Secretion')
    axes[0,2].set_title('RQ')

    row_titles = ["UL", "NL", "CL"]
    for i, title in enumerate(row_titles):
        fig.text(0.09, 0.8 - i * 0.3, title, va='center', ha='center', rotation='vertical', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0.1, 0, 1, 0.95])
    plt.show()
    return df

def get_unity_colormap(palette='rocket',vmin=0,vmax=10,blend_width=10):
    # Load the colormap from seaborn
    rocket = sns.color_palette(palette, as_cmap=True)

    # Define a custom colormap by blending inferno with pink at the center
    colors = rocket(np.linspace(0, 1, 256))
    
    center_index = int(256/(vmax-vmin))  # Approximate midpoint index in the colors array
    # Define pink as RGBA and create a gradient blend with the original colormap
    pink_rgba = np.array(np.array([255, 36, 228, 1])/255)  # RGBA for pink


    # Blend colors around the center with pink by gradually interpolating
    for i in range(-blend_width, blend_width + 1):
        # Compute the blending ratio (gradual transition around 1)
        ratio = (blend_width - abs(i)) / blend_width
        colors[center_index + i] = ratio * pink_rgba + (1 - ratio) * colors[center_index + i]

    cmap = LinearSegmentedColormap.from_list("unity", colors)
    norm = None#TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)

    return cmap, norm

