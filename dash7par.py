# Style Components


colors = {'background':'#111111','text':'#7FDBFF'}
container = {
    'width' : '1208px',
    'height': '1025px',
    'background': colors['background'],
    'border':'2px green solid',
    'font-family':'Arial'
}

map = {
    'width': '892px',
    'height': '596px',
    'padding': '2px',
    'backgroundColor': "black",
    'align':'right',
    'vertical-align':'top',
    'display':'inline-block',
    'border':'2px green solid'

}

map_plot = {
    'width': '888px',
    'height': '592px',
    'padding': '2px',
    'backgroundColor': "black",
    'align':'right',
    'vertical-align':'top',
    'display':'inline-block',
    'border':'2px green solid'

}

controls = {
    'width': '300px',
    'height': '596px',
    'padding': '2px',
    'overflow':'auto',
    'backgroundColor': "black",
    'align':'left',
    'vertical-align':'top',
    'display':'inline-block',
    'border':'2px green solid'

}

disease = {
    'width': 'auto',
    'margin':'auto',
    'height': '150px',
    'overflow':'auto',
    'padding': '2px',
    'align':'center',
    'color':'white',
    'vertical-align':'bottom',
    'font-size':'12px',
    'text-align':'left',
    'border':'2px white solid',
    'multi':True,
    'font-family':'Arial'

}

state = {
    'width': 'auto',
    'margin':'auto',
    'height': '215px',
    'overflow':'auto',
    'padding': '2px',
    'align':'center',
    'color':'white',
    'vertical-align':'bottom',
    'font-size':'14px',
    'text-align':'left',
    'border':'2px white solid',
    'multi':True,
    'font-family':'Arial'

}

button1 = {
    'width': '100px',
    'margin-top':'2px',
    'height': '35px',
    'padding': '2px',
    'align':'center',
    'background': 'black',
    'color':'white',
    'vertical-align':'bottom',
    'font-size':'16px',
    'text-align':'center',
    'border':'2px white solid',
    'font-family':'Arial',
    'font-weight':'bold'

}
button2 = {
    'width': '100px',
    'margin-top':'2px',
    'height': '35px',
    'padding': '2px',
    'align':'center',
    'background': 'black',
    'color':'white',
    'vertical-align':'bottom',
    'font-size':'16px',
    'text-align':'center',
    'border':'2px white solid',
    'font-family':'Arial',
    'font-weight':'bold'

}


quantiles = {
    'width': 'auto',
    'margin':'auto',
    'height': '50px',
    'overflow':'auto',
    'padding': '2px',
    'align':'center',
    'color':'white',
    'vertical-align':'bottom',
    'font-size':'18px',
    'text-align':'left',
    'border':'2px white solid',
    'font-family':'Arial'

}


grid = {
    'width': '1204px',
    'height': '392px',
    'padding': '2px',
    'backgroundColor':'green',
    'overflow-y':'auto',
    'overflow-x':'auto'
}

header =  {
    'width': '1204px',
    'height': '50px',
    'padding': '2px',
    'text-align':'center',
    'vertical-align':'middle',
    'backgroundColor':'blue',
    'font-family':'Arial',
    'color':'white',
    'font-size':'30px',
    'font-weight':'bold'
}

control_labels = {
    'font-family':'Arial',
    'color':'gray',
    'font-size':'18px',
    'font-weight':'bold',
    'text-align':'center',
    'vertical-align':'middle',
}
tab_labels = {
    'font-family':'Arial',
    'color':'blue',
    'font-size':'18px',
    'font-weight':'bold',
    'text-align':'center',
    'vertical-align':'middle',
    'background': colors['background'],
}

tabs_container = {
    'font-family':'Arial',
    'color':'blue',
    'font-size':'18px',
    'font-weight':'bold',
    'text-align':'center',
    'vertical-align':'middle',
    'background': colors['background'],
    'width' : '1208px'
}

import dash
from dash import dash_table, callback_context
#import dash_core_components as dcc
from dash import dcc
from dash import html
#import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import os
import time
import json
import ast
import pyproj
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State

import ray

cdc_data_original = pd.read_csv('cdc_2019_summary.csv', converters = {'fips': str})
# First pull a list of all unique fips codes
sorted_fips = cdc_data_original['fips']
sorted_fips = sorted_fips.unique()
sorted_fips.sort()

# Convert to Fractions for extimating quantiles
cdc_data_original['value'] = cdc_data_original['value']/ 100
# Invert the good ones

cdc_sample_table = cdc_data_original[['county','state','measure_id','value']]
conditions_list = [condition for condition in cdc_data_original['measure_id'].unique()]
conditions_dict = ["{'label': '" + j +"',"  + "'value': '" + j +"'}"  for j in conditions_list]
conditions_display = [ast.literal_eval(i) for i in conditions_dict]
states_list = [state for state in cdc_data_original['state'].unique()]
states_list.sort()
states_list_none = ['United States']
states_list_checked = states_list_none
states_dict = ["{'label': '" + j +"',"  + "'value': '" + j +"'}"  for j in states_list]
states_display = [ast.literal_eval(i) for i in states_dict]

cdc_data_original.loc['value'] = 1 - cdc_data_original[cdc_data_original['measure_id'] == "CHECKUP"]['value']
cdc_data_original.loc['value'] =(1 - cdc_data_original[cdc_data_original['measure_id'] == "MAMMOUSE"]['value'])
cdc_data_original.loc['value'] =(1 - cdc_data_original[cdc_data_original['measure_id'] == "COREM"]['value'])
cdc_data_original.loc['value'] =(1 - cdc_data_original[cdc_data_original['measure_id'] == "COREW"]['value'])
cdc_data_original.loc['value'] =(1 - cdc_data_original[cdc_data_original['measure_id'] == "CHOLSCREEN"]['value'])
cdc_data_original.loc['value'] =(1 - cdc_data_original[cdc_data_original['measure_id'] == "COLON_SCREEN"]['value'])
cdc_data_original.loc['value'] =(1 - cdc_data_original[cdc_data_original['measure_id'] == "DENTAL"]['value'])
cdc_data_original.loc['value'] =(1 - cdc_data_original[cdc_data_original['measure_id'] == "BPMED"]['value'])


geo_df = gpd.read_file('cb_2020_us_county_20m/cb_2020_us_county_20m.shp')
with open("cdc_data.geojson") as geofile:
    map_df = json.load(geofile)

# FLAG for which binning to use
use_quartile = True

quartile_breaks = [.25,.5,.75]
quintile_breaks = [.2,.4,.6,.8]

# DEFAULTS
rangecolor_list=(1,4)
tickvals_list=[1,2,3,4]
ticktext_list=['1','2','3','4']


@ray.remote
def calculate_by_condition(condition,in_df,use_quartile):
    global quartile_breaks
    global quintile_breaks
    ranked_df = in_df.copy(deep=True)
    ranked_df = ranked_df[ranked_df['measure_id'] == condition]
    #print(ranked_df)
    #clean_ranked_df = in_df.copy(deep=True)
    #return(ranked_df)
    #ranked_df = cdc_df.copy(deep=True)
    if use_quartile == True:
        x = ranked_df.loc[ranked_df['measure_id'] == condition]['value'].quantile(quartile_breaks)
        x.index = ['1','2','3']
        #print(f'Quartile breaks: {x["1"]} ,{x["2"]},{x["3"]}'  )
        for _ in ranked_df:
            ranked_df['Rank'] = np.where(
                    ((ranked_df['value'] < x['1']) & (ranked_df['value'] < 1) &
                    (ranked_df['measure_id'] == condition )),1,
                    np.where(
                    ((ranked_df['value'] >= x['1']) & (ranked_df['value'] < 1) &
                    (ranked_df['value'] < x['2']) & (ranked_df['measure_id'] == condition )),2,
                    np.where(
                    ( (ranked_df['value'] >= x['2']) & (ranked_df['value'] < 1) &
                    (ranked_df['value'] < x['3']) & (ranked_df['measure_id'] == condition )),3,
                    np.where(
                    ( (ranked_df['value'] >= x['3']) & (ranked_df['value'] < 1) &
                    (ranked_df['measure_id'] == condition )),4,ranked_df['value']))))
    else:
        x = ranked_df.loc[ranked_df['measure_id'] == condition]['value'].quantile(quintile_breaks)
        x.index = ['1','2','3','4']
        #print(f'Quintile breaks: {x["1"]} ,{x["2"]},{x["3"]},{x["4"]}'  )
        for _ in ranked_df:
            ranked_df['Rank'] = np.where(
                            ( (ranked_df['value'] < x['1']) & (ranked_df['value'] < 1) &
                            (ranked_df['measure_id'] == condition )),1,
                            np.where(
                            ((ranked_df['value'] >= x['1']) & (ranked_df['value'] < 1) &
                            (ranked_df['value'] < x['2']) & (ranked_df['measure_id'] == condition )),2,
                            np.where(
                            ((ranked_df['value'] >= x['2']) & (ranked_df['value'] < 1) &
                            (ranked_df['value'] < x['3']) & (ranked_df['measure_id'] == condition )),3,
                            np.where(
                            ((ranked_df['value'] >= x['3']) & (ranked_df['value'] < 1) &
                            (ranked_df['value'] < x['4']) & (ranked_df['measure_id'] == condition )),4,
                            np.where(
                            ((ranked_df['value'] >= x['4']) & (ranked_df['value'] < 1) &
                            (ranked_df['measure_id'] == condition )),5,ranked_df['value'])))))
    #print(ranked_df)
    return ranked_df

def calculate_county_rank(cdc_df,conditions=[],fips=[]):
    global sorted_fips
    ranked_df_par = cdc_df.copy(deep=True)

    result_ids = [calculate_by_condition.remote(conditions[i],ranked_df_par,use_quartile) for i in range(0,len(conditions))]
    results = ray.get(result_ids)
    conditions_checked = len(results)
    return_df = pd.DataFrame()
    for i in range(conditions_checked):
        return_df = return_df.append(results[i])
    #print(f'length of combined results {len(return_df)}')
    #print(results)
    shortened_ranked_df = return_df[['fips','measure_id','Rank']]
    pivoted_shortened_ranked_df = shortened_ranked_df.pivot(index='fips',columns='measure_id',values='Rank')
    just_locations = pd.DataFrame()
    # Adding the .loc keeps from getting a copy slice warning
    just_locations.loc[:,'fips'] = return_df.loc[:,'fips']
    just_locations.loc[:,'State'] = return_df.loc[:,'state']
    just_locations.loc[:,'County'] = return_df.loc[:,'county']
    just_locations.loc[:,'Population'] = return_df.loc[:,'population']
    #just_locations.loc[:,'value'] = ranked_df.loc[:,'value']
    ranked_df_par = just_locations.merge(pivoted_shortened_ranked_df,left_on='fips',right_on='fips',how='left')
    ranked_df_par.insert(4,'Rank',round(ranked_df_par[conditions].sum(axis=1)/len(conditions),1))
    ranked_df_par.drop_duplicates(subset='fips',inplace=True) # default keep = first
    #ranked_df_par['Rank'] = round(ranked_df_par[conditions].sum(axis=1)/len(conditions),1)
    #print(ranked_df_par)
    return(ranked_df_par)


def build_map(selectedData,conditions=[],states=[]):
    #print('build_map called')
    global cdc_data_original
    global geo_df
    global rangecolor_list
    global tickvals_list
    global ticktext_list
    if selectedData == None:
        #print("No Selected Data")
        #print(states_list)
        #cdc_data = cdc_data_original.loc[cdc_data_original['fips'] == '00059']
        cdc_data = cdc_data_original.loc[cdc_data_original['state'].isin(states_list)]
    else:
        cdc_data = cdc_data_original.loc[cdc_data_original['fips'].isin(point['location'] for point in selectedData['points'])]
        #for fips in cdc_data['fips']:
            #print(cdc_data['fips'])
    cdc_data = cdc_data.loc[cdc_data['measure_id'].isin(conditions)]
    cdc_data = cdc_data.loc[cdc_data['state'].isin(states)]
    fips = cdc_data['fips'].unique()
    returned_df = calculate_county_rank(cdc_data,conditions,fips)

    fig = px.choropleth(returned_df,
        geojson= map_df,
        featureidkey = 'properties.GEOID',
        locations='fips',
        color='Rank',
        range_color=rangecolor_list,
        color_continuous_scale="Viridis",
        scope="usa",
        labels={'Rank':'Rank'}, # TEMPORARY
        template='plotly_dark',
        height = 596,
        hover_data={'fips':False,'County':True,'State':True,'Rank':':%d'})

    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig.update_layout(coloraxis_colorbar=dict(
    thicknessmode="pixels", thickness=10,
    lenmode="pixels", len=200,
    yanchor="top", y=0.7,
    #ticks="outside", ticksuffix=" ",
    tickmode='array',
    tickvals=tickvals_list,
    ticktext= ticktext_list
    #dtick=1
    ))
    #fig.update_layout(legend_traceorder="grouped")
    #print('build_map completed')
    return fig,returned_df

app = dash.Dash()

app.layout = html.Div(children=[
    dcc.Tabs([
    dcc.Tab(label="2019 CDC Places Ranking by County",children=[
    html.Div(children=[
        html.Div(children=[
            html.Br(),
            html.Label('Quantile',style=control_labels),
             dcc.RadioItems(id='quantile',
                options=[
                    {'label':'Quartile','value':4},
                    {'label':'Quintile','value':5}
                ],labelStyle={'color':'white','display': 'block','vertical-align': 'center',},
                style=quantiles,value=4),
            html.Br(),
            html.Label('Select Condition(s)',style=control_labels),
            dcc.Checklist(id='conditions', options=conditions_display,
            #value=['ARTHRITIS'],
            value=['CSMOKING','DIABETES','MHLTH'],
            #value=['ARTHRITIS','OBESITY','DIABETES','STROKE','MAMMOUSE','CASTHMA','BINGE','CSMOKING'],
            #value=['ARTHRITIS','OBESITY','DIABETES','STROKE','MAMMOUSE','CASTHMA','BINGE','CSMOKING','LPA','SLEEP','DENTAL','PHLTH'],
            labelStyle = dict(display='block'),style=disease)]),
            html.Br(),
        html.Label('Select State(s)',style=control_labels),
        html.Div(children=[
        dcc.Checklist(id='states',
            options=states_display,
            value=states_list_checked
            ,labelStyle = dict(display='block'),style=state),
        html.Button('Update',id='update_button',n_clicks=1,style=button1),
        html.Button('None',id='none_button',n_clicks=0,style=button2),
        html.Button('All',id='all_button',n_clicks=0,style=button2)]
        )
    ],style=controls),
    html.Div(children=[
    dcc.Graph(id='indicator_map_chart')],
    style=map),
    html.Div(id='cdc_ranking_table',style=grid)

],style=tab_labels),
        dcc.Tab(label="Healthcare Disparitie Among Minorities\nComning SOON",style=tab_labels),
        dcc.Tab(label="Regional Income Dynamics using Spatial Markov Models\nComning SOON",style=tab_labels)
],style=tabs_container)]
)


@app.callback(Output('indicator_map_chart', 'figure'),
              Output('cdc_ranking_table','children'),
              Output('states','value'),
            [State('conditions','value'),
             State('states','value'),
             State('quantile','value'),
             Input('update_button','n_clicks'),
             Input('none_button','n_clicks'),
             Input('all_button','n_clicks'),
             State('indicator_map_chart','selectedData')])
def show_map(conditions,states,quantile,n_clicks,non_button,all_button,selectedData):
    global use_quartile
    global dash_table_headers
    global rangecolor_list
    global tickvals_list
    global ticktext_list
    states_list_checked = []
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'none_button' in changed_id:
        #print('none button pressed')
        states = states_list_none
        states_list_checked = states_list_none
    elif 'all_button' in changed_id:
        #print('all button pressed')
        states = states_list
        states_list_checked = states_list
    elif 'update_button' in changed_id:
        states_list_checked = states
    #print('Update requested')
    #print(states)
    if quantile == 4:
        use_quartile = True
        rangecolor_list=(1,4)
        tickvals_list=[1,2,3,4]
        ticktext_list=['1','2','3','4']
    else:
        use_quartile = False
        rangecolor_list=(1,5)
        tickvals_list=[1,2,3,4,5]
        ticktext_list=['1','2','3','4','5']
    if conditions == None:
        fig,returned_df = build_map(['ARTHRITIS'])
        data = returned_df.to_dict(orient='records')
        columns =  [{"name": i, "id": i,} for i in (returned_df.columns)]
        return fig,dash_table.DataTable(id='cdc_data_table',data=data, columns=columns,sort_action='native',fixed_rows={'headers':True},
                                                                        export_format='csv',
                                                                        filter_action="native",
                                                                        style_table={'overflowX': 'auto','maxHeight': '390',
                                                                        'overflowY': 'scroll'},
                                                                        style_header={'backgroundColor': "#111111",
                                                                                    'fontWeight': 'bold',
                                                                                    'font-size':'12px',
                                                                                    'textAlign': 'center',
                                                                                    'vertical-align':'bottom'},
                                                                        style_filter={
                                                                        'backgroundColor': 'WhiteSmoke'
                                                                        },
                                                                    style_cell={'backgroundColor': "#222222",
                                                                                'color':'lightgrey',
                                                                                'font-family':'Arial',
                                                                                'textAlign': 'center',
                                                                                'height': 'auto',
                                                                                'minWidth': '100px',
                                                                                'width': '100px', 'maxWidth': '100px',
                                                                                'whiteSpace': 'normal'},
                                                                        ),states_list_checked
    fig,returned_df = build_map(selectedData,conditions,states)
    data = returned_df.to_dict(orient='records')
    columns =  [{"name": i, "id": i,} for i in (returned_df.columns)]
    #print(time.asctime())
    return fig,dash_table.DataTable(id='cdc_data_table',data=data, columns=columns,sort_action='native',
                                                                    export_format='csv',
                                                                    fixed_rows={'headers':True},
                                                                    filter_action="native",
                                                                    style_table={'overflowX': 'auto','maxHeight': '390',
                                                                    'overflowY': 'scroll'},
                                                                    style_header={'backgroundColor': "#111111",
                                                                                'font-family':'Arial',
                                                                                'font-size':'12px',
                                                                                'fontWeight': 'bold',
                                                                                'textAlign': 'center',
                                                                                'vertical-align':'bottom'},
                                                                        style_filter={
                                                                        'backgroundColor': 'WhiteSmoke'
                                                                        },
                                                                    style_cell={'backgroundColor': "#222222",
                                                                                'color':'lightgrey',
                                                                                'font-family':'Arial',
                                                                                'textAlign': 'center',
                                                                                'height': 'auto',
                                                                                'minWidth': '100px',
                                                                                'width': '100px', 'maxWidth': '100px',
                                                                                'whiteSpace': 'normal'}
                                                                        ),states_list_checked


if __name__ == "__main__":
    #print(time.asctime())
    ray.init(num_cpus = 12)
    app.run_server()
