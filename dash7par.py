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
    'width': '898px',
    'height': '596px',
    'padding': '2px',
    'backgroundColor': "black",
    'align':'right',
    'vertical-align':'top',
    'display':'inline-block',
    'border':'2px green solid'

}

map_plot = {
    'width': '894px',
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
    'height': '170px',
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
    'width': '97px',
    'margin-top':'2px',
    'height': '26px',
    'padding': '1px',
    'align':'center',
    'background': 'black',
    'color':'white',
    'vertical-align':'bottom',
    'font-size':'12px',
    'text-align':'center',
    'border':'1px white solid',
    'font-family':'Arial',
    'font-weight':'normal'

}
button2 = {
    'width': '97px',
    'margin-top':'2px',
    'height': '26px',
    'padding': '1px',
    'align':'center',
    'background': 'black',
    'color':'white',
    'vertical-align':'bottom',
    'font-size':'12px',
    'text-align':'center',
    'border':'1px white solid',
    'font-family':'Arial',
    'font-weight':'normal'

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
    'font-size':'14px',
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
    'font-size':'16px',
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

popover_container = {
    'maxHeight': '700px', 
    'overflowY': 'auto', 
    'width': '800px',
    'whiteSpace': 'pre-wrap',
    'wordWrap': 'break-word',
    'wordBreak': 'normal',
    'padding': '10px',
    'boxSizing': 'border-box',
    'display': 'block'
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
import dash_bootstrap_components as dbc  # Make sure this import is here

import ray

import openai
from openai import OpenAI
import base64

# HTTP Authentication


cdc_data_original = pd.read_csv('cdc_2021_summarya.csv', converters = {'fips': str})
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

# Declare the global variables
disease_list = []
full_table = pd.DataFrame()


geo_df = gpd.read_file('cb_2022_us_county_20m/cb_2022_us_county_20m.shp')
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

### GET DATA ABOUT THE MAP
def call_openai_1():
    # Your OpenAI function logic here
    # Save the figure as a PNG file
    print("Function call_openai_1 executed")
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY') )

    conditions_string = ", ".join(disease_list)
    print(conditions_string)
    result_string = f"Provide a brief three sentence description of the condition {conditions_string} .\
                 For the condition {conditions_string} and the provided image file, write a detailed description of the regional concerns\
                    related to the values for {conditions_string} as depicted by the heatmap. Provide any additional insights about other\
                    factors such as environmental or social determinants of health that could affect areas with higher rates of {conditions_string}."
    print(result_string)
    # Read the content of the file
    import base64

    MODEL="gpt-4o"

    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "cdc_map_fig.png")
    #IMAGE_PATH = "image_path"

    # Open the image file and encode it as a base64 string
    def encode_image(file_path):
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(file_path)


    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in Markdown."},
            {"role": "user", "content": [
                {"type": "text", "text": result_string},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    #print(f"Full response: {response}")  # Print the entire response to understand its structure
    #return response['choices'][0]['message']['content']
    message_content  = response.choices[0].message.content
    return message_content

def call_openai_2(prompt):
    # Function logic to interact with OpenAI API
    conditions_string = ", ".join(disease_list)  # Get the current disease

    print("Function call_openai_2 executed with prompt:", prompt)
    
    openai.api_key = os.getenv('OPENAI_API_KEY')
    MODEL = "gpt-4o"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "cdc_map_fig.png")

    # Open the image file and encode it as a base64 string
    def encode_image(file_path):
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(file_path)
    
    #result_string = f"{prompt}"
    result_string = f"Given the heat map shows the incidence of {conditions_string}, \
        preface the response with a few sentences about {conditions_string}. {prompt}"

    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in Markdown."},
            {"role": "user", "content": [
                {"type": "text", "text": result_string},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    
    # Correctly access the response content
    message_content  = response.choices[0].message.content
    return message_content

### OPENAI 3 - Pass data frame to OpenAI
def call_openai_3(prompt, dataframe):
    conditions_string = ", ".join(disease_list)  # Get the current disease
    # Function logic to interact with OpenAI API
    print("Function call_openai_3 executed with prompt:", prompt)
    
    openai.api_key = os.getenv('OPENAI_API_KEY')
    MODEL = "gpt-4o"
    
    # Convert DataFrame to JSON string
    dataframe_json = dataframe.to_json(orient='split')
    
    result_string = f"Given the data frame shows the incidence of {conditions_string}, \
        preface the response with a few sentences about {conditions_string}. {prompt} \
        \nHere is the data:\n{dataframe_json}"
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds in Markdown."},
            {"role": "user", "content": result_string}
        ],
        temperature=0.0,
    )
    
    # Correctly access the response content
    message_content = response.choices[0].message.content
    return message_content



def calculate_county_rank(cdc_df, conditions=[], fips=[]):
    global sorted_fips
    global disease_list
    global full_table
    ranked_df_par = cdc_df.copy(deep=True)


    result_ids = [calculate_by_condition.remote(conditions[i], ranked_df_par, use_quartile) for i in range(len(conditions))]
    results = ray.get(result_ids)
    conditions_checked = len(results)
    # SAVE CONDITIONS TO GLOBAL VARIABLE
    if conditions_checked > 0:
        disease_list = conditions
    return_df = pd.DataFrame()
    
    # Use pd.concat with the correct syntax for the list comprehension
    return_df = pd.concat([pd.DataFrame(results[i]) for i in range(conditions_checked)], ignore_index=True)
    
    #print(f'length of combined results {len(return_df)}')
    #print(results)
    shortened_ranked_df = return_df[['fips', 'measure_id', 'Rank']]
    pivoted_shortened_ranked_df = shortened_ranked_df.pivot(index='fips', columns='measure_id', values='Rank')
    just_locations = pd.DataFrame()
    
    # Adding the .loc keeps from getting a copy slice warning
    just_locations.loc[:, 'fips'] = return_df.loc[:, 'fips']
    just_locations.loc[:, 'State'] = return_df.loc[:, 'state']
    just_locations.loc[:, 'County'] = return_df.loc[:, 'county']
    just_locations.loc[:, 'Population'] = return_df.loc[:, 'population']
    #just_locations.loc[:, 'value'] = ranked_df.loc[:, 'value']
    
    ranked_df_par = just_locations.merge(pivoted_shortened_ranked_df, left_on='fips', right_on='fips', how='left')
    ranked_df_par.insert(4, 'Rank', round(ranked_df_par[conditions].sum(axis=1) / len(conditions), 1))
    ranked_df_par.drop_duplicates(subset='fips', inplace=True)  # default keep = first
    # SAVE THE TABLE GLOBALLY FOR OPENAI
    full_table = ranked_df_par.drop(columns=['fips'])
    print(full_table.head())
    #ranked_df_par['Rank'] = round(ranked_df_par[conditions].sum(axis=1) / len(conditions), 1)
    #print(ranked_df_par)
    return ranked_df_par


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
        color_continuous_scale="Spectral_r",
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

#app = dash.Dash()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])



app.layout = html.Div(children=[
    dcc.Tabs([
    dcc.Tab(label="2021 CDC Places Ranking by County (2024 Release, FL=2019 data)",children=[
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
            value=['COPD'],
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
        html.Button('All States',id='all_button',n_clicks=0,style=button2),
        html.Button('Auto AI',id='ai_button',n_clicks=0,style=button2),
        html.Button('Map AI', id='ai_button2', n_clicks=0, style=button2),
        html.Button('Table AI', id='ai_button3', n_clicks=0, style=button2),
        #dcc.Store(id='ai_button_click_store', data={'last_n_clicks': 0}),
        dbc.Popover(
            [
                dbc.PopoverHeader("AI Response"),
                dbc.PopoverBody([
                    html.Div(id='popover-content', style=popover_container),
                    html.Button('Close', id='close-popover', n_clicks=0, className='mt-2')
                ])
            ],
            id='ai-popover',
            target='ai_button',
            placement='right',
            is_open=False
        ),

        # New Popover for the second AI button
        dbc.Popover(
            [
                dbc.PopoverHeader("AI Response"),
                dbc.PopoverBody([
                    html.Div(id='popover-content2', style=popover_container),
                    html.Button('Close', id='close-popover2', n_clicks=0, className='mt-2')
                ])
            ],
            id='ai-popover2',
            target='ai_button2',
            placement='right',
            is_open=False
        ),

        # Modal for user input
        dbc.Modal(
            [
                dbc.ModalHeader("Enter your prompt"),
                dbc.ModalBody(dcc.Textarea(id='user-prompt', style={'width': '100%', 'height': '200px'})),
                dbc.ModalFooter(
                    dbc.Button('Submit', id='submit-prompt', n_clicks=0)
                ),
            ],
            id='prompt-modal',
            is_open=False,
        ),
        # New Popover for the third AI button (Table AI)
        dbc.Popover(
            [
                dbc.PopoverHeader("AI Response"),
                dbc.PopoverBody([
                    html.Div(id='popover-content3', style=popover_container),
                    html.Button('Close', id='close-popover3', n_clicks=0, className='mt-2')
                ])
            ],
            id='ai-popover3',
            target='ai_button3',
            placement='right',
            is_open=False
        ),

        # Modal for user input (for Table AI)
        dbc.Modal(
            [
                dbc.ModalHeader("Enter your prompt"),
                dbc.ModalBody(dcc.Textarea(id='user-prompt3', style={'width': '100%', 'height': '200px'})),
                dbc.ModalFooter(
                    dbc.Button('Submit', id='submit-prompt3', n_clicks=0)
                ),
            ],
            id='prompt-modal3',
            is_open=False,
        )
    ])
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
        fig.write_image("cdc_map_fig.png")
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
    fig.write_image("cdc_map_fig.png")
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


### CALLBACK FOR AI BUTTON
""" @app.callback(
    Output('ai_button_click_store', 'data'),
    Input('ai_button', 'n_clicks'),
    State('ai_button_click_store', 'data')
)
def execute_ai_function(n_clicks, store_data):
    if n_clicks > store_data['last_n_clicks']:
        call_openai_1()
        store_data['last_n_clicks'] = n_clicks
    return store_data
 """

# Define the callback to show or hide the popover
@app.callback(
    [Output('ai-popover', 'is_open'),
     Output('popover-content', 'children')],
    [Input('ai_button', 'n_clicks'),
     Input('close-popover', 'n_clicks')],
    [State('ai-popover', 'is_open')]
)
def toggle_popover(ai_n_clicks, close_n_clicks, is_open):
    ctx = dash.callback_context

    if not ctx.triggered:
        return is_open, dash.no_update

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'ai_button':
        if ai_n_clicks > 0:
            # Call the OpenAI function and get the response
            ai_response = call_openai_1()
            #print(f"AI response: {ai_response}")

            return True, ai_response
    
    elif trigger == 'close-popover':
        return False, dash.no_update

    return is_open, dash.no_update

@app.callback(
    Output('prompt-modal', 'is_open'),
    [Input('ai_button2', 'n_clicks'),
     Input('submit-prompt', 'n_clicks')],
    [State('prompt-modal', 'is_open')]
)
def toggle_modal(ai_n_clicks2, submit_n_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Modal triggered by: {trigger}")

    if trigger == 'ai_button2' and ai_n_clicks2 > 0:
        return True
    elif trigger == 'submit-prompt' and submit_n_clicks > 0:
        return False
    return is_open

@app.callback(
    [Output('ai-popover2', 'is_open'),
     Output('popover-content2', 'children')],
    [Input('submit-prompt', 'n_clicks'),
     Input('close-popover2', 'n_clicks')],
    [State('user-prompt', 'value'),
     State('ai-popover2', 'is_open')]
)
def handle_user_prompt(submit_n_clicks, close_n_clicks, user_prompt, is_open):
    ctx = dash.callback_context

    if not ctx.triggered:
        print("Popover handle not triggered")
        return is_open, dash.no_update

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Popover triggered by: {trigger}")

    if trigger == 'submit-prompt':
        if submit_n_clicks > 0:
            print(f"User prompt: {user_prompt}")
            if user_prompt:
                ai_response = call_openai_2(user_prompt)
                print(f"AI response: {ai_response}")
                return True, ai_response  # Return just the response content
    
    elif trigger == 'close-popover2':
        print("Close popover2 triggered")
        return False, dash.no_update

    return is_open, dash.no_update


@app.callback(
    Output('prompt-modal3', 'is_open'),
    [Input('ai_button3', 'n_clicks'),
     Input('submit-prompt3', 'n_clicks')],
    [State('prompt-modal3', 'is_open')]
)
def toggle_modal3(ai_n_clicks3, submit_n_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        print("Modal toggle not triggered")
        return is_open
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Modal triggered by: {trigger}")

    if trigger == 'ai_button3' and ai_n_clicks3 > 0:
        return True
    elif trigger == 'submit-prompt3' and submit_n_clicks > 0:
        return False
    return is_open

@app.callback(
    [Output('ai-popover3', 'is_open'),
     Output('popover-content3', 'children')],
    [Input('submit-prompt3', 'n_clicks'),
     Input('close-popover3', 'n_clicks')],
    [State('user-prompt3', 'value'),
     State('ai-popover3', 'is_open')]
)
def handle_user_prompt3(submit_n_clicks, close_n_clicks, user_prompt, is_open):
    ctx = dash.callback_context

    if not ctx.triggered:
        print("Popover handle not triggered")
        return is_open, dash.no_update

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Popover triggered by: {trigger}")

    if trigger == 'submit-prompt3':
        if submit_n_clicks > 0:
            print(f"User prompt: {user_prompt}")
            if user_prompt:
                ai_response = call_openai_3(user_prompt, full_table)
                print(f"AI response: {ai_response}")
                return True, html.Div([
                    html.Div(ai_response, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'Courier, "Courier New", monospace'}),
                    html.Button('Close', id='close-popover3', n_clicks=0, className='mt-2')
                ])
    
    elif trigger == 'close-popover3':
        print("Close popover3 triggered")
        return False, dash.no_update

    return is_open, dash.no_update


if __name__ == "__main__":
    #print(time.asctime())
    ray.init(num_cpus = 10)
    app.run_server(debug=False, port=8051)  # Change the port number as needed
    #app.run_server()

