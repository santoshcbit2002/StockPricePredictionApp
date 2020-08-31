import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_gif_component as gif
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash_table
import pandas as pd 
import numpy as np
import datetime 
import time
import warnings
import os
import plotly.express as px
import plotly.graph_objs as go
import newsapi
import smtplib
from newsapi.newsapi_client import NewsApiClient
import pandas_market_calendars as mcal



warnings.filterwarnings("ignore")

###################################################################
# Define all Constants and Look up area here                      #
###################################################################

data_source_text='Disclaimer : The developers have no financial interest in any of the stocks.This is a research project with no intent for financial gains or advocating stock trade.'
data_source_text2='This is a research project with no intent for financial gains or advocating stock trade.'
data_refresh_text='Previous Business day trading numbers.Last Data refresh : ' + str(datetime.datetime.now().date()-datetime.timedelta(days=1))
title_text_1='Study Performance of Hybrid Deep Learning Algorithms'
title_text_2=' while Forecasting Random Time Series' 
copy_right_text1= "Built by: www.linkedin.com/in/santoshambaprasad/"
copy_right_text2= "          www.linkedin.com/in/bhargavikanchiraju/"

colors = {
    'background': '#1e1e1e',
    'plotbackground': '#000000',
    'borderline' : '#ffffff',
    'backgroundrecovered': '#000000',
    'backgroundconfirmed': '#000000',
    'textrateofraise' :'#ff7518',
    'green': '#138808',
    'textconfirmed': '#1e948a',
    'red': '#B20000',
    'amber': '#FFA500',
    'text': '#ff7518',
    'mapmarkers':'#B20000',
    'ledtext' : '#ff7518',
    'ledbackground':'#1f1f1f',
    'pie_colors' : ['#728C00','#018d61','#d40000'],
    'tabbackground':'#808080',
    'seltabbackground':'#303030'
}

###################################################################
# Load data into dataframe.                                       #
################################################################### 
print('*'*50)
print('Program has started  - {}'.format(datetime.datetime.now()))
print('Current Working Directory: {}'.format(os.getcwd()))

global data
data=pd.read_csv('./Data/model_data.csv')
table_data=pd.read_csv('./Data/table_data.csv')
latest_date=data.tail(1)['Date'].values[0]
print(latest_date)
#data=data[:-5][:]
#print(data.head(2))


###################################################################
# Prepare html layout components                                  #
################################################################### 

santosh_ln= html.Div( [ 
            dcc.Link('https://www.linkedin.com/in/santoshambaprasad/',href='https://www.linkedin.com/in/santoshambaprasad/',
             style={
                'textAlign': 'center', 
                'backgroundColor':colors['plotbackground'],
                'padding': '1px',
                'color': colors['text'],
                'font-size':'100%',
             #   'font-family':'Open Sans', 
                'font-weight': 'normal',
                'letter-spacing': 0.5,
                'margin-left': '3px'}     
            )
            ])

bhargavi_ln=html.Div( [ 
            dcc.Link('https://www.linkedin.com/in/bhargavikanchiraju/',href='https://www.linkedin.com/in/bhargavikanchiraju/',
            style={
                'textAlign': 'left', 
                'backgroundColor':colors['plotbackground'],
                'padding': '1px',
                'color': colors['text'],
                'font-size':'100%',
           #     'font-family':'Open Sans', 
                'font-weight': 'normal',
                'letter-spacing': 0.5,
                'margin-left': '3px'}     
                ) 
            ]
            )

dashboard_title=html.H1(
            style={
            'textAlign': 'center', 
            'backgroundColor':colors['plotbackground'],
            'padding': '12px',
            'color': colors['text'],
            'font-size':'250%',
         #   'font-family':'Open Sans', 
            'font-weight': 'normal',
            'letter-spacing': 1.0,
            'margin': 0},
            children=title_text_1
            )

dashboard_source_text=html.Div( 
            children=[
            html.H3(
                    children=data_source_text,
                     style={
                        'textAlign': 'center', 
                        'backgroundColor':colors['plotbackground'],
                        'color': colors['text'],
                        'font-size':'100%',
                 #       'font-family':'Open Sans', 
                        'font-weight': 'normal',
                        'letter-spacing': 0.2
                        },
                    )   
            ])

dashboard_refresh_text=html.Div( style={},
            children=[
            html.H3(
                    children=data_refresh_text,
                     style={
                        'textAlign': 'left', 
                        'margin-left': '50px',
                        'backgroundColor':colors['plotbackground'],
                        'color': colors['text'],
                        'font-size':'100%',
                  #      'font-family':'Open Sans', 
                        'font-weight': 'normal',
                        'letter-spacing': 0.2
                        },
                    )   
            ])
###################################################################
# Prepare Date for Filters                                        #
################################################################### 

start_date = html.Div([
    dcc.DatePickerRange(
        id='my-date-picker-range',
        start_date_placeholder_text='Start Date',
        end_date_placeholder_text='End Date',
        calendar_orientation='vertical',
            ),
            html.Div(id='output-container-date-picker-range')
        ],
         style={
                        'textAlign': 'left',
                         'margin-left':'30px',
                        'backgroundColor':colors['plotbackground'],
                        'color': colors['text'],
                       # 'border': '5px solid orange',
                        'font-size':'100%',
                       # 'font-family':'Open Sans', 
                        'font-weight': 'normal',
                        'letter-spacing': 0.2
                        }
    
   )

###################################################################
# Filters                                                         #
################################################################### 

pred_selection= html.Div([
        html.Div(id='prediction_selection_text'),
        dcc.RadioItems(id='prediction_selection', 
        options=[
             {'label': ' Short Term Prediction - 1 day  ', 'value': 1},
             {'label': ' Medium Term Prediction - 3 days', 'value': 2,'disabled': True},
             {'label': ' Long Term Prediction - 5 days', 'value': 3,'disabled': True},
             {'label': ' Predictions made on test sets during development','value': 4,'disabled': False},
             {'label': ' Reset','value':0}
             ])
             ],style={ 'textAlign': 'left',
                         'margin-left':'60px',
                        'backgroundColor':colors['plotbackground'],
                        'color': colors['text'],
                        'letter-spacing': 0.2 } )

###################################################################
# Prepare Data for Predictions                                    #
################################################################### 


LSTM_chart= html.Div([
                    dcc.Graph(
                        id='lstm_pred_chart',
                        figure={
                                'data': [
                                    go.Scatter(x=data['Date'],
                                               y=data['Lloyds_Close'],
                                               hovertemplate =
                                                '<i>Price</i>: %{y:.2f}'+
                                                '<br><b>Date</b>: %{x}<br>',         
                                                name='LSTM',stackgroup='one',
                                                mode='lines')
                                       ],
                                'layout': {
                                    'title':{ 'text':'Lloyds Banking Group Stock Close Price'},
                                    'plot_bgcolor': colors['plotbackground'],
                                    'paper_bgcolor': colors['plotbackground'],
                                    'font': {'color': colors['text']},
                                    'margin': '0',
                                        'padding': '20px',   
                                        'style': {  'width': '120%',
                                                    'height': '40px',
                                                    'border': '5px solid orange',
                                                    'borderWidth': '5px',
                                                    'borderStyle': 'solid',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center' 
                                               }           
                                       }
                                }
                            )  
                    ],
                   style={'backgroundColor': colors['plotbackground']
                        }  
                )

###################################################################
# Prepare Data Stock Indicators                                  #
################################################################### 

# All constants

current_day=table_data[table_data['day']==1]
previous_day=table_data[table_data['day']==0]

LLOY_high=current_day['LLOY_high']
LLOY_open=current_day['LLOY_open']
LLOY_close=current_day['LLOY_close']
LLOY_low=current_day['LLOY_low']

BARC_high=current_day['BARC_high']
BARC_open=current_day['BARC_open']
BARC_close=current_day['BARC_close']
BARC_low=current_day['BARC_low']

RBS_high=current_day['RBS_high']
RBS_open=current_day['RBS_open']
RBS_close=current_day['RBS_close']
RBS_low=current_day['RBS_low']

#OIL_high=current_day['OIL_high']
#OIL_open=current_day['OIL_open']
OIL_close=current_day['Oil']
#OIL_low=current_day['OIL_low']

#USD_high=current_day['USD_high']
#USD_open=current_day['USD_open']
USD_close=current_day['USDGBP']
#USD_low=current_day['USD_low']

FTSE_high=current_day['FTSE_high']
FTSE_open=current_day['FTSE_open']
FTSE_close=current_day['FTSE_close']
FTSE_low=current_day['FTSE_low']


if previous_day['LLOY_close'].values == current_day['LLOY_close'].values:
    LLOY_style={'backgroundColor':colors['plotbackground']}
else:
    if previous_day['LLOY_close'].values > current_day['LLOY_close'].values:
         LLOY_style={'backgroundColor':colors['red']}    
    else:
          LLOY_style={'backgroundColor':colors['green']} 

if previous_day['BARC_close'].values == current_day['BARC_close'].values:
    BARC_style={'backgroundColor':colors['plotbackground']}
else:
    if previous_day['BARC_close'].values > current_day['BARC_close'].values:
        BARC_style={'backgroundColor':colors['red']}    
    else:
          BARC_style={'backgroundColor':colors['green']} 

if previous_day['RBS_close'].values == current_day['RBS_close'].values:
    RBS_style={'backgroundColor':colors['plotbackground']}
else:
    if previous_day['RBS_close'].values > current_day['RBS_close'].values:
         RBS_style={'backgroundColor':colors['red']}    
    else:
          RBS_style={'backgroundColor':colors['green']} 

if previous_day['FTSE_close'].values == current_day['FTSE_close'].values:
    FTSE_style={'backgroundColor':colors['plotbackground']}
else:
    if previous_day['FTSE_close'].values > current_day['FTSE_close'].values:
         FTSE_style={'backgroundColor':colors['red']}    
    else:
          FTSE_style={'backgroundColor':colors['green']} 



table_header = [
    html.Thead(html.Tr([html.Th("Indicator"), html.Th("Name"),html.Th("Open"),html.Th("High"),html.Th("Low"),html.Th("Prev. Close")]))
]


row1 = html.Tr(
    [html.Td("LLOY"),
    html.Td("Lloyds Banking Group Plc"), 
    html.Td(LLOY_open), 
    html.Td(LLOY_high),
    html.Td(LLOY_low),
    html.Td(LLOY_close,style=LLOY_style)
    ])

row2 = html.Tr([
    html.Td("RBS"), 
    html.Td("Royal Bank of Scotland Plc"), 
    html.Td(RBS_open), 
    html.Td(RBS_high),
    html.Td(RBS_low),
    html.Td(RBS_close,style=RBS_style)
    ])

row3 = html.Tr([
    html.Td("BARC"), 
    html.Td("Barclays Bank Plc"), 
    html.Td(BARC_open), 
    html.Td(BARC_high),
    html.Td(BARC_low),
    html.Td(BARC_close,style=BARC_style)
    ])

row4 = html.Tr([
    html.Td("USD"), 
    html.Td("US Dollor/GBP"), 
    html.Td(" - "), 
    html.Td(" - "),
    html.Td(" - "),
    html.Td(USD_close)
    ])

row5 = html.Tr([
    html.Td("Oil"),
    html.Td("Oil"), 
    html.Td(" - "), 
    html.Td(" - "),
    html.Td(" - "),
    html.Td(OIL_close)
    ])

row6 = html.Tr([
    html.Td("FTSE"),
    html.Td("FTSE Index"), 
    html.Td(FTSE_open), 
    html.Td(FTSE_high),
    html.Td(FTSE_low),
    html.Td(FTSE_close,style=FTSE_style)
    ])

table_body = [html.Tbody([row1, row2, row3, row4, row5, row6])]

table = dbc.Table(
    table_header + table_body,
    bordered=False,
    dark=True,
    hover=True,
    responsive=True,
    striped=True,
    size='md',
    style={
          #  'margin-left':'2px',
          #  'font-size':'105%',
          #   'font-family':'Open Sans', 
             'font-weight': 'normal',
             'backgroundColor':colors['plotbackground'],
             'color': colors['text']
            }
)


###################################################################
#       Prepare data for News                                     #
###################################################################
newsapi_obj = NewsApiClient(api_key='e233bbffef0a42279c871476d6772e8f')
headline_string=['lloyds','barclays','rbs','oil','usd']
source_name=[]
url_text=[]
new_title=[]
published=[]

for search_string in headline_string: 
    top_headlines = newsapi_obj.get_everything(q=search_string,
                            sources='bloomberg,financial-post,fortune',
                            from_param='2020-08-27',
                            to='2020-08-27',
                            sort_by='relevancy',
                            # sources='bbc-news',
                                          language='en') 
    #print('===========')
    #print('Search string is {}'.format(search_string)   )
    #print(top_headlines)

    for i in range(0,top_headlines['totalResults']):
        source_name.append(top_headlines['articles'][i]['source']['name'])
        new_title.append(top_headlines['articles'][i]['title'])
        url_text.append(top_headlines['articles'][i]['url'])  
        published.append(top_headlines['articles'][i]['publishedAt'])

news_item=[]

for i in range(0,len(new_title)):
    news_item.append(dbc.ListGroupItem(new_title[i],href=url_text[i],color=colors['text']))

dashboard_news_table=html.Div(
                             dbc.Card([
                                        dbc.Alert('Top Headlines',color='warning'),
                                        dbc.ListGroup(news_item, flush=True)
                                        ],
                            outline=True), style={'width':'90%',
                                                   'textAlign':'left',
                                                'margin-left':'10px',
                                                'margin-right':'50px',
                                                'backgroundColor':colors['plotbackground'],
                                                },
)

################################################################### 
# Prepare feedback layout                                         #
################################################################### 

def sendemail(from_addr, to_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587', cc_addr_list=[]):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message
 
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems

def triggeremail(message_txt):    
     sendemail(from_addr    = 'dashboardstest2020@gmail.com', 
          to_addr_list = ['bhargavi.sivapurapu@gmail.com','santosh.cbit2002@gmail.com'],
          subject      = 'StockMarket DashBoard User Feedback', 
          message      = message_txt, 
          login        = 'dashboardstest2020@gmail.com', 
          password     = 'offbduwbzbrizdrg') 
    


feedback_def_value='Please leave feedback here. Please leave your contact, if you wish to be contacted. Thank you :-)'


dashboard_feedback_area = dbc.Container([ dbc.FormGroup([  html.Br(),
                                                           dbc.Input(id="text-area", placeholder='', type="text",bs_size=5),
                                                           dbc.FormText(feedback_def_value,color='warning',
                                                                        style={ 'font-size':'120%',
                                                                                'font-family':'Open Sans', 
                                                                                'font-weight': 'normal'
                                                                        }),
                                                           html.Br(),
                                                           dbc.Button("Submit", color="primary", id='submit-button',className="mr-2"),
                                                           html.P(id="output",style={'color':'#ff7518'})
                                                            ])
                                            ])
dashboard_copyright_stmt= html.Div(
                        dcc.Markdown(
                        '''
                         News headlines powered by NewsAPI.org
                        '''
                        ),
                        style={
                        'textAlign': 'center', 
                        'backgroundColor':colors['plotbackground'],
                        'color': colors['text'],
                        'font-size':'100%',
                        'margin-top':'20px',
                        'font-family':'Open Sans', 
                        'font-weight': 'normal'
                       
                        },
                                    )

##############################################################################
#                               Create DASH App instance                     #
##############################################################################

app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])

server=app.server

##############################################################################
#                               HTML Layout                                  #
##############################################################################

app.layout = html.Div(style={'backgroundColor': colors['plotbackground'] },

    
    children=[
        dbc.Row(dbc.Col(dashboard_source_text)),    
        dbc.Row(
            [dbc.Col(santosh_ln,style={'textAlign': 'left'}),
            dbc.Col(bhargavi_ln,style={'textAlign': 'right'})]),
        html.Br(),
        dbc.Row(dbc.Col(html.H1(title_text_1+title_text_2,style={'color':colors['text'],'textAlign':'center'}),width=12,align='center')),  
        html.Hr(style={'margin-left':'50px','margin-right':'50px'}),
        html.Br(),
        dbc.Row(dbc.Col(dashboard_refresh_text)),  
        
        dbc.Row([
            dbc.Col(
                    html.Div(
                         children=table,style={
                                                      'width':'100%',
                                                     'margin-left':'10px'
                                                     }          
                            )),
                 dbc.Col(
                     html.Div(
                         children=dashboard_news_table,style={
                                                      'width':'100%',
                                                      'textAlign':'left',
                                                     'margin-right':'10px'
                                                     }
                 )),            
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(pred_selection,align='center',width=6),
            dbc.Col(start_date,width='6'),
            ]),
        html.Br(),
        dbc.Row(dbc.Col(LSTM_chart,style={'margin-right':'20px','margin-left':'0px'})),
        dashboard_feedback_area,
        dashboard_copyright_stmt
    ])


# Call Back function for date range picker help
@app.callback(
    dash.dependencies.Output('output-container-date-picker-range', 'children'),
    [dash.dependencies.Input('my-date-picker-range', 'start_date'),
     dash.dependencies.Input('my-date-picker-range', 'end_date')])
def update_output(start_date, end_date):
       if start_date==None:
             start_date='2009-01-01'
       if end_date==None:
            end_date=datetime.datetime.now().strftime('%d-%m-%Y')
       
       return 'Select a date range. Default is from 2009/01/01 till today'

# Call Back function for LSTM Chart
@app.callback(
    dash.dependencies.Output('lstm_pred_chart', 'figure'),
    [dash.dependencies.Input('my-date-picker-range', 'start_date'),
     dash.dependencies.Input('my-date-picker-range', 'end_date'),
     dash.dependencies.Input('prediction_selection', 'value')
     ])
def update_output(start_date, end_date,value):
    print('value',value)

    if value==5:
        print('prediction selection Value in 5  ')
        start_date=None
        end_date=None

    if start_date==None:
        start_date=datetime.datetime.strptime('2009-01-01','%Y-%m-%d').strftime('%Y-%m-%d')
    if end_date==None:
        #end_date=datetime.datetime.now().strftime('%Y-%m-%d')
        end_date=latest_date
    
    #print('prediction selection Value is : ',value)
    fig_data=[]
    #print('selected start date: ',start_date)
    #print('selected end date: ',end_date)
    data_index=data.set_index('Date')
    temp=data_index[start_date:end_date].reset_index()
    #print(temp.head(2))

    fig_data.append(go.Scatter(x=temp['Date'],
                            y=temp['Lloyds_Close'],
                            hovertemplate =
                                '<i>Price</i>: %{y:.2f}'+
                                '<br><b>Date</b>: %{x}<br>',         
                                name='Actual',
                               # stackgroup='one',
                                mode='lines')
                            
                            )

    if value==1:
        print('prediction selection Value in 1 ')
       # LSE = mcal.get_calendar('LSE')
       # a=LSE.valid_days(start_date=latest_date, end_date='2020-12-31')[2]
       # print(a)
       # pred_dates=[]

       # for i in range(0,2):
       #     pred_dates.append(a[i].date().strftime('%Y-%m-%d'))

        pred=np.load('./Data/predictions_dump.npy')
        dates=data.tail(11)['Date'].values
        pred_frame=pd.DataFrame({'Date':dates,
                                 'Lloyds_Close':pred})
        print(pred_frame)
        pred_frame_index=pred_frame.set_index('Date')
        temp=pred_frame_index[start_date:end_date].reset_index()
        fig_data.append(go.Scatter(x=temp['Date'],
                            y=temp['Lloyds_Close'],
                            hovertemplate =
                                '<i>Price</i>: %{y:.2f}'+
                                '<br><b>Date</b>: %{x}<br>',         
                                name='Predictions',
                                #stackgroup='two',
                                mode='lines')
                            
                            )
    if value==4:
        print('prediction selection Value in 4 ')
        LSE = mcal.get_calendar('LSE')
        a=LSE.valid_days(start_date='2019-07-05 ', end_date='2020-12-31')[:200]
        pred_dates=[]

        for i in range(0,200):
            pred_dates.append(a[i].date().strftime('%Y-%m-%d'))

        pred=np.load('./Data/development_predictions.npy')
        pred=pred[:200]
        
        pred_frame=pd.DataFrame({'Date':pred_dates[:len(pred)],
                    'Lloyds_Close':pred})
        #print(pred_frame.head(5))

        fig_data.append(go.Scatter(x=pred_frame['Date'],
                            y=pred_frame['Lloyds_Close'],
                            hovertemplate =
                                '<i>Price</i>: %{y:.2f}'+
                                '<br><b>Date</b>: %{x}<br>',         
                                name='Predictions',
                               # stackgroup='two',
                                mode='lines')
                            
                            )
    
    return {
             'data': fig_data,
            'layout':   {
                                    'title':{ 'text':'Lloyds Banking Group Stock Close Price'},
                                    'plot_bgcolor': colors['plotbackground'],
                                    'paper_bgcolor': colors['plotbackground'],
                                    'font': {'color': colors['text']},
                                    'font-size':'120%',
                                    #'xaxis': { 'gridcolor': "#636363", "showline": False},
                                    'margin': '0',
                                        'padding': '20px',   
                                        'style': {  'width': '100%',
                                                    'height': '40px',
                                                    'border': '5px solid orange',
                                                    'borderWidth': '5px',
                                                    'borderStyle': 'solid',
                                                    'borderRadius': '5px',
                                                    'textAlign': 'center' 
                                               }           
                                       }
                }   

# Call Back function for feedback text
@app.callback(
    [Output('output', 'children'),Output('text-area','value')],
    [Input('submit-button', 'n_clicks')],
    [State('text-area', 'value')]
)
def update_output(button_clicks,value):
        if button_clicks is None:
             button_clicks=0
        
        if button_clicks > 0:
           triggeremail(value)
           return 'Thank you for the feed back',''
        else:
            return '',''


if __name__ == '__main__':
    app.run_server(debug=True)  