import base64
import datetime
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    "style.css",
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

######### Parameters you can change #########
colors = {
    'text': '#003366',
    'text-header': '#ffffff',
    'text-attention': '#cc0000',
    'background-header': '#001a33',
    'markers': 'rgb(0, 153, 255)',
    'border': '#001a33',
    'text-size': '150%'
}

image_src = 'https://chanzuckerberg.com/wp-content/uploads/2019/11/dash-logo.jpg'

markdown_text = '''
# DESCRIPTIVE ANALYSIS

This is just a demo. You need to upload a csv file to be able to test it.
'''


######### Parameters to initialize info #########

df = pd.DataFrame()
available_indicators = np.array(['NO DATA'])


######### Dash App Layout and Components #########

app.layout = html.Div([

    html.Div([
        html.Div([
            html.Img(src=image_src,
                width='200', height='80'),
        ], style={'width': '10%', 'float': 'right', 'display': 'inline-block', 'margin': '20px'}),        

        html.Div([  
            dcc.Markdown(children=markdown_text,
                style={
                    'textAlign': 'left',
                    'color': colors['text-header']
                    }),
        ], style={'width': '80%', 'float': 'left', 'display': 'inline-block', 'margin': '20px'}), 
    ], style={'backgroundColor': colors['background-header'], 'columnCount': 1}),

    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '100px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin-top': '30px',
                'lineColor': colors['border'],
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
    ]),

    html.Div([
        html.Hr(style={'width': '100%', 'display': 'block'}), # horizontal line

        html.Div([ 
            dcc.Markdown(children='''Variables Selection''',
                style={
                    'textAlign': 'left',
                    'color': colors['text']
                    }),

            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=available_indicators[0]
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '15%', 'float': 'left', 'display': 'inline-block', 'margin': '10px', 'margin-bottom': '30px'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=available_indicators[0]
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '15%', 'float': 'center', 'display': 'inline-block', 'margin': '10px', 'margin-bottom': '30px', 'margin-top': '40px'}),

        html.Div([
            dcc.Markdown(children='''Date Filter''',
                style={
                    'textAlign': 'left',
                    'color': colors['text']
                    }),

            dcc.RangeSlider(
            id='date--slider',
            min=0,
            max=10,
            value=[0, 10],
            marks={i: '{}'.format(i) for i in range(1, 11)}
            ),
        ], style={'width': '60%', 'float': 'right', 'display': 'inline-block', 'margin': '10px', 'margin-bottom': '30px'}),

        html.Hr(style={'width': '100%', 'display': 'block'}), # horizontal line
    ], style={'columnCount': 1}), 

    dcc.Tabs([
        dcc.Tab(label='Variables comparison', children=[
            html.Hr(),  # horizontal line

            html.Div([                 
                dcc.Markdown(children='''### Correlation Analysis''',
                                style={
                                    'textAlign': 'left',
                                    'color': colors['text']
                                    }),
                
                dcc.Graph(id='scatter-plot'),

            ], style={'columnCount': 1}),

            html.Hr(),  # horizontal line

            html.Div([
                dcc.Markdown(children='''### Variables Distribution''',
                                style={
                                    'textAlign': 'left',
                                    'color': colors['text']
                                    }),

                html.Div([
                    dcc.Graph(id='xaxis-histogram'),
                ], style={'width': '40%', 'float': 'left', 'display': 'inline-block'}), 

                html.Div([
                    dcc.Graph(id='yaxis-histogram'),
                ], style={'width': '40%', 'float': 'center', 'display': 'inline-block'}),

                html.Div([
                    dash_table.DataTable(
                        id='table-statistics',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),
                    )
                ], style={'width': '20%', 'float': 'right', 'display': 'inline-block', 'margin-top': '10px'}),        
            ], style={'columnCount': 1}),

            html.Hr(),  # horizontal line            
        ], style={'backgroundColor': colors['background-header'], 'color': colors['text-header']}),

        dcc.Tab(label='Time Series', children=[
            html.Hr(),  # horizontal line

            dcc.Graph(id='xaxis-time-series'),

            html.Hr(),  # horizontal line

            dcc.Graph(id='yaxis-time-series'),

            html.Hr(),  # horizontal line                           
        ], style={'backgroundColor': colors['background-header'], 'color': colors['text-header']}),

        dcc.Tab(label='Correlation Matrix', children=[
            html.Hr(),  # horizontal line

            dcc.Graph(id='correlation-matrix'),

            html.Hr(),  # horizontal line                           
        ], style={'backgroundColor': colors['background-header'], 'color': colors['text-header']}),

        dcc.Tab(label='Raw Data', children=[
            html.Hr(),  # horizontal line
            html.Div(id='output-data-upload'),                
        ], style={'backgroundColor': colors['background-header'], 'color': colors['text-header']}),
    ]), 
])


######### Call back functions to update info #########

def filter_data(df, slider_value):
    df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
    min_date = pd.to_datetime(slider_value[0], unit='s')
    max_date = pd.to_datetime(slider_value[1], unit='s')
    df = df[df.iloc[:,0].between(min_date, max_date)]

    return df

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

def parse_contents(contents, filename, date, slider_value):
    
    df = parse_data(contents, filename)
    df = filter_data(df, slider_value)
    df = df.head(20).round(decimals=2)

    return html.Div([
        html.H6('Raw data first 20 lines: '),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        html.H6('File uploaded: ' + filename),
        html.H6('File timestamp: ' + pd.to_datetime(date, unit='s').strftime('%Y-%m-%d %H:%M:%S')),

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents'),
     Input('date--slider', 'value')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')])
def update_output(list_of_contents, slider_value, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d, slider_value) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(
    [Output('xaxis-column', 'options'),
     Output('yaxis-column', 'options'),
     Output('xaxis-column', 'value'),
     Output('yaxis-column', 'value')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_date_dropdown(contents, filename):
    
    available_indicators = np.array(['NO DATA'])
    value = available_indicators[0]
    
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        available_indicators = np.array(df.columns)
        value = available_indicators[1]

    options=[{'label': i, 'value': i} for i in available_indicators]

    return options, options, value, value

@app.callback(
    [Output('date--slider', 'min'),
     Output('date--slider', 'max'),
     Output('date--slider', 'value'),
     Output('date--slider', 'marks')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_date_rangeslider(contents, filename):
    
    min = 0
    max = 10
    value = [min, max]
    marks={i: '{}'.format(i) for i in range(min, max+1)}

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        dates = pd.to_datetime(df.iloc[:,0])
        full_dates = dates.dt.strftime('%Y-%m').unique()
        full_dates.sort()
        last_value = pd.to_datetime(full_dates[-1]).timestamp() + 86400*32
        last_value = pd.to_datetime(last_value, unit='s').strftime('%Y-%m')
        full_dates = np.append(full_dates, last_value)
        min = dates.min().timestamp()
        max = dates.max().timestamp()
        value = [min, max]
        marks={int(pd.to_datetime(i).timestamp()): '{}'.format(i) for i in full_dates}
    
    return min, max, value, marks

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('xaxis-type', 'value'),
     Input('yaxis-type', 'value'),
     Input('date--slider', 'value')])
def update_scatter_plot(contents, filename,
                 xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 slider_value):
    
    x = pd.Series()
    y = pd.Series()
    hover = []
    corr = ''

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        dff = filter_data(df, slider_value).round(decimals=3)
        x = dff.loc[:, xaxis_column_name]
        y = dff.loc[:, yaxis_column_name]
        hover = dff.iloc[:,0]
        corr = (dff.loc[:, [xaxis_column_name, yaxis_column_name]]
                   .corr(method='pearson')
                   .round(decimals=2)
                   .iloc[0, 1])

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x, 
                y=y, 
                mode='markers',
                marker_color= colors['markers'],
                hovertext=hover),
            ])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest', height=600)

    fig.add_annotation(
            x=x.max(),
            y=y.max(),
            text='Correlation: ' + str(corr),
            showarrow=False,
            font=dict(
                color=colors['text-attention'],
                size=20),
            )

    fig.update_xaxes(title=xaxis_column_name, 
                     type='linear' if xaxis_type == 'Linear' else 'log') 

    fig.update_yaxes(title=yaxis_column_name, 
                     type='linear' if yaxis_type == 'Linear' else 'log') 

    return fig

@app.callback(
    [Output('xaxis-time-series', 'figure'),
     Output('yaxis-time-series', 'figure')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('xaxis-type', 'value'),
     Input('yaxis-type', 'value'),
     Input('date--slider', 'value')])
def update_time_series(contents, filename,
                       xaxis_column_name, yaxis_column_name,
                       xaxis_type, yaxis_type,
                       slider_value):
    
    x = pd.Series()
    y1 = pd.Series()
    y2 = pd.Series()
    hover = []

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        dff = filter_data(df, slider_value).round(decimals=3)
        x = pd.to_datetime(dff.iloc[:,0], unit='s')
        y1 = dff.loc[:, xaxis_column_name]
        y2 = dff.loc[:, yaxis_column_name]
        hover = dff.iloc[:,0]

    # Time Series for variable 1 (xaxis_column_name)
    fig_x = go.Figure(
        data=[
            go.Scatter(
                x=x, 
                y=y1, 
                mode='markers',
                marker_color= colors['markers'],
                hovertext=hover),
            ])

    fig_x.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig_x.update_xaxes(title='Timestamp') 

    fig_x.update_yaxes(title=xaxis_column_name, 
                       type='linear' if xaxis_type == 'Linear' else 'log')

    # Time Series for variable 2 (yaxis_column_name)
    fig_y = go.Figure(
        data=[
            go.Scatter(
                x=x, 
                y=y2, 
                mode='markers',
                marker_color= colors['markers'],
                hovertext=hover),
            ])

    fig_y.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig_y.update_xaxes(title='Timestamp') 

    fig_y.update_yaxes(title=yaxis_column_name, 
                     type='linear' if yaxis_type == 'Linear' else 'log') 

    return fig_x, fig_y

@app.callback(
    [Output('xaxis-histogram', 'figure'),
     Output('yaxis-histogram', 'figure')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('date--slider', 'value')])
def update_histograms(contents, filename,
                      xaxis_column_name, yaxis_column_name,
                      slider_value):
    
    x = []
    y = []

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        dff = filter_data(df, slider_value)
        x = dff.loc[:, xaxis_column_name]
        y = dff.loc[:, yaxis_column_name]

    fig_x = go.Figure(data=[
                    go.Histogram(x=x, marker_color=colors['markers'])
                    ])

    fig_x.update_layout(
        margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
        xaxis_title_text=xaxis_column_name,
        yaxis_title_text='Count',
        height=450,
    )

    fig_y = go.Figure(data=[
                    go.Histogram(x=y, marker_color=colors['markers'])
                    ])

    fig_y.update_layout(
        margin={'l': 40, 'b': 40, 't': 10, 'r': 40},
        xaxis_title_text=yaxis_column_name,
        yaxis_title_text='Count',
        height=450,
    )

    return fig_x, fig_y

@app.callback(
    [Output('table-statistics', 'columns'),
     Output('table-statistics', 'data')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('date--slider', 'value')])
def update_statistics_table(contents, filename,
                            xaxis_column_name,
                            yaxis_column_name,
                            slider_value):
    
    dff = pd.DataFrame()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        dff = filter_data(df, slider_value)
        dff = (dff.loc[:, [xaxis_column_name, yaxis_column_name]]
                 .describe(percentiles=[.01, .10, .25, .50, .75, .90, .99])
                 .round(decimals=2)
                 .reset_index()
                 .rename(columns={'index': 'metric'}))

    columns = [{"name": i, "id": i} for i in dff.columns]
    data = dff.to_dict('records')

    return columns, data

@app.callback(
    Output('correlation-matrix', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('date--slider', 'value')])
def update_correlation_matrix(contents, filename,
                 xaxis_column_name, yaxis_column_name,
                 slider_value):
    
    corr_values= []
    corr = pd.DataFrame(columns=['x', 'y', 'value'])

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        dff = filter_data(df, slider_value).round(decimals=3)
        cols = list(dff.columns[1:])
        corr = (dff.loc[:, cols]
                   .corr(method='pearson')
                   .round(decimals=2))
        corr = pd.melt(corr.reset_index(), id_vars='index')
        corr.columns = ['x', 'y', 'value']
        corr_values = list(corr.loc[:, 'value'])
        

    fig = go.Figure(data=go.Heatmap(
                    z=corr_values,
                    x=corr['x'],
                    y=corr['y'],
                    colorscale='Viridis'))

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest', height=1200)
            
    return fig

if __name__ == '__main__':
    app.run_server(
        host='0.0.0.0',
        port=8050,
        debug=True
    )