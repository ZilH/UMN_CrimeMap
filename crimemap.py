from pyproj import Proj, transform
import panel as pn

import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, WIKIMEDIA, CARTODBPOSITRON, STAMEN_TERRAIN, STAMEN_TONER, ESRI_IMAGERY, OSM
from bokeh.io import output_notebook, show, curdoc
from bokeh.models import Select, ColumnDataSource, FactorRange, DataRange1d, Panel, Tabs, DataTable, TableColumn, DateFormatter, Div, LabelSet
from bokeh.layouts import gridplot, column, row
from bokeh.events import Tap
from bokeh.models import ColorBar

import warnings
import logging

from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6, RdBu6, Magma256

import sys
from bokeh.models.widgets import Button


def button_callback():
    sys.exit()  # Stop the server

button = Button(label="Stop", button_type="success")
button.on_click(button_callback)

log = logging.getLogger('bokeh')


# customCMap = ['#00BFFF','#1E90FF','#fecc5c','#fd8d3c','#f03b20','#bd0026']
customCMap = ['#A9A9A9','#FFF8DC','#fecc5c','#fd8d3c','#f03b20','#bd0026']

columns = [
            TableColumn(field="Case Number", title="Case Number"),
            TableColumn(field="Incident Location", title="Incident Location"),
            TableColumn(field="Nature of Offense", title="Nature of Offense"),
            TableColumn(field="Date Reported", title="Date Reported",formatter=DateFormatter()),
            TableColumn(field="Time Occured", title="Time Occured"),
            TableColumn(field="Disposition", title="Disposition"),
        ]


df_all = pd.read_pickle('2020_2021_cleaned_geocoded_all_Mercator')
df_all['Incident Location'] = df_all['Incident Location'].str.upper()
df_all.reset_index(drop=True,inplace=True)

df_2020 = df_all[df_all['Date Reported'].dt.year == 2020]
df_2021 = df_all[df_all['Date Reported'].dt.year == 2021]

Upleft = [44.991509, -93.256765]
Lowerright = [44.941402, -93.064918]

def calFreq(df_1, year):

    if year != 'All years':

        df_1 = df_1[df_1['Date Reported'].dt.year == int(year)]

    filter = (df_1['lat'] > Lowerright[0]) & (df_1['lat'] < Upleft[0]) & (df_1['lng'] > Upleft[1]) & (df_1['lng'] < Lowerright[1])
    df_1 = df_1[filter]
    # lat, lon = df_1['lat'].mean(), df_1['lng'].mean()
    unique_locations = df_1['Incident Location'].value_counts()

    # df_1.groupby('Incident Location').apply(lambda x: x['Nature of Offense'].values.tolist())

    CR_index = pd.DataFrame({"Raw_String" : unique_locations.index, "ValueCount":unique_locations})
    CR_index.index = range(len(unique_locations))

    MercatorX = []
    MercatorY = []
    color = []
    label = []
    crimes = []

    for i in CR_index['Raw_String']:
        id = df_1['Incident Location'] == i
        dfSel = (df_1[id])

        if(len(dfSel) == 0):
            MercatorX.append(None)
            MercatorY.append(None)
            color.append(None)
            label.append(None)
            crimes.append(None)
            continue
        else:
            if(len(dfSel) < 10):
              color.append("dodgerblue")
              label.append("freq < 10")
            if(len(dfSel) >= 10 and len(dfSel) < 20):
              color.append("lightsalmon")
              label.append("freq < 20")
            if(len(dfSel) >= 20):
              color.append("red")
              label.append("freq > 20")

        MercatorX.append(dfSel['MercatorX'].iloc[0])
        MercatorY.append(dfSel['MercatorY'].iloc[0])
        crimes.append(dfSel.groupby('Incident Location').apply(lambda x: list(df_1['Nature of Offense'])))

        # print(i)
        # print(dfSel['MercatorX'].iloc[0])
        # print(dfSel['MercatorY'].iloc[0])

    CR_index['MercatorX'] = MercatorX
    CR_index['MercatorY'] = MercatorY
    CR_index['color'] = color
    CR_index['label'] = label
    # CR_index['crimes'] = crimes

    return ColumnDataSource(data = CR_index)


def map_plot(source, year):

    CR_index = source.data

    # inProj = Proj(init='epsg:3857')
    # outProj = Proj(init='epsg:4326')

    # world_lon1, world_lat1 = transform(outProj,inProj,-180,-85)
    # world_lon2, world_lat2 = transform(outProj,inProj,180,85)

    world_lon1, world_lat1 = CR_index["MercatorX"].min(),CR_index["MercatorY"].min()
    world_lon2, world_lat2 = CR_index["MercatorX"].max(),CR_index["MercatorY"].max()

    cartodb = get_provider(CARTODBPOSITRON)

    TOOLTIPS = [
        ("Frequency", "@ValueCount"),
        ("Location", "@Raw_String"),
        # ("desc", "@desc"),
    ]

    TOOLS = "pan,wheel_zoom,box_select,reset,tap"

    plot_options = dict(width=1100, plot_height=600, tools=TOOLS , tooltips=TOOLTIPS,

                x_axis_type="mercator", y_axis_type="mercator")

    fig = figure(x_range=(world_lon1, world_lon2),y_range=(world_lat1, world_lat2),**plot_options,margin=(0,30,30,0))
    fig.add_tile(cartodb)
    fig.title.text = "UMN Crime Map in " + year
    fig.title.align = "center"

    # fig2 = figure(x_range=fig.x_range, y_range=fig.y_range, **plot_options)
    # fig2.add_tile(cartodb)
    # fig2.title.text = "UMN Crime Map 2021"
    # fig2.title.align = "center"

    # fig = figure(plot_width=400, plot_height=700, tools = TOOLS, tooltips=TOOLTIPS,
    #             x_range=(world_lon1, world_lon2),
    #             y_range=(world_lat1, world_lat2),
    #             x_axis_type="mercator", y_axis_type="mercator")
    # fig.add_tile(cartodb)

    # fig2 = figure(plot_width=400, plot_height=700, tools = TOOLS, tooltips=TOOLTIPS,
    #             x_range=(world_lon1, world_lon2),
    #             y_range=(world_lat1, world_lat2),
    #             x_axis_type="mercator", y_axis_type="mercator")
    # fig2.add_tile(cartodb)

        # definition of the column data source:
    # source = ColumnDataSource(CR_index)
    # source2 = ColumnDataSource(CR_index2)
        # see how we specify the x and y columns as strings,
        # and how to declare as a source the ColumnDataSource:
    # Magma256.reverse()
    mapper = linear_cmap(field_name='ValueCount', palette=customCMap ,low=min(CR_index['ValueCount']) ,high=max(CR_index['ValueCount']))
    # mapper = linear_cmap(field_name='ValueCount', palette=customCMap ,low=100 ,high=min(CR_index['ValueCount']))


    center = fig.circle(x="MercatorX", y="MercatorY",
            size="ValueCount",
            # fill_color="color", line_color="color",
            line_color=mapper,color=mapper,
            fill_alpha=0.3 ,
            # legend_group='label',
            source=source)
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8)

    fig.add_layout(color_bar, 'right')

    fig.title.text_font_size = "20px"

    # center2 = fig2.circle(x="MercatorX", y="MercatorY",
    #         size="ValueCount",
    #         fill_color="color", line_color="color",
    #         fill_alpha=0.3 ,legend_group='label',
    #         source=CR_index2)

    # p = gridplot([[fig, fig2]])
    #
    # year_select = Select(name='Years', options=['All', '2021', '2020'])
    # width_slider = pn.widgets.FloatSlider(name='Line Width', start=0.1, end=10)
    # width_slider.jslink(r.glyph, value='line_width')
    # markdown = pn.pane.Markdown("Some text")

    return fig


def get_loc_dataset(df, loc, year):

    if year != 'All years':

        df = df[df['Date Reported'].dt.year == int(year)]

    df_by_loc = df.groupby(['Incident Location','Nature of Offense'])['Case Number'].count().to_frame('Case Number').reset_index()

    df_now = df_by_loc[df_by_loc['Incident Location'] == loc].sort_values(by=['Case Number'],ascending=True)

    df_now.reset_index(drop=True,inplace=True)

    return ColumnDataSource(data=df_now)


def get_year_dataset(df, year):

    if year == 'All years':

        df_by_cate = df.groupby('Nature of Offense')['Case Number'].count().to_frame('Case Number').reset_index()

        df_by_cate_major = df_by_cate[(df_by_cate['Case Number']>24)].sort_values(by=['Case Number'],ascending=True)

        df_now = {
            'cr': df_by_cate_major['Nature of Offense'].tolist(),
            '2020': [df_2020[df_2020['Nature of Offense']==i].shape[0] for i in df_by_cate_major['Nature of Offense']],
            '2021': [df_2021[df_2021['Nature of Offense']==i].shape[0] for i in df_by_cate_major['Nature of Offense']]
        }

    elif year == '2020':

        df_by_cate = df[df['Date Reported'].dt.year == int(year)].groupby('Nature of Offense')['Case Number'].count().to_frame('Case Number').reset_index()
        df_by_cate_major = df_by_cate[(df_by_cate['Case Number']>24)].sort_values(by=['Case Number'],ascending=True)
        df_now = {
            'cr': df_by_cate_major['Nature of Offense'].tolist(),
            '2020':  df_by_cate_major['Case Number'].tolist(),
            '2021': [0] * len(df_by_cate_major)
        }

    elif year == '2021':

        df_by_cate = df[df['Date Reported'].dt.year == int(year)].groupby('Nature of Offense')['Case Number'].count().to_frame('Case Number').reset_index()
        df_by_cate_major = df_by_cate[(df_by_cate['Case Number']>24)].sort_values(by=['Case Number'],ascending=True)
        df_now = {
            'cr': df_by_cate_major['Nature of Offense'].tolist(),
            '2021':  df_by_cate_major['Case Number'].tolist(),
            '2020': [0] * len(df_by_cate_major)
        }

    return ColumnDataSource(data=df_now)

def get_data_grid(df, loc, year):

    if year != 'All years':

        df_now = df[(df['Incident Location']==loc)&(df['Date Reported'].dt.year==int(year))][['Case Number','Nature of Offense','Date Reported','Disposition','Time Occured', 'Incident Location']]

    else:
        df_now = df[df['Incident Location']==loc][['Case Number','Nature of Offense','Date Reported','Disposition','Time Occured','Incident Location']]

    df_now.reset_index(drop=True,inplace=True)

    for i in range(len(df_now)):
        my_str = df_now['Time Occured'][i]
        df_now['Time Occured'][i] = ':'.join(my_str[i:i+2] for i in range(1, len(my_str), 2))

    return df_now


def make_loc_plot(source, loc):

    p = figure(x_axis_label='Counts', title = "Total Counts of Crimes at "+ loc, width = 500, height = 600,
               x_range=DataRange1d(range_padding=0.0), y_range=FactorRange(factors = source.data['Nature of Offense'].tolist()),tools='')

    p.hbar(source=source, y='Nature of Offense', right='Case Number', height =0.6, fill_color="dodgerblue"
    , line_color="dodgerblue", fill_alpha=0.3)

    # Turn off gridlines on categorical axis
    p.ygrid.grid_line_color = None
    p.title.text_font_size = "15px"
    p.title.vertical_align = 'top'
    p.title.align = "center"

    return p


def make_year_plot(source, year):

    p = figure(x_axis_label='Counts', title = "Total Counts of Crimes in "+str(year), width = 500, height = 600,
        x_range=DataRange1d(range_padding=0.0), y_range=source.data['cr'], tools='')

    years = ['2020','2021']

    p.hbar_stack(years, y='cr', height=0.7, alpha=0.5, color=["blue", "red"], source=source,
                legend_label=years)

    # left_labels = LabelSet(x='2020', y='cr', text='2020', source=source, render_mode='canvas')

    p.legend.location = "center_right"
    p.title.text_font_size = "15px"
    p.ygrid.grid_line_color = None
    p.title.align = "center"
    # p.add_layout(left_labels)

    return p


def make_data_grid(source):

    source = ColumnDataSource(data=source)

    columns = [
            TableColumn(field="Case Number", title="Case Number"),
            TableColumn(field="Incident Location", title="Incident Location"),
            TableColumn(field="Nature of Offense", title="Nature of Offense"),
            TableColumn(field="Date Reported", title="Date Reported",formatter=DateFormatter()),
            TableColumn(field="Time Occured", title="Time Occured"),
            TableColumn(field="Disposition", title="Disposition"),
        ]

    data_table = DataTable(source=source, columns=columns, width=1000, editable = True)

    return data_table


def update_plots(attrname, old, new):
    year = year_select.value
    loc = loc_select.value

    src_loc = get_loc_dataset(df_all, loc, year)
    loc_source.data.update(src_loc.data)
    loc_plot.title.text = "Total Counts of Crimes at "+ loc
    loc_plot.x_range = DataRange1d(range_padding=0.0)
    loc_plot.y_range.factors =  loc_source.data['Nature of Offense'].tolist()

    src_year = get_year_dataset(df_all, year)
    year_source.data.update(src_year.data)
    year_plot.title.text = "Total Counts of Crimes in "+ year
    year_plot.x_range = DataRange1d(range_padding=0.0)
    # year_plot.y_range.factors =  year_source.data['Nature of Offense'].tolist()
    year_plot.y_range.factors = year_source.data['cr']
    # left_labels.source = src_year


    src_map = calFreq(df_all, year)
    map_source.data.update(src_map.data)
    map_plot.title.text = "UMN Crime Map in " + year

    src_grid = get_data_grid(df_all, loc, year)
    data_grid.source.data = src_grid


def my_tap_handler(attr, old, new):
    year = year_select.value

    if new != []:
        index = new[0]

        loc = calFreq(df_all, year).data['Raw_String'][index]
        src_loc = get_loc_dataset(df_all, loc, year)
        loc_source.data.update(src_loc.data)
        loc_plot.title.text = "Total Counts of Crimes at "+ loc
        loc_plot.x_range = DataRange1d(range_padding=0.0)
        loc_plot.y_range.factors = loc_source.data['Nature of Offense'].tolist()
        loc_select.value = loc

        src_grid = get_data_grid(df_all, loc, year)
        data_grid.source.data = src_grid



loc = '100 CHURCH ST SE, MN'
year = 'All years'

locs = df_all['Incident Location'].unique().tolist()
years = ['All years','2020','2021']

year_select = Select(value=year, title='Year', options=sorted(years),margin=(40,0,20,0))
loc_select = Select(value=loc, title='Location', options=sorted(locs))


loc_source = get_loc_dataset(df_all, loc, year)
year_source = get_year_dataset(df_all, year)
loc_plot = make_loc_plot(loc_source, loc)
year_plot = make_year_plot(year_source, year)
map_source = calFreq(df_all, year)
map_plot = map_plot(map_source, year)
data_grid_source = get_data_grid(df_all, loc, year)
data_grid = make_data_grid(data_grid_source)

loc_select.on_change('value', update_plots)
year_select.on_change('value', update_plots)
map_source.selected.on_change("indices", my_tap_handler)

controls = column(year_select, loc_select, button)

# div1 = Div(text="<img src='https://github.com/VL914/random_demo/raw/master/Picture1.png'>")
div2 = Div(text="<img src='https://github.com/VL914/random_demo/raw/master/Picture2.png'>", width = 400, height = 200)

tab1 = Panel(child=column(row(map_plot,controls),row(loc_plot, year_plot),height=2000), title="Map")
tab2 = Panel(child=data_grid, title="Data Grid")
tab3 = Panel(child=div2, title="About data")
tabs = Tabs(tabs=[tab1, tab2, tab3])

curdoc().add_root(tabs)
