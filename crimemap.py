from pyproj import Proj, transform
import panel as pn

import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, WIKIMEDIA, CARTODBPOSITRON, STAMEN_TERRAIN, STAMEN_TONER, ESRI_IMAGERY, OSM
from bokeh.io import output_notebook, show, curdoc
from bokeh.models import Select, ColumnDataSource, FactorRange, DataRange1d
from bokeh.layouts import gridplot, column, row
from bokeh.events import Tap

import warnings
import logging

log = logging.getLogger('bokeh')


df_all = pd.read_pickle('2020_2021_cleaned_geocoded_all_Mercator')
df_all['Incident Location'] = df_all['Incident Location'].str.upper()
df_all.reset_index(drop=True,inplace=True)

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

    plot_options = dict(width=1000, plot_height=800, tools=TOOLS , tooltips=TOOLTIPS,

                x_axis_type="mercator", y_axis_type="mercator")

    fig = figure(x_range=(world_lon1, world_lon2),y_range=(world_lat1, world_lat2),**plot_options)
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


    center = fig.circle(x="MercatorX", y="MercatorY",
            size="ValueCount",
            fill_color="color", line_color="color",
            fill_alpha=0.3 ,legend_group='label',
            source=source)
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

    return ColumnDataSource(data=df_now)


def get_year_dataset(df, year):

    if year == 'All years':

        df_by_cate = df.groupby('Nature of Offense')['Case Number'].count().to_frame('Case Number').reset_index()

        df_by_cate_major = df_by_cate[(df_by_cate['Case Number']>24)].sort_values(by=['Case Number'],ascending=True)

    else:

        df_by_cate = df[df['Date Reported'].dt.year == int(year)].groupby('Nature of Offense')['Case Number'].count().to_frame('Case Number').reset_index()

        df_by_cate_major = df_by_cate[(df_by_cate['Case Number']>24)].sort_values(by=['Case Number'],ascending=True)


    df_now = df_by_cate_major

    return ColumnDataSource(data=df_now)


def make_loc_plot(source, loc):

    p = figure(x_axis_label='Counts', title = "Total Counts of Crimes at "+ loc, width = 500, height = 600,
               x_range=DataRange1d(range_padding=0.0), y_range=FactorRange(factors = source.data['Nature of Offense'].tolist()),tools='')

    p.hbar(source=source, y='Nature of Offense', right='Case Number', height =0.6, fill_color="dodgerblue"
    , line_color="dodgerblue", fill_alpha=0.3)

    # Turn off gridlines on categorical axis
    p.ygrid.grid_line_color = None

    return p


def make_year_plot(source, year):

    p = figure(x_axis_label='Counts', title = "Total Counts of Crimes in "+str(year), width = 500, height = 600,
              x_range=DataRange1d(range_padding=0.0),y_range=FactorRange(factors = source.data['Nature of Offense'].tolist()), tools='')

    p.hbar(source=source, y='Nature of Offense', right='Case Number',height =0.6,  fill_color="red"
      , line_color="red", fill_alpha=0.3)

    return p


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
    year_plot.y_range.factors =  year_source.data['Nature of Offense'].tolist()

    src_map = calFreq(df_all, year)
    map_source.data.update(src_map.data)
    map_plot.title.text = "UMN Crime Map in " + year



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


loc = '10 Church St Se, MN'
year = 'All years'

locs = df_all['Incident Location'].unique().tolist()
years = ['All years','2020','2021']

loc_select = Select(value=loc, title='Location', options=sorted(locs))
year_select = Select(value=year, title='Year', options=sorted(years))

loc_source = get_loc_dataset(df_all, loc, year)
year_source = get_year_dataset(df_all, year)
loc_plot = make_loc_plot(loc_source, loc)
year_plot = make_year_plot(year_source, year)
map_source = calFreq(df_all, year)
map_plot = map_plot(map_source, year)

loc_select.on_change('value', update_plots)
year_select.on_change('value', update_plots)
map_source.selected.on_change("indices", my_tap_handler)

controls = column(loc_select,year_select)

curdoc().add_root(column(row(map_plot,controls),row(loc_plot, year_plot)))
