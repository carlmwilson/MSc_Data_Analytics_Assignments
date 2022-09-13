"""
A dashboard for exploring car features and their
impact on fuel economy and CO2 emissions.
"""
import math
import pandas as pd
import plotly.express as px
import plotly.io as pio
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output

pio.templates.default="simple_white"

master_data_import=pd.read_csv("CanadaCO2Data_Clean.csv")

working_data_frame=master_data_import.sort_values([
    "make","model","vehicle_class","transmission","fuel_type"],
                                                  ascending=[True,True,True,True,True
                                                             ]
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H2(
            "Passenger Vehicle Fuel Economy & CO2 Emissions Dashboard",
            style={"text-align":"center"}

        ),

        html.Br(),
        html.Div([
            html.Div([
                html.H4("Vehicle Type Selections"),
                dcc.Dropdown(id="VehicleClass",
                             options=[
                                 {"label": vehicle_class, "value": vehicle_class}
                                 for vehicle_class in sorted(
                                     working_data_frame.vehicle_class.unique()
                                 )],
                             value=["COMPACT", "FULL-SIZE", "MID-SIZE", "SUV - STANDARD"],
                             multi=True,
                             style={"width": "100%"},
                             searchable=True
                             ),
            ],className="four columns"),

            html.Div([
                html.H4("Fuel Type Selections"),
                dcc.Dropdown(id="FuelType",
                             options=[
                                 {"label": fuel, "value": fuel}
                                 for fuel in sorted(working_data_frame.fuel_type.unique())],
                             value=["Diesel", "Premium Gasoline", "Regular Gasoline"],
                             multi=True,
                             style={"width": "100%"},
                             searchable=True,
                             ),
            ],className="four columns"),

            html.Div([
                html.H4("Transmission Selections"),
                dcc.Dropdown(id="Transmission",
                             options=[
                                 {"label": transmission, "value": transmission}
                                 for transmission in sorted(
                                     working_data_frame.transmission.unique()
                                 )],
                             value=["Automatic", "Manual"],
                             multi=True,
                             style={"width": "100%"},
                             searchable=True
                             ),
            ],className="four columns"),
        ], className="row"),

        html.Br(),
        html.Div([
            html.Div([
                html.H4("Fuel Economy Range [mpg]"),
                dcc.RangeSlider(
                    id="fuel_economy_slider",
                    marks={i: str(i) for i in range(0, 70, 1)},
                    min=0,
                    max=70,
                    step=1,
                    value=[0, 70],
                    dots=True,
                    allowCross=False,
                ),
            ]),
        ], className="row"),

        html.Br(),
        html.Br(),

        html.Div([
            dcc.Tabs([
                dcc.Tab(label="Vehicle Type [Full Size, Compact, ...]", children=[
                    dcc.Graph(id="vehicle_type_bar_chart", figure={})
                ]),

                dcc.Tab(label="Transmission Type [Manual, Automatic, ...]", children=[
                    dcc.Graph(id="transmission_bar_chart", figure={})
                ]),

                dcc.Tab(label="Fuel Type, Economy & Emissions", children=[
                    dcc.Graph(id="fuel_vs_emissions_scatter", figure={})
                ]),

                dcc.Tab(label="Vehicle Shortlist...", children=[
                    dash_table.DataTable(id="results_table",
                                         columns=[{"id": "make", "name":"Make"},
                                                  {"id": "model", "name":"Model"},
                                                  {"id": "vehicle_class", "name": "Vehicle Type"},
                                                  {"id": "fuel_type", "name": "Fuel"},
                                                  {"id": "transmission","name":"Transmission"},
                                                  {"id": "fuel_consumption_combined_mpg",
                                                   "name":"Combined Fuel Consumption [mpg]"},
                                                  {"id": "co2_emissions_g_per_km",
                                                   "name":"CO2 Emissions [g/km]"},
                                                  {"id": "engine_size", "name": "Engine Size [L]"},
                                                  {"id": "cylinder", "name": "Number of Cylinders"},
                                                  {"id": "gears", "name": "Number of Gears"}
                                         ],

                                         style_cell={
                                             "textAlign":"left",
                                             "fontSize":18,
                                             "font-family":"Calibri"
                                         },

                                         style_header={
                                             "backgroundColor":"rgb(66, 135, 245)",
                                             "color":"rgb(255,255,255)",
                                             "fontWeight":"bold",
                                             "fontSize":20,
                                             "font-family":"Calibri"
                                         },
                                         style_as_list_view=True,
                                         sort_action="native"

                                         )
                ]),

            ]),
        ]),
    ]),
])

@app.callback(
    [Output(component_id="vehicle_type_bar_chart",component_property="figure"),
     Output(component_id="transmission_bar_chart",component_property="figure"),
     Output(component_id="fuel_vs_emissions_scatter",component_property="figure"),
     Output(component_id="fuel_economy_slider",component_property="min"),
     Output(component_id="fuel_economy_slider",component_property="max"),
     Output(component_id="results_table",component_property="data"),
     ],
    [Input(component_id="fuel_economy_slider",component_property="value"),
     Input(component_id="Transmission",component_property="value"),
     Input(component_id="FuelType",component_property="value"),
     Input(component_id="VehicleClass",component_property="value")
     ]
)

def update_graph(selected_fuel_economy,selected_transmissions,selected_fuels,selected_class):
    dff=working_data_frame.copy()

    if type(selected_transmissions)!=str:
        dff=dff[
            dff["transmission"].isin(selected_transmissions)
            & dff["fuel_type"].isin(selected_fuels)
            & dff["vehicle_class"].isin(selected_class)
            ]
    else:
        dff=dff[
            dff["transmission"] == selected_transmissions
            & dff["fuel_type"] == selected_fuels
            & dff["vehicle_class"] == selected_class
            ]

    fe_min=math.floor(dff.fuel_consumption_combined_mpg.min())
    fe_max=math.ceil(dff.fuel_consumption_combined_mpg.max())

    fe_low, fe_high = selected_fuel_economy
    mask = (dff["fuel_consumption_combined_mpg"] < fe_high) & \
           (dff["fuel_consumption_combined_mpg"] > fe_low)
    dff=dff[mask]


    table_output=dff.to_dict("records")



    scatter_plot = px.scatter(
        dff,
        x="co2_emissions_g_per_km",
        y="fuel_consumption_combined_mpg",
        color="fuel_type",
        hover_data=["make","model"],
        title="Fuel Consumption vs. CO2 Emissions",
        width=1800, height=600,
    )

    scatter_plot.update_layout(
        xaxis_title="Tailpipe CO2 Emissions [g/km]",
        yaxis_title="Combined Cycle Fuel Consumption [mpg]"
    )

    box_vehicle = px.box(
        dff,
        y="fuel_consumption_combined_mpg",
        x="vehicle_class",
        color="fuel_type",
        hover_data=["make","model"],
        title="Fuel Consumption vs. Vehicle Class",
        width=1800, height=600,
    )

    box_vehicle.update_layout(
        xaxis_title="Vehicle Class",
        yaxis_title="Combined Cycle Fuel Consumption [mpg]"
    )

    box_transmission = px.box(
        dff,
        y="fuel_consumption_combined_mpg",
        x="transmission",
        color="fuel_type",
        hover_data=["make","model"],
        title="Fuel Consumption vs. Transmission Type",
        width=1800, height=600,
    )

    box_transmission.update_layout(
        xaxis_title="Transmission Type",
        yaxis_title="Combined Cycle Fuel Consumption [mpg]"
    )

    return box_vehicle, box_transmission, scatter_plot, fe_min, fe_max, table_output

if __name__ == "__main__":
    app.run_server(debug=False)