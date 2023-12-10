# TRIED ALREADY

1) različno število local custom layerjev brez skip connectionv
2) skip connections na zadnji plasti
3) skip connections na vsaki plasti, 5 in 12 plasti


# WORKS BEST

1) skip connections na vsak plasti 8 plasti, 128 hidden channels


# TO DO LGCN (Testirat je treba z 128 5 layer modelom, drugače traja sto let, primerjaj z layerd_128_static_wo_lat)

1) NN agregation

2) testirat k1 in k2 povezave

3) različni scatter reduce to try: amax, amin, sum, prod ....

4) leaky_relu, elu -- RUNNING

5) testirat razične načine kako dodati node feature v agregaciji/updatu

6) multihead attention pri agregacija

7) dropout, batch norm

8) DODAT PODATKE


# SPREMENLJIVKE, KI SO NA VOLJO

1) geopotential 250hPa, 500hPa, 700hPa, 850hPa 
2) potencialna vrtinčnost 300 hPa, 500hPa - potential_vorticity
3) precipitation - padavine 
4) sic - sea ice cover 
5) snow - snow depth
6) soil_moisture_layer_1 - Soil moisture 0-7cm
7) surface_temperature
8) t2m - T at 2m
9) t 500 800 - T at 500 hPa in 800 hPa
10) TOA???
11) twv - total column water vapour
12) u v 10m - veter v-z in s-j na 10 m
13) u v 200 500 700 900 - veter v-z in s-j na 200 500 700 900 hPa

# KERE SPREMENLJIVKE IZBRAT

to prav Gregor:

Set 1: Basic Essential Variables

This set focuses on the core variables that are fundamental for most weather forecasting models.

* Mean Sea Level Pressure (MSLP)
* Geopotential Height at 500 hPa (geop500)
* Surface Temperature (surface T)
* Temperature at 850 hPa (T 850hPa)
* Total Column Water Vapor
* Precipitation
* Top of Atmosphere Radiation (TOA)

set1 = ["geopotential_500",
        "mslp",
        "precipitation",
        "surface_temperature",
        "t850",
        "twv",
        "toa_solar_radiation",
         ]

Set 2: Intermediate Detailed Variables

This set includes all the variables from Set 1 and adds more variables that offer additional detail, particularly in atmospheric dynamics.

* All variables from Set 1
* Geopotential Height at 850 hPa (geop850) and 250 hPa (geop250)
* Potential Vorticity at 500 hPa (pot vor 500)
* Wind Speed and Direction at 500 hPa (u500hPa, v500hPa) and 850 hPa (u850hPa, v850hPa)
* Temperature at 500 hPa (T 500hPa)
* Soil Moisture (0-7 cm)
* Sea Ice Cover


set2 = ["geopotential_250",
        "geopotential_500",
        "geopotential_850",
        "mslp",
        "potential_vorticity_500",
        "precipitation",
        "sic",
        "soil_moisture_layer_1",
        "surface_temperature",
        "t500",
        "toa_solar_radiation",
        "twv",
        "u_500",
        "v_500",
        "u_900",
        "v_900",
         ]

Set 3: Comprehensive Advanced Variables

This set is the most comprehensive, including all variables from the previous sets, and incorporates additional upper and lower atmosphere variables for a more complete picture.

* All variables from Set 2
* Geopotential Height at 700 hPa (geop700)
* Potential Vorticity at 300 hPa (pot vor 300)
* Wind Speed and Direction at 200 hPa (u200hPa, v200hPa), 700 hPa (u700hPa, v700hPa), and 10m (u10m, v10m)
* Temperature at 2m (T 2m) and 700 hPa (T 700hPa)
* Snow Depth

set3 = ["geopotential_250",
        "geopotential_500",
        "geopotential_700",
        "geopotential_850",
        "mslp",
        "potential_vorticity_500",
        "potential_vorticity_300",
        "precipitation",
        "sic",
        "snow",
        "soil_moisture_layer_1",
        "surface_temperature",
        "t2m",
        "t500",
        "t800",
        "toa_solar_radiation",
        "twv",
        "u_10m",
        "v_10m",
        "u_200",
        "v_200",
        "u_500",
        "v_500",
        "u_700",
        "v_700",
        "u_900",
        "v_900",
         ]