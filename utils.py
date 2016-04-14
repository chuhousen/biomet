import numpy as np
import pandas as pd
import scipy.io as sio

HALF_HOURLY_SCALE = 1800. / 1e6 # each sample is half-hourly, so multiply by 1800
CARBON_SCALE = 1800. * 10e-6 * 12

###############################################
# PROCESS VARIOUS DATA FILES
###############################################

def process_cimis(filename, interpolate_missing=True):
    print "Processing {0}...".format(filename)
    df = pd.read_csv(filename)
    # replace bad values with NaN
    for col in ["Net Rad (W/sq.m)", "Avg Air Temp (C)", "Avg Soil Temp (C)"]:
        qc = df.columns.get_loc(col) + 1
        df.loc[df[df.columns[qc]] != ' ', col] = np.nan
    df2 = {}
    df2['year'] = df['Date'].apply(lambda x: int(x[-4:]))
    df2['doy'] = df['Jul']
    df2['net_rad'] = df['Net Rad (W/sq.m)']
    df2['avg_air_temp'] = df['Avg Air Temp (C)']
    df2['avg_soil_temp'] = df['Avg Soil Temp (C)']
    df2 = pd.DataFrame(df2)
    if interpolate_missing:
        df2 = interpolate_missing_values(df2)
    return df2

def process_modis_reflectance_veg_index(filename, prefix="", interpolate_missing=True):
    print "Processing {0}...".format(filename)
    if prefix:
        prefix += "_"
    df = pd.read_csv(filename)
    # select only good data rows
    df = df.loc[df["QC"] == 0]
    df2 = {}
    df2['doy'] = df['DOY']
    df2["year"] = df["YR"]
    df2[prefix + "ndvi"] = df["NDVI"]
    df2[prefix + "evi"] = df["EVI"]
    df2[prefix + "lswi"] = df["LSWI"]
    df2[prefix + "bnd1"] = df["bnd1.ref"]
    df2[prefix + "bnd2"] = df["bnd2.ref"]
    df2[prefix + "bnd3"] = df["bnd3.ref"]
    df2[prefix + "bnd7"] = df["bnd7.ref"]
    df2 = pd.DataFrame(df2)
    if interpolate_missing:
        df2 = interpolate_missing_values(df2)
    return df2

def process_modis_lst_emissivity(filename, prefix="", interpolate_missing=True):
    print "Processing {0}...".format(filename)
    if prefix:
        prefix += "_"
    df = pd.read_csv(filename)
    df.loc[df["QC.day"] != 0, "LST.day"] = np.nan
    df.loc[df["QC.night"] != 0, "LST.night"] = np.nan
    del df["QC.day"]
    del df["QC.night"]
    df.rename(columns={"YR": "year", "DOY": "doy", "LST.day": prefix + "LST.day",
                        "LST.night": prefix + "LST.night"}, inplace=True)
    if interpolate_missing:
        df = interpolate_missing_values(df)
    return df

def process_tower(filename, prefix="", interpolate_missing=True):
    print "Processing {0}...".format(filename)
    if prefix:
        prefix += "_"
    data = sio.loadmat(filename)
    d = data['data'][0][0]
    df = {}
    df["doy"] = d['DOY'][:,0]
    df["year"] = d['year'][:,0]
    # CO2 flux (umol m-2 s-1) => gC / m^2 / day
    df[prefix + "co2_gf"] = d['wc_gf'][:,0] * CARBON_SCALE
    # CH4 flux (nmol m-2 s-1) => mgC / m^2 / day
    df[prefix + "ch4_gf"] = d['wm_gf'][:,0] * CARBON_SCALE
    # ER => gC / m^2 / day
    df[prefix + "er"] = d['er_ANNnight'][:,0] * CARBON_SCALE
    # gross primary productivity => gC / m^2 / day
    df[prefix + "gpp"] = d['gpp_ANNnight'][:,0] * CARBON_SCALE
    # latent heat flux (W m-2) => Mj / m^2 / day
    df[prefix + "le"] = d['LE_gf'][:,0] * HALF_HOURLY_SCALE
    # sensible heat flux (W m-2) => Mj / m^2 / day
    df[prefix + "h"] = d['H_gf'][:,0] * HALF_HOURLY_SCALE
    # net radiation => Mj / m^2 / day
    df[prefix + "RNET"] = df["RNET"][:,0] * HALF_HOURLY_SCALE
    if "Rlong_in" in d:
        # long-wave radiation => Mj / m^2 / day
        df[prefix + "Rlong_in"] = df["Rlong_in"][:,0] * HALF_HOURLY_SCALE

    df = pd.DataFrame(df)
    grouped = df.groupby(["year", "doy"]).aggregate(np.sum).reset_index()
    if interpolate_missing:
        grouped = interpolate_missing_values(grouped)
    return grouped

###############################################
# INTERPOLATE MISSING DATA
###############################################

def interpolate_missing_values(df):
    min_year, min_doy = df["year"].min(), df[df["year"] == df["year"].min()]["doy"].min()
    max_year, max_doy = df["year"].max(), df[df["year"] == df["year"].max()]["doy"].max()
    missing_rows = []
    year, doy = min_year, min_doy
    while year <= max_year:
        if year == max_year and doy > max_doy:
            break
        row_exists = sum((df["year"] == year) & (df["doy"] == doy)) == 1
        if not row_exists:
            new_row = dict(zip(df.columns, [np.nan] * len(df.columns)))
            new_row["year"] = year
            new_row["doy"] = doy
            missing_rows.append(new_row)
        if (year % 4 == 0 and doy == 366) or (year % 4 != 0 and doy == 365):
            year += 1
            doy = 1
        else:
            doy += 1
    df = pd.concat((df, pd.DataFrame(missing_rows)))
    return df.sort_values(["year", "doy"]).reset_index(drop=True).interpolate(method='spline')

###############################################
# MERGE DATAFRAMES TOGETHER
###############################################

def merge_dataframes(dfs):
    df = reduce(lambda left, right: pd.merge(left, right, on=['year', 'doy']), dfs)
    return df

if __name__ == "__main__":
    WESTPOND_DATAFILE = "input/WP_2012195to2015126_L3.mat"
    # process_tower(WESTPOND_DATAFILE)
