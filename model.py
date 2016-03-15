import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold, cross_val_score

NDVI_DATAFILE = "input/ndvi.csv"
WETLAND_DATAFILE = "input/WP_2012195to2015126_L3.mat"
CIMIS_DATAFILE = "input/cimis.csv"

def process_cimis(data_file=CIMIS_DATAFILE):
    df = pd.read_csv(data_file)
    df2 = {}
    df2['year'] = df['Date'].apply(lambda x: int(x[-4:]))
    df2['doy'] = df['Jul']
    df2['sol_rad'] = df['Sol Rad (W/sq.m)']
    df2['net_rad'] = df['Net Rad (W/sq.m)']
    df2['max_air_temp'] = df['Max Air Temp (C)']
    df2['min_air_temp'] = df['Min Air Temp (C)']
    df2['avg_air_temp'] = df['Avg Air Temp (C)']
    df2['max_soil_temp'] = df['Max Soil Temp (C)']
    df2['min_soil_temp'] = cimis_fill_missing(df['Min Soil Temp (C)'].copy()).astype(np.float)
    df2['avg_soil_temp'] = cimis_fill_missing(df['Avg Soil Temp (C)'].copy()).astype(np.float)
    df2 = pd.DataFrame(df2)
    return df2

def cimis_fill_missing(series, n=1):
    for i in range(series.size):
        try:
            float(series[i])
        except:
            series[i] = np.mean([float(series[i - 1]), float(series[i + 1])])
    return series

def process_ndvi(data_file=NDVI_DATAFILE):
    df = pd.read_csv(data_file)
    df2 = {}
    df2['year'] = df['date[YYYYDDD]'].apply(lambda x: int(x[1:5]))
    df2['doy'] = df['date[YYYYDDD]'].apply(lambda x: int(x[-3:]))
    df2['ndvi'] = df['max']
    return pd.DataFrame(df2)

def process_wetland(data_file=WETLAND_DATAFILE):
    #import pdb; pdb.set_trace()
    # data
    data = sio.loadmat(data_file)
    # general flux data
    d = data['data'][0][0]
    df = {}
    df["co2_gf"] = d['wc_gf'][:,0]
    df["ch4_gf"] = d['wm_gf'][:,0]
    df["doy"] = d['DOY'][:,0]
    df["year"] = d['year'][:,0]
    df = pd.DataFrame(df)

    df2 = {}
    grouped = df.groupby(["year", "doy"]).aggregate(np.sum).reset_index()

    return grouped

def combine_all_data():
    ndvi_df = process_ndvi(data_file=NDVI_DATAFILE)
    wt_df = process_wetland(data_file=WETLAND_DATAFILE)
    cimis = process_cimis(data_file=CIMIS_DATAFILE)

    together = pd.merge(cimis, wt_df, on=['year', 'doy'])
    together = together.ix[14:].reset_index()

    # average 16 days worth of data
    every_16 = []
    i = 0
    while i < together.shape[0]:
        if i + 16 > together.shape[0]:
            break
        if together['doy'][i] + 16 > 365:
            num_days = 365 if together['year'][i] % 4 != 0 else 366
            i += num_days - together['doy'][i] + 1
            continue
        reduced = together[i:i+16].apply(np.mean)
        reduced['doy'] = together['doy'][i]
        reduced['year'] = together['year'][i]
        every_16.append(reduced)

        i += 16
    together = pd.concat(every_16, axis=1).T
    together = pd.merge(together, ndvi_df, on=['year', 'doy'])

    return together

def co2_create_XY(combined_data):
    Y = np.ravel(np.array([combined_data["co2_gf"]]).T)
    X = combined_data
    del X["doy"]
    del X["year"]
    del X["co2_gf"]
    del X["ch4_gf"]
    del X["index"]
    cols = X.columns
    X = np.array(X)
    X = preprocessing.scale(X)
    return X, Y, cols

def ch4_create_XY(combined_data):
    Y = np.ravel(np.array([combined_data["ch4_gf"]]).T)
    X = combined_data
    del X["doy"]
    del X["year"]
    del X["co2_gf"]
    del X["ch4_gf"]
    del X["index"]
    cols = X.columns
    X = np.array(X)
    X = preprocessing.scale(X)
    return X, Y, cols

def train(X, Y, print_oob_score=True, regressor=RandomForestRegressor(n_estimators=200, max_features='sqrt', oob_score=True)):
    regressor.fit(X, Y)
    if print_oob_score:
        print "Out of bag score", regressor.oob_score_
    return regressor

def cross_validation(X, Y, n_folds=3):
    kf = KFold(X.shape[0], n_folds=n_folds, shuffle=True)
    scores = []
    for train_indices, test_indices in kf:
        regressor = train(X[train_indices], Y[train_indices], print_oob_score=False)
        # print "Feature importances: ", regressor.feature_importances_
        score = regressor.score(X[test_indices], Y[test_indices])
        print "Cross validation w/ ", n_folds, "folds - score: ", score
        scores.append(score)
    print "Average Cross Validation Score w/ ", n_folds, "folds -", np.mean(scores)

def co2():
    combined_data = combine_all_data()
    X, Y, cols = co2_create_XY(combined_data)
    cross_validation(X, Y)
    regressor = train(X, Y)
    print dict(zip(cols, regressor.feature_importances_))
    return regressor

def ch4():
    combined_data = combine_all_data()
    X, Y, cols = ch4_create_XY(combined_data)
    cross_validation(X, Y)
    regressor = train(X, Y)
    print dict(zip(cols, regressor.feature_importances_))
    return regressor

if __name__ == "__main__":
    print "Running CO2 Regression..."
    co2_rgr = co2()
    print ""
    print "Running CH4 Regression..."
    ch4_rgr = ch4()
