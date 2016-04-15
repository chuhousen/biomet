import pandas as pd
import utils
import os
import numpy as np

###############################################
# FEATURIZE
###############################################

def featurize(df, train_cols, test_col, years=None):
    """
    years - list of years (e.g. [2012, 2013]) to filter the data
    """
    if years:
        df = filter_df_by_years(df, years)
    X, Y = np.array(df[train_cols]), np.ravel(np.array(df[test_col]))
    return X, Y

def filter_df_by_years(df, years):
    return df[reduce(lambda x, y: x | y, map(lambda year: df["year"] == year, years))]

###############################################
# EXPERIMENT 1
###############################################

REL_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
CIMIS_DATAFILE = REL_PATH + "data/CIMIS/20160413-CIMIS-TWT-daily-2001-2016.csv"
WESTPOND_DATAFILE = REL_PATH + "input/WP_2012195to2015126_L3.mat"
WESTPOND_REFL_VEG_IDX = REL_PATH + "data/MODIS/2016-04-13-US-Tw1-MOD13Q1.csv"
WESTPOND_LST_EM = REL_PATH + "data/MODIS/2016-04-13-US-Tw1-MOD11A2.csv"
MAYBERRY_DATAFILE = REL_PATH + "input/MB_2010287to2016055_L3.mat"
MAYBERRY_REFL_VEG_IDX = REL_PATH + "data/MODIS/2016-04-13-US-Myb-MOD13Q1.csv"
MAYBERRY_LST_EM = REL_PATH + "data/MODIS/2016-04-13-US-Myb-MOD11A2.csv"
TOWER_LWI = REL_PATH + "data/Tower-data/2016-04-14_daily_TOWER_LW.csv"

def get_exp1_data(save_filename="data/exp1.csv"):
    if os.path.isfile(REL_PATH + save_filename):
        return pd.read_csv(REL_PATH + save_filename, index_col=0)
    cimis_df = utils.process_cimis(CIMIS_DATAFILE)

    wp_df = utils.process_tower(WESTPOND_DATAFILE, prefix="wp")
    mb_df = utils.process_tower(MAYBERRY_DATAFILE, prefix="mb")

    wp_rf_vgi_df = utils.process_modis_reflectance_veg_index(WESTPOND_REFL_VEG_IDX, prefix="wp")
    mb_rf_vgi_df = utils.process_modis_reflectance_veg_index(MAYBERRY_REFL_VEG_IDX, prefix="mb")

    wp_lst_em_df = utils.process_modis_lst_emissivity(WESTPOND_LST_EM, prefix="wp")
    mb_lst_em_df = utils.process_modis_lst_emissivity(MAYBERRY_LST_EM, prefix="mb")
    df = utils.merge_dataframes([cimis_df,
                            wp_df,
                            mb_df,
                            wp_rf_vgi_df,
                            mb_rf_vgi_df,
                            wp_lst_em_df,
                            mb_lst_em_df])
    df.to_csv(REL_PATH + save_filename)
    return df

def get_wp_energy_data(save_filename="data/wp_energy.csv"):
    if os.path.isfile(REL_PATH + save_filename):
        return pd.read_csv(REL_PATH + save_filename, index_col=0)
    cimis_df = utils.process_cimis(CIMIS_DATAFILE)

    wp_df = utils.process_tower(WESTPOND_DATAFILE, prefix="wp")
    wp_rf_vgi_df = utils.process_modis_reflectance_veg_index(WESTPOND_REFL_VEG_IDX, prefix="wp")
    wp_lst_em_df = utils.process_modis_lst_emissivity(WESTPOND_LST_EM, prefix="wp")

    tower_lwi = utils.process_tower_lwi(TOWER_LWI, interpolate_missing=True)
    df = utils.merge_dataframes([cimis_df,
                                wp_df,
                                wp_rf_vgi_df,
                                wp_lst_em_df,
                                tower_lwi])
    df.to_csv(REL_PATH + save_filename)
    return df
