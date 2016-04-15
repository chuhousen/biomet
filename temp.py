# 4/13: NO QC VALUES IN DATA
# # replace bad values with NaN
# for col in ["Net Rad (W/sq.m)", "Avg Air Temp (C)", "Avg Soil Temp (C)"]:
#     qc = df.columns.get_loc(col) + 1
#     df.loc[df[df.columns[qc]] != ' ', col] = np.nan
