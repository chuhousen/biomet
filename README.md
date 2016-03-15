# Classifying CO2 and CH4 Fluxes using CIMIS and MODIS data

####CIMIS DATA
1. solar radiation
2. net radiation
3. max air temperature
4. min air temperature
5. avg air temperature
6. max soil temperature
7. min soil temperature
8. avg soil temperature

####NDVI DATA
1. ndvi (every 16 days)

####WETLAND
1. co2_gf
2. ch4_gf

The WETLAND data and CIMIS data was averaged over consecutive 16 day periods and merged with the NDVI data before training.

---

### Installing
Highly recommended `virutalenv`: `pip install virtualenv`

```
git clone https://github.com/bsuper/biomet.git
cd biomet
virtualenv venv
pip install -r requirements.txt
cp [your WP_2012195to2015126_L3.mat location] input
```

### Running
```
python model.py
```
or

```
jupyter notebook
```
[Example](https://github.com/bsuper/biomet/blob/master/model.ipynb)
