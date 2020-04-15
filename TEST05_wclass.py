# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:16:38 2019

Grass Growth Model Adapted for urban areas
Developed by: Elton V. Escobar
              Vandoir Bourscheidt
              
Under the Grants: FAPESP and CNPq

Based on ...

Necessary inputs
Date    The date                                #
TMin    Minimum Temperature of the day (°C)     #
TMax    Maximum Temperature of the day (°C)     #
TMean   Average Temperature of the day (°C)     ## minimum necessary variables
RH      The relative humidity of the day (%)    #
Rain    Daily acumulated rain (mm)              #
Prev    Previous day data                       #

Other Basic Inputs and default values
LAI0 = 0.5      # is the inicial LAI after planting or harvesting. 
PHU = 3300      # is the number of the total heat units required for dormancy or senescence (heat units). 
                # The value of PHU depends on a particular vegetation and it must be provided by the user.
k_l = 1.1       # The light extinction coefficient. 
RUE = 1.6       # The vegetation radiation-use efficiency. 
HUI_1 = 0.22    # The fraction of growing season (i.e. fraction of total potential heat units) 
                # corresponding to the 1st point on the optimal leaf area development curve. 
HUI_2 = 0.36    # is the fraction of growing season corresponding to the 2st point on the optimal 
                # leaf area development curve.
fr_LAI1 = 0.13  # is the fraction of the maximum plant leaf area corresponding 
                # to the 1st point on the optimal leaf area development curve and is usually provided
fr_LAI2 = 0.44  # is the fraction of the maximum plant leaf area corresponding 
                # to the 2nd point on the optimal leaf area development curve and is usually provided 
dLAI = 0.80     # the fraction of the growing season when the leaf area index starts declining
                # It is provided by the user of the model
T_MIN = 12      # The minimal temperature (°C) for growth
T_MAX = 45      # The maximal temperature (°C) for growth
T_OPT1 = 30     # The first optimal temperature (°C) for growth
T_OPT2 = 40     # The second optimal temperatures (°C) for growth
can_MAX = 0.5   # can_MAX is is the maximum amount of water that can be trapped 
                # in the canopy when the canopy is fully developed
LAI_MAX = 3.2    # The value of LAI_MAX depends on a particular vegetation and is provided by the user of the model
LAI_MIN = 0.80  # The value of LAI_MIN depends on a particular vegetation and is provided by the user of the model

Other variables
Ra              # is the extraterrestrial radiation (MJ m-2 day-1)
HU              # Heat Units
AHU             # Is the sum oh heat units
AHU_dor         # AHU adjusted for dormance period
HUI             # Is the fraction between AHU and PHU (maximum HU acummulated before dormance)
H_phosy         # Is the intercepted photosynthetic active radiation

prev_data       # Data from previous run, necessary for the continuity of the model

LAI = np.zeros((1096))
PG = np.zeros((1096))
PLAI = np.zeros((1096))
fr_mx = np.zeros((1096))
LAI_act = np.zeros((1096))
RG = np.zeros((1096))
Biomass = np.zeros((1096))
Residue = np.zeros((1096))
EF = np.zeros((1096))
T_str = np.zeros((1096))
W_str = np.zeros((1096))
Rad_str = np.zeros((1096))

T_F = np.zeros((1096))
W_F = np.zeros((1096))
Rad_F = np.zeros((1096))
 
ET0 = np.zeros((1096))              # ET0 is the Potential Evapotranspiration
can = np.zeros((1096))              # it is the maximum amount of water that can be trapped in the canopy on day i 
Canopy_water_i = np.zeros((1096))   # it is the initial amount of free water held in the canopy on day i
Canopy_water_f = np.zeros((1096))   # it is the final amount of free water held in the canopy on day i
E_can = np.zeros((1096))            # it is the amount of evaporation (mm)  from free water in the canopy on day i
ET_adj = np.zeros((1096))           # the remaining evaporative water demandit or the potential evapotranspiration adjusted for evaporation of free water in the canopy
Etp = np.zeros((1096))              # Etp is the maximum plant transpiration on day i (mm) 


soil_water_i = np.zeros((1096))     # Amount of available water in the soil at the beginning of the day
soil_water_f = np.zeros((1096))     # Amount of available water in the soil at the end of the day

                 
r = np.zeros((1096))
t = np.zeros((1096))
    
"""

class Growth_model():
    def __init__(self,Date, TMin,TMax,TMean,RH,Rain,   # minimum necessary variables
                 LAI0=None,PHU=None,k_l=None,RUE=None,HUI_1=None, SWi=None,SWf=None,    #
                 HUI_2=None,fr_LAI1=None,fr_LAI2=None,dLAI=None,T_MIN=None,             #  aditional
                 T_MAX=None,T_OPT1=None,T_OPT2=None,can_MAX=None, LAI=None,LAIdor=None, #  variables
                 LAI_MIN=None,LAI_MAX=None,ET0=None,RAD=None,LAT=None,FDAYS=None,       #  for 
                 FDAYW=None,AHU=None,CANW =None,BIOM=None,RES=None,HAND=None,           #  continuity
                 UHAND=None,FRMX=None,PRAIN=None,DD=None,AHUdor=None):          
                                              #
        self.Date = Date                #
        self.TMin = TMin                #
        self.TMax = TMax                #
        self.TMean = TMean              ## minimum necessary variables
        self.RH = RH                    #
        self.Rain = Rain                #
        self.LAI0 = float(0.80 if LAI0 is None else LAI0)               
        self.PHU = 3300 if PHU is None else PHU
        self.k_l = 1.1 if k_l is None else k_l
        self.RUE = 1.6 if RUE is None else RUE
        self.HUI_1 = 0.22 if HUI_1 is None else HUI_1
        self.HUI_2 = 0.36 if HUI_2 is None else HUI_2
        self.fr_LAI1 = 0.13 if fr_LAI1 is None else fr_LAI1
        self.fr_LAI2 = 0.44 if fr_LAI2 is None else fr_LAI2
        self.dLAI = 0.80 if dLAI is None else dLAI 
        self.T_MIN = 12 if T_MIN is None else T_MIN 
        self.T_MAX = 45 if T_MAX is None else T_MAX
        self.T_OPT1 = 32 if T_OPT1 is None else T_OPT1
        self.T_OPT2 = 40 if T_OPT2 is None else T_OPT2
        self.can_MAX = 0.5 if can_MAX is None else can_MAX
        self.LAI_MIN = 0.8 if LAI_MIN is None else LAI_MIN
        self.LAI_MAX = 3.2 if LAI_MAX is None else LAI_MAX
        self.LAT = -22.0 if LAT is None else LAT
        #get Julian Day from given date
        self.jday = self.get_date(Date).timetuple().tm_yday
        #if RAD not present in the dataset, use Solar_Rad method (ignoring clouds)
        self.RAD = self.Solar_Rad(self.jday,self.LAT) if RAD is None else RAD
        #if ET0 not present in the dataset, use HARGREAVES method
        self.ET0 = self.HARGREAVES_ETO(self.RAD,self.TMin,self.TMax,self.TMean) if ET0 is None else ET0
        self.useHAND = True if UHAND is None else UHAND
        self.HAND = 0.0 if HAND is None else HAND
        self.LAI = self.LAI0 if LAI is None else LAI
        self.LAIdor = self.LAI if LAIdor is None else LAIdor
        self.AHU = 0.0 if AHU is None else AHU
        self.AHUdor = 0.0 if AHUdor is None else AHUdor
        self.fr_mx = 0.0 if FRMX is None else FRMX
        self.soil_water = 0.0 if SWi is None else SWi
        self.Canopy_water = 0.0 if CANW is None else CANW
        self.Biomass = 0.0 if BIOM is None else BIOM
        self.Residue = 0.0 if RES is None else RES
        self.drydays = 0 if DD is None else DD
        self.Etp = 0.0
        self.W_str = 0.0
        self.T_str = 0.0
        self.R_str = 0.0
        self.firstdayS = True if FDAYS is None else FDAYS
        self.firstdayW = True if FDAYW is None else FDAYW

    def cut_reset(self):
        self.LAI = self.LAI0 #start again with initial LAI
        self.LAIdor = self.LAI0 #restart LAI for dormancy period
        self.AHU = 0.0 # heat units are restarted
        self.Canopy_water = 0.0 # if cut, no water on leafs
        self.Biomass = 0.0 # biomass will be removed
        self.Residue = 0.0 # the same for residual biomass
        
    def get_date(self, Date):
        import sys
        import datetime
        if not isinstance(Date, (datetime.datetime, np.datetime64, datetime.date)):
            date_patterns = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]
            for pattern in date_patterns:
                try:
                    self.Date = datetime.datetime.strptime(Date, pattern).date()
                    return self.Date
                except:
                    pass
            print("Date is not in expected format: %s" %(Date))
            sys.exit(0)
        else:
            self.Date = Date
            return Date

# Potential evapotranspiration calculed by HARGREAVES approach   
    def HARGREAVES_ETO(self, RAD, TMin, TMax, TMean): 
        # equation from ###
        # 2.45 is the value of the latent heat.
        # The value of the latent heat varies only slightly over normal temperature ranges. 
        # A single value may be taken (for T = 20 °C): l = 2.45 MJ kg-1
        Et_HARGREAVES = (0.0023 * RAD * np.power((TMax -
                            TMin),0.5) * (TMean + 17.8)) / 2.45 
        return Et_HARGREAVES
    
#calculating the solar radiation from the date and latitude
# Extraterrestrial radiation 
# <http://www.fao.org/3/X0490E/x0490e07.htm>
    def Solar_Rad(self, jday,LAT):
        G_sc = 0.0820   # G_sc is the solar constant (MJ m-2 min-1)
        LAT_Rad = (np.pi / 180.0) * LAT # get latitude in radians 
        # d_r is the inverse relative distance Earth-Sun
        d_r = 1.0 + 0.033 * np.cos(((2.0 * np.pi) / 365.0) * (jday + 1.0))
        # determine the solar declination
        Solar_dec = 0.409 * np.sin (((2 * np.pi  * (jday + 1)) /365.0) - 1.39) 
        # S_ang is the sunset hour angle (rad)
        Sunset_ang = np.arccos(-np.tan(LAT_Rad) * np.tan(Solar_dec))
        Ra = ((24.0*60.0)/np.pi) * G_sc * d_r * (Sunset_ang * np.sin(LAT_Rad)*np.sin(Solar_dec) 
                        + np.cos(LAT_Rad) * np.cos(Solar_dec) * np.sin(Sunset_ang))
        self.daylength = (24.0 / np.pi) * Sunset_ang  # Daylight hours
        return Ra

#determine the stress factor for Temperature (T_str)
    def Temp_Stress(self,Tmin, TMax, TMean):         
        if TMean <= self.T_MIN:
            self.T_str = 1.0        
        elif self.T_OPT1 <= TMean <= self.T_OPT2: 
            self.T_str = 0.0
        elif self.T_MIN < TMean < 32.0: # 32 <= TMedia <= 40 is the range of the optimal temperatures (°C)
            self.T_str = 1.0 - ((TMean - Tmin) / (self.T_OPT1 - Tmin))
        elif self.T_OPT2 < TMean < self.T_MAX:
            self.T_str = 1.0 - (TMean - TMax) / (self.T_OPT2 - TMax)
        elif TMean >= self.T_OPT2:
            self.T_str = 1.0
        return self.T_str
# determine the stress factor for Radiation (Rad_str)
    def Rad_Stress(self,RAD):
        if ((0.48 * RAD)) < 5.0:
            Rad_stress = 0.0
        else:               # using min to avoid values larger than 1
            Rad_stress = min(0.044 * (((0.48 * RAD)) - 5), 1.0)
        
        return Rad_stress

# calculating the potential evapotranspiration adjusted for 
# evaporation of free water in the canopy:
    def evapotransp(self,LAI, Rain,Canopy_water,ET0):
        # determining the amount of free water held in the canopy 
        #from Rainfall (Precipitation) 
        can = self.can_MAX * (LAI / self.LAI_MAX)
        Canopy_water = min(can - Canopy_water,Rain)
            
        ## Evaporation of  water held in the canopy    
        Canopy_water = max(Canopy_water - ET0,0)
            
        # calculating the potential evapotranspiration adjusted for 
        # evaporation of free water in the canopy:
        ET_adj = max(0,ET0 - Canopy_water)       
        self.Canopy_water = Canopy_water #update the value for next day
            
        if 0 <= LAI <= 3:
            self.Etp = (ET_adj * LAI) / 3 
        elif LAI> 3:
            self.Etp = ET_adj
        else:
            print(self.Date,LAI)       
        return self.Etp
        
# determine the stress factor for Water (W_str)
# *different conditions are considered for different previous rain rates
    def Water_Stress(self,rain, ETP, CanW,SoilW, LAI):
        # determining the water balance between evapotransp and precipitation:
        SoilW2 = rain*(1 - (np.exp(-0.5*LAI) - 0.05)) + SoilW #- CanW
        Ks = (np.tanh(5.0*(SoilW2-12.5)/25))/2 + 0.5
        if Ks > 0.98:
            Ks = 1        
        if Ks < 0.025:
            Ks = 0
        #ks = pow(0.9,SoilW)
        Etr = ETP*Ks
        soil_water = max(0,(SoilW2 - Etr)*0.85)
        self.soil_water = soil_water #update the final soil water value
        W_str = 1 - min(1,Ks)  
        return W_str
         
    # Plant Growth Model
    #@property     
    def plant_growth(self, mode):
        #evaluates the mode
        if mode=='SMI':
            rule1 = self.Date.month <6 or self.Date.month > 8
            rule2 = self.Date.month > 5 and self.Date.month < 9
        if mode=='Daylength':
            rule1 = self.daylength > 12
            rule2 = self.daylength <= 12        
        
        # get the count of dry days
        if self.Rain > 0:
            drydays = 0
        elif self.Rain == 0:
            drydays = self.drydays + 1
        self.drydays = drydays
            
        #changes the HAND index depending on the number of drydays considering 
        # a maximum of six months until the base flow goes to zero (no more base flow)
        HANDI = self.HAND * (1 - drydays/90)
        self.drydays = drydays #update the value of drydays for next day
        #print(self.Rain,self.HANDI,self.drydays)
        # l1 and l2 (shape coefficients)
        l2 = (np.log((self.HUI_1 / self.fr_LAI1) - self.HUI_1) 
            - np.log((self.HUI_2 / self.fr_LAI2) - self.HUI_2)) / (self.HUI_2 - self.HUI_1)
        l1 = np.log(((self.HUI_1 / self.fr_LAI1) - self.HUI_1)) + l2 * self.HUI_1
        # Heat Units for each day resulting from mean temperature
        HU = max(self.TMean - 11.0,0.0)
        #create a boolean to check if it is the first day

        # rule 1: summer or rainy period
        if rule1:
            # check if is the first day of the model, returning to the initial condition
            if self.firstdayS:
                self.LAI = self.LAI0
                self.AHU = 0.0
                self.firstdayS = False
            
            AHU = self.AHU + HU
            HUI = AHU/self.PHU
            self.AHU = AHU #update the value of AHU for next day
            fr_mx = HUI / (HUI + np.exp((l1 - l2) * HUI))
            
            #determines potential LAI
            PLAI = (fr_mx - self.fr_mx) * self.LAI_MAX * (1 - np.exp(5 * (self.LAI0 - self.LAI_MAX)))
            PLAI = max(0,PLAI) # if PLAI < 0, then PLAI=0
            self.fr_mx = fr_mx #update the value of fr_mx for next day
            
            #calculating the evapotranspiration considering the water hold by the plant:
            self.Etp= self.evapotransp(self.LAI, self.Rain,self.Canopy_water,self.ET0)
            
            #calculating water stress:
            self.W_str = self.Water_Stress(self.Rain, self.Etp,self.Canopy_water,self.soil_water,self.LAI)
            
            #handi is the variation depending on the number of drydays
            if self.useHAND:
                self.W_str = 1-HANDI*self.W_str
            
            #temperature stress factor
            self.T_str = self.Temp_Stress(self.TMin,self.TMax,self.TMean)
            
            #radiation stress factor
            self.R_str = self.Rad_Stress(self.RAD)
            
            #choosing between the environmental stress factors:
            EF = max(0,1 - max(self.W_str,self.T_str,self.R_str))
            
            #applying the stress to get the actual LAI
            LAI_act = PLAI * np.sqrt(EF)
            LAI = LAI_act + self.LAI # sum with previous day LAI
            if self.AHU < self.PHU:
                H_phosy = 0.5 * (self.RAD / 1000) *(1 - np.exp(-self.k_l * LAI))  
                PG = self.RUE * H_phosy
                RG = PG * EF 
                self.Biomass = RG + self.Biomass
                self.Residue = self.Residue * 0.95
            self.LAI = LAI#update the value for next day
#            if np.isnan(LAI):
#                print(LAI)
            #self.LAI = max(LAI,self.LAI)
            self.firstdayW = True
        
        # rule 1: winter or dry period    
        if rule2:
            #print(self.Date,self.LAIdor,self.AHUdor)
            #remind the maximum LAI for decai curve
            # reset accumulated heat units to 0.0
            if self.firstdayW:
                self.LAIdor = self.LAI
                self.AHU = 0.0
                self.AHUdor = 0.0
                self.firstdayW = False
            
            AHU = self.AHUdor + HU
            HUI = AHU/self.PHU + 0.8
            self.AHUdor = AHU #update the value of AHU for next day
            self.AHU = AHU #update the value of AHU for next day
            
            #keep calculating ETP and other variables for the future :
            self.Etp= self.evapotransp(self.LAI, self.Rain,self.Canopy_water,self.ET0)

            ## logistic decay of LAI in cold/dry season
            rvalue = (1 - HUI) / (1 - self.dLAI)
            tvalue = -12 * (rvalue - 0.5)

            LAI = ((self.LAIdor - self.LAI_MIN) / (1 + np.exp(tvalue))) + self.LAI_MIN
            self.LAI = LAI#update the value for next day
            
            # reducing biomass due to decomposition at rate of 0.1%
            Biomass = self.Biomass * 0.9925
            # Conversion of biomass to residue 
            Residue = self.Residue + (self.Biomass - Biomass)
            self.Biomass = Biomass #update the value for next day
            self.Residue = Residue #update the value for next day
            self.firstdayS = True

            
########################################################################################

if __name__ == "__main__":
    #load all datasets
    import pandas as pd
    import numpy as np
    #import seaborn as sns
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import matplotlib.animation as manimation
    import seaborn as sns; sns.set()
    from matplotlib.colors import ListedColormap
    paleta = ListedColormap(sns.color_palette("ch:2,r=.2,l=.6").as_hex())
    #import cartopy
    
    filename = r'C:\Users\vando\OneDrive\ORIENTACOES\Mestrado\Elton\Weather DATA\1_Embrapa_diario_Oct2017_to_July2019.txt'

    dados = pd.read_csv(filename, skiprows=0,usecols=['Dia','TMedia','TMax','TMin','RHMedia','RHMax','RHMmin',\
        'Chuva','VMed','VMax','VDir','RadS','Eto'], sep=';', engine='python', parse_dates=True)


    dados['Data'] = pd.to_datetime(dados.Dia, format="%d-%m-%Y")
    
    db = r'C:\Users\vando\Google Drive\DIGITAL_GLOBE_GRANT\clip_Elton.shp'
    
    areas = gpd.read_file(db)
    lst_dict = []
    df1 = pd.DataFrame()
#    for idx, area in areas.iterrows():
#        areas['LAI'] = None
#        areasLAI = []
    
    for idx, area in areas.iterrows():
        HANDstr = area.Elev_exp
        for index, row in dados.iterrows():
            if index == 0:
                VG = Growth_model(row.Data,row.TMin,row.TMax,row.TMedia,
                                row.RHMedia,row.Chuva,ET0=row.Eto,UHAND=True,HAND=HANDstr)
                VG.plant_growth('SMI')
            else:
                #print(VG.drydays)
                VG = Growth_model(row.Data,row.TMin,row.TMax,row.TMedia,
                            row.RHMedia,row.Chuva,LAI=VG.LAI,AHU=VG.AHU,ET0=row.Eto,FDAYS=VG.firstdayS,
                            FDAYW=VG.firstdayW,CANW=VG.Canopy_water,LAIdor=VG.LAIdor,FRMX=VG.fr_mx,
                            BIOM=VG.Biomass, AHUdor=VG.AHUdor,DD=VG.drydays,UHAND=True,HAND=HANDstr)
                
                #apply the cutting if LAI exceed a threshold
                if VG.LAI > 2:
                    VG.cut_reset()
                VG.plant_growth('SMI')
                #print(VG)
            #append results to list
            lst_dict.append({'polID':area.name, 'Data':VG.Date, 'FRMX':VG.fr_mx,
                             'AHU': VG.AHU,'LAI': VG.LAI,'Biomass': VG.Biomass,
                             'W_str': VG.W_str,'ETP':VG.Etp,'DD':VG.drydays,
                             'T_str': VG.T_str,'R_str': VG.R_str})
        VG = None
    #create a dataframe with the results
    df4 = df1.append(lst_dict)
     
    #plot areas for eac day of the year in a movie
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Vegetation Growth Movie', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=2, metadata=metadata)
    
    fig = plt.figure(figsize=[10,7]) 
    with writer.saving(fig, "teste_crescimento2.mp4", 100):
        for dia in df4.Data.unique():
            grupo = df4.groupby('Data').get_group(dia)
            areas['LAI'] = grupo.LAI.values
            #print(areas.LAI.min(),areas.LAI.max())
            plt.clf()
            mapa = areas.plot(column='LAI',cmap=paleta, vmin=0.5, vmax=2.)
            plt.yticks(rotation=90,va='center')
            plt.title("LAI on Day: "+str(pd.to_datetime(np.datetime_as_string(dia)).date()))
            fig = mapa.get_figure()
            cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            sm = plt.cm.ScalarMappable(cmap=paleta, norm=plt.Normalize(vmin=0.5, vmax=2.))
            sm._A = []
            fig.colorbar(sm, cax=cax)
            #plt.savefig('D:\FigurasModelo\\'+str(pd.to_datetime(dia).strftime("%m-%d-%Y"))+'.png', dpi=200)
            plt.show()
            writer.grab_frame()
            plt.pause(0.1)
        plt.close()


    #df4.plot(x='Data',y=['LAI','W_str'])
#    plt.figure(figsize=(10,5))
#    plt.plot(dados.Data,LAI)
#    plt.ylabel('LAI')
#    plt.xlabel('Period (days)')  1.1609769532112562 1.1744347440417493
