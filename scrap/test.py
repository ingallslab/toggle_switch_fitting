# Importing Libraries
from matplotlib import colors 
from matplotlib.ticker import PercentFormatter 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import statistics as st
import math

def modify_genetic_toggle_results(spreadsheet, precent_aTc):
    spreadsheet["Log(RFP/GFP)"]=np.log(spreadsheet["Intensity_MeanIntensity_RFP"]/spreadsheet["Intensity_MeanIntensity_GFP"])
    spreadsheet["Cell Number"]=spreadsheet["ObjectNumber"]+(spreadsheet["ImageNumber"]/10)
    spreadsheet["% aTc"] = precent_aTc
    
    return spreadsheet

#Import Files
febuary_9_aTc_aTc=pd.read_csv("Feb_09_Results_czi/Feb_9_aTc_aTc_czi.csv")
febuary_9_IPTG_aTc=pd.read_csv("Feb_09_Results_czi/Feb_9_IPTG_aTc_czi.csv")
febuary_9_IPTG_IPTG=pd.read_csv("Feb_09_Results_czi/Feb_9_IPTG_IPTG_czi.csv")
febuary_9_aTc_IPTG=pd.read_csv("Feb_09_Results_czi/Feb_9_aTc_IPTG_czi.csv")

modify_genetic_toggle_results(febuary_9_aTc_aTc, 100)
modify_genetic_toggle_results(febuary_9_IPTG_aTc, 100)
modify_genetic_toggle_results(febuary_9_IPTG_IPTG, 0)
modify_genetic_toggle_results(febuary_9_aTc_IPTG, 0)

Feb_9_ = pd.concat([febuary_9_aTc_aTc,febuary_9_IPTG_aTc,febuary_9_IPTG_IPTG,febuary_9_aTc_IPTG],ignore_index=True)

january_20_25_aTc=pd.read_csv("Jan_20_Results_czi/Jan_20_25%_aTc_czi.csv")
january_20_50_aTc=pd.read_csv("Jan_20_Results_czi/Jan_20_50%_aTc_czi.csv")
january_20_75_aTc=pd.read_csv("Jan_20_Results_czi/Jan_20_75%_aTc_czi.csv")

modify_genetic_toggle_results(january_20_25_aTc, 25)
modify_genetic_toggle_results(january_20_50_aTc, 50)
modify_genetic_toggle_results(january_20_75_aTc, 75)

Jan_20_ = pd.concat([january_20_25_aTc,january_20_50_aTc,january_20_75_aTc],ignore_index=True)


january_25_5_aTc=pd.read_csv("Jan_25_Results_czi/Jan_25_5%_aTc_czi.csv")
january_25_10_aTc=pd.read_csv("Jan_25_Results_czi/Jan_25_10%_aTc_czi.csv")
january_25_15_aTc=pd.read_csv("Jan_25_Results_czi/Jan_25_15%_aTc_czi.csv")
january_25_20_aTc=pd.read_csv("Jan_25_Results_czi/Jan_25_20%_aTc_czi.csv")

modify_genetic_toggle_results(january_25_5_aTc, 5)
modify_genetic_toggle_results(january_25_10_aTc, 10)
modify_genetic_toggle_results(january_25_15_aTc, 15)
modify_genetic_toggle_results(january_25_20_aTc, 20)

Jan_25_ = pd.concat([january_25_5_aTc,january_25_10_aTc,january_25_15_aTc,january_25_20_aTc],ignore_index=True)


january_26_11_aTc=pd.read_csv("Jan_26_Results_czi/Jan_26_11%_aTc_czi.csv")
january_26_13_aTc=pd.read_csv("Jan_26_Results_czi/Jan_26_13%_aTc_czi.csv")
january_26_17_aTc=pd.read_csv("Jan_26_Results_czi/Jan_26_17%_aTc_czi.csv")
january_26_19_aTc=pd.read_csv("Jan_26_Results_czi/Jan_26_19%_aTc_czi.csv")

modify_genetic_toggle_results(january_26_11_aTc, 11)
modify_genetic_toggle_results(january_26_13_aTc, 13)
modify_genetic_toggle_results(january_26_17_aTc, 17)
modify_genetic_toggle_results(january_26_19_aTc, 19)

Jan_26_ = pd.concat([january_26_11_aTc,january_26_13_aTc,january_26_17_aTc,january_26_19_aTc],ignore_index=True)


febuary_5_2_aTc=pd.read_csv("Feb_05_Results_czi/Feb_5_2%_aTc_czi.csv")
febuary_5_4_aTc=pd.read_csv("Feb_05_Results_czi/Feb_5_4%_aTc_czi.csv")
febuary_5_6_aTc=pd.read_csv("Feb_05_Results_czi/Feb_5_6%_aTc_czi.csv")

modify_genetic_toggle_results(febuary_5_2_aTc, 2)
modify_genetic_toggle_results(febuary_5_4_aTc, 4)
modify_genetic_toggle_results(febuary_5_6_aTc, 6)

Feb_5_= pd.concat([febuary_5_2_aTc,febuary_5_4_aTc,febuary_5_6_aTc],ignore_index=True)

febuary_8_5_aTc=pd.read_csv("Feb_08_Results_czi/Feb_8_5%_aTc_czi.csv")
febuary_8_10_aTc=pd.read_csv("Feb_08_Results_czi/Feb_8_10%_aTc_czi.csv")
febuary_8_15_aTc=pd.read_csv("Feb_08_Results_czi/Feb_8_15%_aTc_czi.csv")

modify_genetic_toggle_results(febuary_8_5_aTc, 5)
modify_genetic_toggle_results(febuary_8_10_aTc, 10)
modify_genetic_toggle_results(febuary_8_15_aTc, 15)

Feb_8_= pd.concat([febuary_8_5_aTc,febuary_8_10_aTc,febuary_8_15_aTc],ignore_index=True)

Final_Results = pd.concat([Feb_9_, Feb_8_, Feb_5_, Jan_26_, Jan_25_, Jan_20_],ignore_index=True)
Final_Results