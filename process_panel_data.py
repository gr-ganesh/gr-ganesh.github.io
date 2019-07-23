"""
- Date : 10-july-2019.
- Ganesh Mode GMOR Intern.
- Every value estimated is an over-estimate.

*Please run it on IT server*

##To Run File use following command
python36 process_panel_data.py args
three types of args
type1 args = 01 02 03 etc.... specify 1, 2 or more
type2 args = all
type3 args = all_each_panel

*In time_2_ans_bin column actual values are less than assigned value since ceil function is used*

*For example: ceil(16.5) = 17*

### Some categories for time_2_ans_bin

*21 is for all values above 20 mins*

*-1 is for all values less than 0*
"""


import sys
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

def fit_exp(x, a, b):
    return a*np.exp(-b*x)

if len(sys.argv) == 1:
    print("Please provide panel types....")
    print("Please look into file for more instructions")
    sys.exit(1)

load_t1 = time.time()

sas_path = 's3://data-analytics-data-transfer/ganesh_test/survey_answered_status_pandas.csv'
out_path_tfli = 's3://data-analytics-data-transfer/ganesh_test/result/tfli_'
out_path_dist = 's3://data-analytics-data-transfer/ganesh_test/result/distribution_'
out_path_par = 's3://data-analytics-data-transfer/ganesh_test/result/parameter_'


# please do changes accordingly if you more columns
dtypes = {'jn_monitor_id':'int64', 'panel_type':'str', 'research_id':'int64', 'time_2_ans':'int64'}
df_pd = pd.read_csv(sas_path,dtype=dtypes)
df_pd = df_pd[["jn_monitor_id","panel_type","research_id","time_2_ans"]]

df_csv = df_pd.copy()
df_csv["time_2_ans_min"] = df_csv["time_2_ans"]/60
df_csv["time_2_ans_bin"] = 0

df_csv.time_2_ans_bin = np.ceil(df_csv.time_2_ans_min)
df_csv.loc[df_csv["time_2_ans_min"]<=0,"time_2_ans_bin"] = -1
df_csv.loc[df_csv["time_2_ans_min"]>=20,"time_2_ans_bin"] = 21
load_t2 = time.time()
print("Time to load the data -> %s seconds ---" % (load_t2 - load_t1))


def panel_data_process(df_csv):
    df_copy = df_csv[["jn_monitor_id","panel_type","research_id","time_2_ans_bin"]]
    dt = datetime.today().strftime("%Y%m%d_%H%M%S")

    df_copy = df_copy.sort_values(by=["research_id"])

    df_new = df_copy[["research_id","time_2_ans_bin"]]

    df_gr = df_new.groupby("research_id")
    df_gq = df_gr.quantile([0.8])

    df_gq = df_gq["time_2_ans_bin"].values
    r_unq = np.unique(df_copy["research_id"])
    
    new_df = pd.DataFrame(columns=["research_id","allowed_research_time"])

    new_df.research_id = r_unq
    new_df.allowed_research_time = df_gq

    final = pd.merge(df_copy, new_df, on = "research_id")

    final = final.sort_values(by="jn_monitor_id")

    final_copy = pd.merge(df_copy, new_df, on = "research_id")
    final_copy_short = final_copy[["jn_monitor_id","time_2_ans_bin"]]

    final_gr = final_copy_short.groupby(["jn_monitor_id"])

    time_2_ans_50 = final_gr.median().values

    avg_ans_time = final_gr.mean().values

    max_ans_time = final_gr.max().values
    
    monitor_unique_id = np.unique(df_copy.jn_monitor_id)

    start_time = time.time()
    pref_ans_time = final_gr.quantile([0.8]).values
    end_time = time.time()
    print("Time to execute quantile function -> %s seconds ---" % (end_time - start_time))

    df_monitor_behaviour = pd.DataFrame(columns=["jn_monitor_id","max_ans_time","pref_ans_time_0.80","avg_ans_time","time_2_ans_0.50"])
    df_monitor_behaviour["jn_monitor_id"] = monitor_unique_id
    df_monitor_behaviour["max_ans_time"] = max_ans_time
    df_monitor_behaviour["pref_ans_time_0.80"] = pref_ans_time
    df_monitor_behaviour["time_2_ans_0.50"] = time_2_ans_50
    df_monitor_behaviour["avg_ans_time"] = avg_ans_time

    final_df = pd.merge(final, df_monitor_behaviour, on = "jn_monitor_id")

    bins = np.arange(-1,22,1)
    count,bins,patches = plt.hist(pref_ans_time,bins)

    xdata = bins[3:]
    ydata = count[2:]

    popt,pcov = curve_fit(fit_exp, xdata, ydata)

    # /usr/local/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: overflow encountered in exp
    # may arise because of pcov cannot be determined
    
    plt.plot(xdata, ydata, 'b-', label='data')
    plt.plot(xdata, fit_exp(xdata, *popt), 'g--', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

    df_panel = final_df[["panel_type","time_2_ans_bin"]]


    panel_gr = df_panel.groupby(["panel_type"])

    panel_description = panel_gr.describe(percentiles = [.2, .5, .8])
    
    tfli_out_path = out_path_tfli + dt + ".csv"
    final_df.to_csv(tfli_out_path)
    
    panel_desc_out_path = out_path_tfli + "panel_description" + dt + ".csv"
    panel_description.to_csv(panel_desc_out_path)
    
    return final_df,panel_description




#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------


def each_panel_process(df_csv, panel_list):
    num = len(panel_list)
    w, h = 20, num
    dt = datetime.today().strftime("%Y%m%d_%H%M%S")

    xdata = [[0 for x in range(w)] for y in range(h)]
    ydata = [[0 for x in range(w)] for y in range(h)]
    
    popt = np.zeros([h,2])

    for i in np.arange(0,num,1):
        temp_df = df_csv[df_pd.panel_type == panel_list[i]]

        df_copy = temp_df[["jn_monitor_id","panel_type","research_id","time_2_ans_bin"]]

        df_copy = df_copy.sort_values(by=["research_id"])

        df_new = df_copy[["research_id","time_2_ans_bin"]]

        df_gr = df_new.groupby("research_id")
        df_gq = df_gr.quantile([0.8])

        df_gv = df_gq["time_2_ans_bin"].values
        r_unq = np.unique(df_copy["research_id"])

        new_df = pd.DataFrame(columns=["research_id","allowed_research_time"])

        new_df.research_id = r_unq
        new_df.allowed_research_time = df_gv

        final = pd.merge(df_copy, new_df, on = "research_id")

        final = final.sort_values(by="jn_monitor_id")

        final_copy = pd.merge(df_copy, new_df, on = "research_id")
        final_copy_short = final_copy[["jn_monitor_id","time_2_ans_bin"]]

        final_gr = final_copy_short.groupby(["jn_monitor_id"])

        time_2_ans_50 = final_gr.median().values

        avg_ans_time = final_gr.mean().values

        max_ans_time = final_gr.max().values

        monitor_unique_id = np.unique(df_copy.jn_monitor_id)

        start_time = time.time()
        pref_ans_time = final_gr.quantile([0.8]).values
        end_time = time.time()
        print("--- Panel type %s took %s seconds to run. ---" % (panel_list[i] , (end_time - start_time)))

        df_monitor_behaviour = pd.DataFrame(columns=["jn_monitor_id","max_ans_time","pref_ans_time_0.80","avg_ans_time","time_2_ans_0.50"])
        df_monitor_behaviour["jn_monitor_id"] = monitor_unique_id
        df_monitor_behaviour["max_ans_time"] = max_ans_time
        df_monitor_behaviour["pref_ans_time_0.80"] = pref_ans_time
        df_monitor_behaviour["time_2_ans_0.50"] = time_2_ans_50
        df_monitor_behaviour["avg_ans_time"] = avg_ans_time

        final_df = pd.merge(final, df_monitor_behaviour, on = "jn_monitor_id")

        bins = np.arange(-1,22,1)
        count,bins,patches = plt.hist(pref_ans_time,bins)

        xdata[i] = bins[3:]
        ydata[i] = count[2:]
        
        tmp_hist_path = out_path_dist + panel_list[i] + "_histogram_" + dt + ".png"
        plt.savefig(tmp_hist_path)
        plt.clf()


    for i in np.arange(0,num,1):
        popt[i] = curve_fit(fit_exp, xdata[i], ydata[i])[0]
        plt.plot(xdata[i], ydata[i]/sum(ydata[i]), label='data')
        plt.plot(xdata[i], fit_exp(xdata[i], *popt[i]), label='fit: a=%5.3f, b=%5.3f' % tuple(popt[i]))
        
        tmp_fig_path = out_path_dist + panel_list[i] + "_fitted_plot_" + dt + ".png"
        plt.savefig(tmp_fig_path)
        plt.clf()
        


    # /usr/local/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: overflow encountered in exp
    # may arise because of pcov cannot be determined
    
    panel_description = pd.DataFrame(columns=["panel_list","initial_value","lambda_x"])
    panel_description.panel_list = panel_list
    panel_description.initial_value = popt[:,0]
    panel_description.lambda_x = popt[:,1]
    
    tfli_out_path = out_path_tfli + dt + ".csv"
    final_df.to_csv(tfli_out_path)
    
    par_out_path = out_path_par + dt + ".csv"
    panel_description.to_csv(par_out_path)
    
    return final_df, panel_description
    

#------------------------------------------------
#------------------------------------------------



if sys.argv[1] == 'all_each_panel':
    #panel_list = np.unique(df_csv.panel_type)    # all each panel
    panel_list = ['01', '02', '05', '07']         # only major four panels
    print("Only Major 4 panels =")
    print(panel_list)
    final_df,panel_description = each_panel_process(df_csv, panel_list)
elif sys.argv[1] == 'all':
    final_df,panel_description = panel_data_process(df_csv)
else:
    panel_list = sys.argv[1:]
    final_df,panel_description = each_panel_process(df_csv, panel_list)

    

load_t3 = time.time()

print ("final_df_head ->")
print (final_df.head())
print ("panel_description_head ->")
print (panel_description.head())
print ("---Time to run the script -> %s seconds ---" % (load_t3 - load_t1))