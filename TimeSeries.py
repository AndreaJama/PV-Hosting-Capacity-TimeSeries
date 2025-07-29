import random
import py_dss_interface
import functions as fun
import time
import pandas as pd
import compile_fluxo as fluxo

# Initialize the OpenDSS interface
dss = py_dss_interface.DSSDLL()

# Inputs
dss_file = r"C:\123Bus\Run_IEEE123Bus.DSS"
curves_file = r"C:\Mestrado\Pruebas\CH_static\loadShape_Coelba_MT_dia_util.csv"
day = 'SOL'

start = time.time()

location = 1
NPFV_percentage = 100
hours = list(range(1, 25))

# List to store results for each hour
results = []

total_violations_daily_sum = 0

for hour in hours:
    # Run power flow for each hour
    num_OV, num_UV, num_SC, num_DT, buses_voltages, voltages_abc_pu, v_max, v_min, total_pv_p, total_pv_q, \
        total_losses_p_kw, total_p_kw, total_q_kvar, total_pv_p_dict, total_pv_q_dict = \
        fluxo.compile_fluxo(dss, dss_file, day, hour, curves_file, location, NPFV_percentage)

    # Store hourly data as a list of dictionaries
    results.append({
        "Location": location,
        "NPFV": NPFV_percentage,
        "Hour": hour,
        "V_max": v_max,
        "V_min": v_min,
        "Num_OV": num_OV,
        "Num_UV": num_UV,
        "Num_SC": num_SC,
        "buses_voltages": voltages_abc_pu,
        "total_pv_p": total_pv_p,
        "total_pv_q": total_pv_q,
        "feeder_kw": total_p_kw,
        "feeder_kvar": total_q_kvar,
        "pv_kw": total_pv_p_dict,
        "pv_kvar": total_pv_q_dict,
        "losses": total_losses_p_kw
    })

# Convert list of dictionaries to DataFrame
df_results = pd.DataFrame(results)

end = time.time()  # End of simulation time

print('\n Simulation time = ' + str(round(((end - start) / 60), 2)) + ' minutes')