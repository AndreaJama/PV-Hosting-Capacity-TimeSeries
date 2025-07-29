# ⚡ IEEE123 Power Flow Simulator

This Python script performs hourly power flow simulations on the IEEE 123 Bus distribution system using OpenDSS via the `py_dss_interface`. It evaluates voltage profiles, PV generation impacts, and feeder-level power flows over a 24-hour period based on input load shapes.

---

## 🔍 Overview

This simulator:
- Loads a DSS master file for the IEEE 123 Bus system
- Uses real-world load shape curves for hourly simulation
- Performs power flow analysis for each hour (1–24)
- Detects voltage violations (overvoltage, undervoltage)
- Tracks feeder power flows, PV contributions, and system losses

---

## ✨ Key Features

- ✅ **24-Hour Simulation Loop**  
  Runs a complete daily simulation by hour

- ☀️ **PV Integration Assessment**  
  Simulates the effect of 100% non-programmable PV at a selected location

- 📊 **Result Storage with Pandas**  
  All hourly metrics are compiled into a DataFrame for further analysis

- 💡 **Voltage and Power Metrics**  
  Captures max/min voltages, PV power, total feeder flow, and losses

- ⏱️ **Execution Time Tracking**  
  Reports the total simulation time in minutes

---

## 📦 Python Packages Used

| Package            | Description                                       |
|-------------------|---------------------------------------------------|
| `py_dss_interface`| Python wrapper for OpenDSS simulations            |
| `pandas`           | Organizing and manipulating hourly results        |
| `random`           | (Optional) For introducing stochastic behavior    |
| `time`             | Tracking runtime and timestamps                   |
| `functions`        | Custom helper methods for internal operations     |
| `compile_fluxo`    | Executes hourly simulation with specified inputs  |

---

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ieee123-powerflow-simulator.git
