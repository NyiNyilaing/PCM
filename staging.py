import click
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import numpy as np

# -------------------------
# Utility Functions
# -------------------------
def convert_inches_cube_to_metre_cube(volume):
    return volume * 1.63871 / 10**5

def convert_to_kilo_watt_hour(energy):
    return energy / (3600 * 1000)

def fill_null_values(series):
    return series.interpolate(method="linear").ffill().bfill()

# -------------------------
# Sample Class
# -------------------------
class Sample:
    def __init__(self, code, name, mass, melting_point, latent_heat,
                 volume_ratio, radius, height, onset_temp, T_inf, t_air=None, t_water=None):
        self.code = code
        self.name = name
        self.mass = mass
        self.density = None
        self.melting_point = melting_point
        self.latent_heat = latent_heat
        self.volume_ratio = volume_ratio
        self.radius = radius
        self.height = height
        self.mcomposition = None
        self.energy_density = None
        self.energy_storage_cost = None
        self.volume = None
        self.t_air = t_air
        self.t_water = t_water
        self.heat_capacity = None
        self.thermal_conductivity = None
        self.onset_temp = onset_temp
        self.temp_inf = T_inf

    def calc_sample_volume(self):
        self.volume = math.pi * self.radius**2 * self.height
        return self.volume

    def calc_sample_mass(self, pa_rho, g_rho):
        pa_mass_perc = self.volume_ratio[0] * pa_rho
        g_mass_perc = self.volume_ratio[1] * g_rho
        mass_factor = self.mass / (pa_mass_perc + g_mass_perc)
        pa_mass = mass_factor * pa_mass_perc
        g_mass = mass_factor * g_mass_perc
        self.mcomposition = (pa_mass, g_mass)
        return self.mcomposition

    def calc_energy_storage_cost(self, pa_cost, g_cost):
        Q_total = ((self.heat_capacity/1000) * (self.onset_temp - self.temp_inf)) + (self.latent_heat) 
        Q_kwh = convert_to_kilo_watt_hour(Q_total)      
        USD_exchange = 2000  
        self.energy_storage_cost = (pa_cost * self.mcomposition[0] +
                                    g_cost * self.mcomposition[1]) / (self.mass * Q_kwh * USD_exchange)
        return self.energy_storage_cost

    def calc_energy_density(self):
        sample_volume_m3 = convert_inches_cube_to_metre_cube(self.volume)
        Q_total = ((self.heat_capacity/1000) * (self.onset_temp - self.temp_inf)) + (self.latent_heat) 
        Q_kwh = convert_to_kilo_watt_hour(Q_total)
        self.energy_density = self.mass * Q_kwh / sample_volume_m3
        return self.energy_density
    
    def calc_rho_mix(self, pa_rho, g_rho):
        self.density = (self.mcomposition[0]/self.mass) * pa_rho + (self.mcomposition[1]/self.mass) * g_rho
        return self.density
    
    def calc_Cp_mix(self, pa_Cp, g_Cp):
        self.heat_capacity = (self.mcomposition[0]/self.mass) * pa_Cp + (self.mcomposition[1]/self.mass) * g_Cp
        return self.heat_capacity
    
    def calc_K_eff(self):
        R = self.radius * 0.0254 
        lambda1 = 2.405   
        gamma = self.calc_curve_fit(True)
        alpha = gamma * R**2 / (lambda1**2)
        self.thermal_conductivity = alpha * self.density* 1000 * self.heat_capacity
        return self.thermal_conductivity
    
    def calc_curve_fit(self, flag=False):
        t_exp = np.array(list(self.t_water.to_dict().keys())) * 60  # convert to seconds
        T_exp = np.array(list(self.t_water.to_dict().values()))        
        T_inf = 28.0
        Theta = (T_exp - T_inf) / (T_exp[0] - T_inf)
        
        def exp_model(t, gamma, C):
            return C * np.exp(-gamma * t)
        
        mask = t_exp >= 600
        popt, _ = curve_fit(exp_model, t_exp[mask], Theta[mask], p0=[1e-3, 1])
        gamma, Cfit = popt
        if flag is True:
            return gamma
        else:
            return t_exp, Theta, popt
        

# -------------------------
# Load Data + Init Samples
# -------------------------
def load_samples():
    pa_cost, g_cost = 70, 130     # kyats 
    pa_rho, g_rho = 0.856, 2.266  # g/cm3
    pa_Cp, g_Cp= 2000.0, 700.0    # J/kgK
    sample_mass = 80              # grams
    sample_radius = 1             # inches
    sample_height = 2             # inches
    T_inf = 28                    # °C
    onset_temp = [ 48.99, 59.38, 51.92, 53.89]
    melting_point = [68.63, 67.06, 68.59, 67.34]
    latent_heat = [288.2254, 169.7263, 182.9343, 156.3361]
    volume_ratio = [(100,0),(90,10),(85,15),(80,20)]
    name = ["PA(100%)","PA-G(90%-10%)","PA-G(85%-15%)","PA-G(80%-20%)"]
    code = ['sample_01','sample_02','sample_03','sample_04']

    # Read data
    t_air = pd.read_csv("sample_temp_air_hist_data.csv")
    t_water = pd.read_csv("sample_temp_water_hist_data.csv")

    samples = []
    for i in range(4):
        s = Sample(code[i], name[i], sample_mass, melting_point[i],
                   latent_heat[i], volume_ratio[i], sample_radius, sample_height, onset_temp[i], T_inf,
                   fill_null_values(t_air.iloc[:, i+1]),
                   fill_null_values(t_water.iloc[:, i+1]))
        s.calc_sample_volume()
        s.calc_sample_mass(pa_rho, g_rho)
        s.calc_rho_mix(pa_rho, g_rho)
        s.calc_Cp_mix(pa_Cp, g_Cp)
        s.calc_energy_density()
        s.calc_energy_storage_cost(pa_cost, g_cost)
        s.calc_K_eff()
        samples.append(s)
    return samples


# -------------------------
# Plot Functions
# -------------------------
def plot_bar(samples, attr, ylabel):
    labels = [s.code for s in samples]
    values = [getattr(s, attr) for s in samples]
    bars = plt.bar(labels, values, color='lightgreen')
    plt.ylabel(f'{ylabel}')
    plt.title(attr.replace("_", " ").title())
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  
            height,                            
            f'{height:.2f} kWh',                   
            ha='center', va='bottom'           
        )
    plt.show()

def plot_temp_hist(samples, attr):
    for s in samples:
        series = getattr(s, attr)
        plt.plot(series.index, series.values, label=s.name)
    plt.xlabel("Time (min)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"{attr} history")
    plt.legend()
    plt.show()
    
def plot_lumped(samples):
    def exp_model(t, gamma, C):
        return C * np.exp(-gamma * t)
    
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    for idx, s in enumerate(samples):
        t_exp, Theta, popt = s.calc_curve_fit()
        i,j = idx//2, 0
        if idx % 2 != 0:
            j = 1
        axes[i,j].scatter(t_exp/60, Theta, label="Experiment (normalized)")
        axes[i,j].plot(t_exp/60, exp_model(t_exp, *popt), 'r--', label="Exponential fit")
        axes[i,j].set_xlabel("Time (min)")
        axes[i,j].set_ylabel("Normalized Temperature θ")
        axes[i,j].set_title("Cooling curve fit - Sample_0%d" %(idx+1))
        axes[i,j].legend()
        axes[i,j].grid(True)
    plt.tight_layout(); 
    plt.show()

@click.command()
@click.option("--plot", multiple=True,
              help="Plot type: melting_point, energy_storage_cost, energy_density, thermal_conductivity, t_air, t_water")
@click.option("--expo", is_flag=True, help="Run exponential curve fit")
def main(plot, expo):
    samples = load_samples()

    # Handle individual plots
    for p in plot:
        if p in ["melting_point", "energy_storage_cost", "energy_density","thermal_conductivity"]:
            plot_bar(samples, p, p)
        elif p in ["t_air", "t_water"]:
            plot_temp_hist(samples, p)

    if expo:
        plot_lumped(samples)
            
if __name__ == "__main__":
    main()


