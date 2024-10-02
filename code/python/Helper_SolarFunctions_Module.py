### Solar Helper Functions Module ###

# importing External Modules
import os

import numpy as np
import pandas as pd

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import ssl

import tkinter as tk
from tkinter import filedialog


###############################################################################################

# Module Functions

###############################################################################################

###############################################################################################
# Module Functions : Solar Geometry
###############################################################################################

def alti_azi(dec, L, H):
    """
    Calculate altitude (beta) and azimuth (phi) based on declination, latitude, and hour angle.

    Parameters:
    dec (float or array-like): Declination in degrees.
    L (float): Latitude in degrees.
    H (array-like): Hour angles in degrees.

    Returns:
    tuple: Tuple containing:
        - beta (ndarray): Altitude in degrees.
        - phi (ndarray): Azimuth in degrees.
    """
    dec = np.radians(dec)
    L = np.radians(L)
    H = np.radians(H)
    
    beta = np.degrees(np.arcsin(np.cos(L) * np.cos(dec) * np.cos(H) + np.sin(L) * np.sin(dec)))
    
    azi1 = np.degrees(np.arcsin(np.cos(dec) * np.sin(H) / np.cos(np.radians(beta))))
    azi11 = np.abs(azi1)
    azi2 = 180 - azi11
    azi22 = np.abs(azi2)
    
    x = np.cos(H)
    y = np.tan(dec) / np.tan(L)
    
    phi = np.where(x >= y, azi1, np.where(azi1 >= 0, azi2, -azi2))
    
    return beta, phi

# Example usage:
# dec = 30.0
# L = 40.0
# H = [0.0, 15.0, 30.0]
# beta, phi = alti_azi(dec, L, H)
# print("Beta (Altitude):", beta)
# print("Phi (Azimuth):", phi)

def declination(n):
    """
    Calculate the solar declination angle for a given day of the year.

    Parameters:
    n (array-like): Day of the year (1-365).

    Returns:
    ndarray: Solar declination angle in degrees.
    """
    n = np.array(n)
    dec = 23.45 * np.sin(np.radians((360 / 365) * (n - 81)))

    return dec

# Example usage:
# n = [172, 173, 174]  # Example days of the year
# dec = declination(n)
# print("Declination angles:", dec)

def hour_angle(Hp):
    """
    Calculate the hour angle from the hour past solar noon.

    Parameters:
    Hp (array-like): Hours past solar noon.

    Returns:
    ndarray: Hour angle in degrees.
    """
    Hp = np.array(Hp)
    H = 15 * (12 - Hp)
    
    return H

# Example usage:
# Hp = [1, 2, 3, 4]  # Example hours past solar noon
# H = hour_angle(Hp)
# print("Hour angle:", H)

def beam_diff_ghi(n, GHI, beta):
    """
    Calculate the beam and diffuse components of the Global Horizontal Irradiance (GHI).

    Parameters:
    n (int): Day of the year (1-365).
    GHI (float): Global Horizontal Irradiance in W/m^2.
    beta (float): Solar altitude angle in degrees.

    Returns:
    tuple: Tuple containing:
        - Ib (float): Beam component of the GHI.
        - Id (float): Diffuse component of the GHI.
        - C (float): Correction factor based on the day of the year.
    """
    # Calculate the correction factor C
    C = 0.095 + (0.04 * np.sin((np.pi / 180) * (360 / 365) * (n - 100)))
    
    # Convert beta to radians for trigonometric calculations
    beta_rad = np.radians(beta)
    
    # Calculate the beam component of the GHI
    Ib = GHI / (np.sin(beta_rad) + C)
    
    # Calculate the diffuse component of the GHI
    Id = C * Ib
    
    return Ib, Id, C

# Example usage:
# n = 172  # Day of the year (e.g., 172 for June 21)
# GHI = 1000.0  # Global Horizontal Irradiance in W/m^2
# beta = 45.0  # Solar altitude angle in degrees
# Ib, Id, C = beam_diff_ghi(n, GHI, beta)
# print("Ib (Beam component):", Ib)
# print("Id (Diffuse component):", Id)
# print("C (Correction factor):", C)

def view_factor(beta, phi, tilt, phic):
    """
    Calculate the cosine of the incidence angle based on the given parameters.

    Parameters:
    beta (float): Solar altitude angle in degrees.
    phi (float): Solar azimuth angle in degrees.
    tilt (float): Tilt angle of the surface in degrees.
    phic (float): Azimuth angle of the surface in degrees.

    Returns:
    float: Cosine of the incidence angle.
    """
    beta_rad = np.radians(beta)
    phi_rad = np.radians(phi)
    tilt_rad = np.radians(tilt)
    phic_rad = np.radians(phic)
    
    CosInciAngle = (np.cos(beta_rad) * np.cos(phi_rad - phic_rad) * np.sin(tilt_rad)) + (np.sin(beta_rad) * np.cos(tilt_rad))
    
    return CosInciAngle

# Example usage:
# beta = 30.0  # Solar altitude angle in degrees
# phi = 180.0  # Solar azimuth angle in degrees
# tilt = 45.0  # Tilt angle of the surface in degrees
# phic = 180.0  # Azimuth angle of the surface in degrees
# CosInciAngle = view_factor(beta, phi, tilt, phic)
# print("Cosine of the incidence angle:", CosInciAngle)

def clock_to_solar_time(n, hem, Ltm, L, CT):
    """
    Convert clock time to solar time.

    Parameters:
    n (array-like): Day of the year (1-365).
    hem (int): Hemisphere indicator (+1 for Northern Hemisphere, -1 for Southern Hemisphere).
    Ltm (float): Local time meridian in degrees.
    L (float): Longitude in degrees.
    CT (array-like): Clock time in decimal hours.

    Returns:
    tuple: Tuple containing:
        - ST (ndarray): Solar time in decimal hours.
        - B (ndarray): Intermediate calculation based on the day of the year.
        - E (ndarray): Equation of time correction in minutes.
    """
    n = np.array(n)
    CT = np.array(CT)
    
    B = (360 / 364) * (n - 81)
    E = (9.87 * np.sin(np.radians(2 * B))) - (7.53 * np.cos(np.radians(B))) - (1.5 * np.sin(np.radians(B)))
    
    ST = np.zeros_like(CT, dtype=float)
    
    for i in range(len(n)):
        ST = CT - ((hem * (Ltm - L) * 4) / 60) + (E[i] / 60)
    
    return ST, B, E

# Example usage:
# n = [172]  # Day of the year (e.g., 172 for June 21)
# hem = 1  # Northern Hemisphere
# Ltm = 75.0  # Local time meridian in degrees
# L = 80.0  # Longitude in degrees
# CT = [12.0, 13.0, 14.0]  # Clock times in decimal hours
# ST, B, E = clock_to_solar_time(n, hem, Ltm, L, CT)
# print("ST (Solar time):", ST)
# print("B:", B)
# print("E (Equation of time correction):", E)

def solar_to_clock_time(n, hem, Ltm, L, ST):
    """
    Convert solar time to clock time.

    Parameters:
    n (array-like): Day of the year (1-365).
    hem (int): Hemisphere indicator (+1 for Northern Hemisphere, -1 for Southern Hemisphere).
    Ltm (float): Local time meridian in degrees.
    L (float): Longitude in degrees.
    ST (array-like): Solar time in decimal hours.

    Returns:
    tuple: Tuple containing:
        - CT (ndarray): Clock time in decimal hours.
        - B (ndarray): Intermediate calculation based on the day of the year.
        - E (ndarray): Equation of time correction in minutes.
    """
    n = np.array(n)
    ST = np.array(ST)
    
    B = (360 / 364) * (n - 81)
    E = (9.87 * np.sin(np.radians(2 * B))) - (7.53 * np.cos(np.radians(B))) - (1.5 * np.sin(np.radians(B)))
    
    CT = np.zeros_like(ST, dtype=float)
    
    for i in range(len(n)):
        for j in range(len(ST)):
            CT[j] = ST[j] + ((hem * (Ltm - L) * 4) / 60) - (E[i] / 60)
    
    return CT, B, E

# Example usage:
# n = [172]  # Day of the year (e.g., 172 for June 21)
# hem = 1  # Northern Hemisphere
# Ltm = 75.0  # Local time meridian in degrees
# L = 80.0  # Longitude in degrees
# ST = [12.0, 13.0, 14.0]  # Solar times in decimal hours
# CT, B, E = solar_to_clock_time(n, hem, Ltm, L, ST)
# print("Clock Time (CT):", CT)
# print("B:", B)
# print("E (Equation of time correction):", E)

def sun_rise_set(L, dec):
    """
    Calculate the sunrise and sunset times in solar time (decimal hours) for given latitudes and declinations.

    Parameters:
    L (float): Latitude in degrees.
    dec (array-like): Declinations in degrees.

    Returns:
    tuple: Tuple containing:
        - SunRise (ndarray): Sunrise times in decimal hours.
        - SunSet (ndarray): Sunset times in decimal hours.
        - Indicator (ndarray): Indicator for 24-hour sunlight or night conditions (1 for 24-hour sunlight, -1 for 24-hour night, 0 for normal sunrise and sunset).
    """
    dec = np.array(dec)
    len_dec = len(dec)
    
    SunRise = np.zeros(len_dec)
    SunSet = np.zeros(len_dec)
    Indicator = np.zeros(len_dec)
    
    for i in range(len_dec):
        Hsr = np.degrees(np.arccos(-(np.tan(np.radians(L)) * np.tan(np.radians(dec[i])))))
        hsr = np.abs(Hsr)
        Q = (3.467) / (np.cos(np.radians(L)) * np.cos(np.radians(dec[i])) * np.sin(np.radians(Hsr)))
        
        SunRise[i] = 12 - (hsr / 15) - (Q / 60)
        SunSet[i] = 12 + (hsr / 15) + (Q / 60)
        
        Sr = np.abs(np.imag(SunRise[i]))
        Ss = np.abs(np.imag(SunSet[i]))
        
        if (Sr > 0) and (Ss > 0):
            if (L > 0) and (dec[i] >= 0):
                Indicator[i] = 1  # 24 Hour Sunlight
            elif (L > 0) and (dec[i] <= 0):
                Indicator[i] = -1  # 24 Hour Night
            elif (L < 0) and (dec[i] <= 0):
                Indicator[i] = 1  # 24 Hour Sunlight
            elif (L < 0) and (dec[i] >= 0):
                Indicator[i] = -1  # 24 Hour Night
        else:
            Indicator[i] = 0  # Sunrise and Sunset are present
    
    return SunRise, SunSet, Indicator

# Example usage:
# L = 40.0  # Latitude in degrees
# dec = [23.45, 0.0, -23.45]  # Declinations in degrees
# SunRise, SunSet, Indicator = sun_rise_set(L, dec)
# print("SunRise:", SunRise)
# print("SunSet:", SunSet)
# print("Indicator:", Indicator)

###############################################################################################
# Module Functions : Solar to PV Production 
###############################################################################################


def array_incidence_loss(Ic, CosInciAngle, bo, SF):
    """
    Calculate the solar power available after incidence angle modification and soiling effect.

    Parameters:
    Ic (float): Initial solar power.
    CosInciAngle (float): Cosine of the incidence angle.
    bo (float): Incidence angle modifier coefficient.
    SF (float): Soiling factor as a percentage.

    Returns:
    tuple: Tuple containing:
        - Iciam (float): Solar power available after incidence angle modification.
        - Icsf (float): Solar power available for PV module after soiling effect.
    """
    # Calculate Incidence Angle Modifier
    Fiam = 1 - (bo * ((1 / CosInciAngle) - 1))
    
    # Solar Power Available after Incidence Angle Modification
    Iciam = Ic * Fiam
    
    # Solar Power Available For PV Module after Soiling Effect
    Icsf = Iciam * (1 - (SF / 100))
    
    return Iciam, Icsf

# Example usage:
# Ic = 1000.0  # Initial solar power in W/m^2
# CosInciAngle = 0.866  # Cosine of the incidence angle
# bo = 0.05  # Incidence angle modifier coefficient
# SF = 10.0  # Soiling factor in percentage
# Iciam, Icsf = array_incidence_loss(Ic, CosInciAngle, bo, SF)
# print("Iciam:", Iciam)

def fixed_tilt(Ib, Id, C, beta, phi, tilt, phic, rho):
    """
    Calculate the solar irradiance components on a fixed tilt surface.

    Parameters:
    Ib (float): Beam irradiance.
    Id (float): Diffuse irradiance.
    C (float): Correction factor.
    beta (float): Solar altitude angle in degrees.
    phi (float): Solar azimuth angle in degrees.
    tilt (float): Tilt angle of the surface in degrees.
    phic (float): Azimuth angle of the surface in degrees.
    rho (float): Reflectance of the ground.

    Returns:
    tuple: Tuple containing:
        - Ic (float): Total solar irradiance on the collector.
        - Ibc (float): Beam component on the collector.
        - Idc (float): Diffuse component on the collector.
        - Irc (float): Reflected component on the collector.
        - CosInciAngle (float): Cosine of the incidence angle.
    """
    # Convert angles from degrees to radians
    beta_rad = np.radians(beta)
    phi_rad = np.radians(phi)
    tilt_rad = np.radians(tilt)
    phic_rad = np.radians(phic)
    
    # Calculate Cosine of the Incidence Angle
    CosInciAngle = (np.cos(beta_rad) * np.cos(phi_rad - phic_rad) * np.sin(tilt_rad) +
                    np.sin(beta_rad) * np.cos(tilt_rad))
    
    # Calculate Beam Component on Collector
    Ibc = Ib * CosInciAngle
    
    # Calculate Diffuse Component on Collector
    Idc = Id * ((1 + np.cos(tilt_rad)) / 2)
    
    # Calculate Reflected Component on the Collector
    Irc = rho * Ib * (np.sin(beta_rad) + C) * ((1 - np.cos(tilt_rad)) / 2)
    
    # Calculate Total Solar Irradiance on the Collector
    Ic = Ibc + Idc + Irc
    
    return Ic, Ibc, Idc, Irc, CosInciAngle

# Example usage:
# Ib = 800.0  # Beam irradiance in W/m^2
# Id = 200.0  # Diffuse irradiance in W/m^2
# C = 0.1  # Correction factor
# beta = 30.0  # Solar altitude angle in degrees
# phi = 180.0  # Solar azimuth angle in degrees
# tilt = 45.0  # Tilt angle of the surface in degrees
# phic = 180.0  # Azimuth angle of the surface in degrees
# rho = 0.2  # Reflectance of the ground
# Ic, Ibc, Idc, Irc, CosInciAngle = fixed_tilt(Ib, Id, C, beta, phi, tilt, phic, rho)
# print("Total Solar Irradiance on the Collector (Ic):", Ic)
# print("Beam Component on the Collector (Ibc):", Ibc)
# print("Diffuse Component on the Collector (Idc):", Idc)
# print("Reflected Component on the Collector (Irc):", Irc)
# print("Cosine of the Incidence Angle (CosInciAngle):", CosInciAngle)

def module_power(Pmod, ModTemCF, ModNum, Tn, Gn, Icsf, T, Ic, Uo, U1, Ws):
    """
    Calculate the power generated by a photovoltaic module and the total power generated by multiple modules.

    Parameters:
    Pmod (float): Nominal power of the module.
    ModTemCF (float): Module temperature coefficient (in %).
    ModNum (int): Number of modules.
    Tn (float): Nominal temperature.
    Gn (float): Nominal irradiance.
    Icsf (float): Corrected solar irradiance.
    T (float): Ambient temperature.
    Ic (float): Incident solar radiation.
    Uo (float): Heat loss coefficient (W/m^2K).
    U1 (float): Wind-dependent heat loss coefficient (W/m^2K).
    Ws (float): Wind speed.

    Returns:
    tuple: Tuple containing:
        - Pmodtot (float): Total power generated by all modules.
        - Pmodin (float): Power generated by one module.
        - Tm (float): Module temperature.
    """
    # Calculate the module temperature using Faiman's model
    Tm = T + (Ic / (Uo + (U1 * Ws)))

    # Calculate the power generated by one module
    Pmodin = Pmod * (1 + ((ModTemCF / 100) * (Tm - Tn))) * (Icsf / Gn)

    # Calculate the total power generated by all modules
    Pmodtot = ModNum * Pmodin

    return Pmodtot, Pmodin, Tm

# Example usage:
# Pmod = 250.0  # Nominal power of the module in W
# ModTemCF = -0.5  # Module temperature coefficient in %
# ModNum = 10  # Number of modules
# Tn = 25.0  # Nominal temperature in °C
# Gn = 1000.0  # Nominal irradiance in W/m^2
# Icsf = 800.0  # Corrected solar irradiance in W/m^2
# T = 20.0  # Ambient temperature in °C
# Ic = 900.0  # Incident solar radiation in W/m^2
# Uo = 25.0  # Heat loss coefficient in W/m^2K
# U1 = 6.84  # Wind-dependent heat loss coefficient in W/m^2K
# Ws = 2.0  # Wind speed in m/s
# Pmodtot, Pmodin, Tm = module_power(Pmod, ModTemCF, ModNum, Tn, Gn, Icsf, T, Ic, Uo, U1, Ws)
# print("Total power generated by all modules (Pmodtot):", Pmodtot)
# print("Power generated by one module (Pmodin):", Pmodin)
# print("Module temperature (Tm):", Tm)

def pv_output_power(Pmodtot, LID, LS, Arraymismat, Crys, Shading, OhmicLoss, TrackerL, INVeff, TransLoss):
    """
    Calculate various power output and loss metrics for a photovoltaic (PV) system.

    Parameters:
    Pmodtot (float): Total power generated by the modules.
    LID (float): Light Induced Degradation (for Crystalline Modules) in %.
    LS (float): Light Soaking (for Thin Film Modules) in %.
    Arraymismat (float): Array Mismatch Factor in %.
    Crys (int): PV Technology Crystalline (1) or Thin Film (0).
    Shading (float): Shading Loss Factor in %.
    OhmicLoss (float): Array wiring loss in %.
    TrackerL (float): Tracker Loss Factor in %.
    INVeff (float): Inverter Efficiency in %.
    TransLoss (float): Transformer Loss Factor in %.

    Returns:
    tuple: Tuple containing:
        - PVout (float): Power output from the array.
        - INVpin (float): Power input to the inverter.
        - INVpout (float): Power output from the inverter.
        - Pgrid (float): Power output to the grid through the transformer.
        - ArrayMismatchLoss (float): Array mismatch loss.
        - ShadingLoss (float): Shading loss.
        - LIDLoss (float): Light induced degradation loss.
        - OhmicLossP (float): Ohmic loss.
        - InverterLoss (float): Inverter loss.
        - TransformerLossP (float): Transformer loss.
        - TrackerLossP (float): Tracker loss.
    """
    # Calculate power output from array
    if Crys == 1:
        PVout = Pmodtot * (1 - ((LID + Arraymismat + Shading) / 100))
    else:
        PVout = Pmodtot * (1 - ((Arraymismat + Shading) / 100) + (LS / 100))
    
    # Calculate power input to inverter
    INVpin = PVout * (1 - (OhmicLoss / 100))
    
    # Calculate power output from inverter
    INVpout = INVpin * (INVeff / 100)
    
    # Calculate power output after tracker loss
    TrackerLossPP = INVpout * (1 - (TrackerL / 100))
    
    # Calculate power output to grid through transformer
    Pgrid = TrackerLossPP * (1 - (TransLoss / 100))
    
    # Calculate power losses
    ArrayMismatchLoss = Pmodtot * (Arraymismat / 100)
    ShadingLoss = Pmodtot * (Shading / 100)
    LIDLoss = Pmodtot * (LID / 100)
    OhmicLossP = PVout * (OhmicLoss / 100)
    InverterLoss = INVpin * (1 - (INVeff / 100))
    TrackerLossP = INVpout * (TrackerL / 100)
    TransformerLossP = INVpout * (TransLoss / 100)
    
    return (PVout, INVpin, INVpout, Pgrid, ArrayMismatchLoss, ShadingLoss, 
            LIDLoss, OhmicLossP, InverterLoss, TrackerLossP, TransformerLossP)

# Example usage:
# Pmodtot = 1000.0  # Total power generated by the modules in W
# LID = 2.0  # Light Induced Degradation in %
# LS = 3.0  # Light Soaking in %
# Arraymismat = 2.0  # Array Mismatch Factor in %
# Crys = 1  # PV Technology Crystalline
# Shading = 1.0  # Shading Loss Factor in %
# OhmicLoss = 3.0  # Array wiring loss in %
# TrackerL = 1.0  # Tracker Loss Factor in %
# INVeff = 95.0  # Inverter Efficiency in %
# TransLoss = 1.0  # Transformer Loss Factor in %
# results = pv_output_power(Pmodtot, LID, LS, Arraymismat, Crys, Shading, OhmicLoss, TrackerL, INVeff, TransLoss)
# print("Results:", results)

###############################################################################################
# Module Functions : Data Cleaning and Formatting
###############################################################################################

def date_time_series_slicer(OriginalDataSeries, SeriesNum3, Res, StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay, StartTime, EndTime):
    """
    Slice a time series based on specified start and end dates and times.

    Parameters:
    OriginalDataSeries (ndarray): Original data series with datetime information.
    SeriesNum3 (int): Column index for the data series to extract (0-based index).
    Res (int): Time resolution in minutes.
    StartYear (int): Start year.
    EndYear (int): End year.
    StartMonth (int): Start month.
    EndMonth (int): End month.
    StartDay (int): Start day.
    EndDay (int): End day.
    StartTime (float): Start time in decimal hours.
    EndTime (float): End time in decimal hours.

    Returns:
    tuple: Tuple containing:
        - OriginalSeries (ndarray): Extracted data series.
        - StartIndex (int): Start index of the extracted data.
        - EndIndex (int): End index of the extracted data.
    """
    r, c = OriginalDataSeries.shape

    # Creating Day Time Vector based on the File Resolution
    DayVector = np.arange(0, 24, Res / 60)

    # Finding Time Value Indices within the Day Vector
    DiffStartTime = np.abs(StartTime - DayVector)
    IndexST = np.argmin(DiffStartTime)

    DiffEndTime = np.abs(EndTime - DayVector)
    IndexET = np.argmin(DiffEndTime)

    StartIndex = None
    EndIndex = None

    # Finding the Start Index
    for i in range(0, r, len(DayVector)):
        if (OriginalDataSeries[i, 2] == StartDay and
            OriginalDataSeries[i, 1] == StartMonth and
            OriginalDataSeries[i, 0] == StartYear):
            StartIndex = i + IndexST
            break

    # Finding the End Index
    for i in range(0, r, len(DayVector)):
        if (OriginalDataSeries[i, 2] == EndDay and
            OriginalDataSeries[i, 1] == EndMonth and
            OriginalDataSeries[i, 0] == EndYear):
            EndIndex = i + IndexET
            break

    if StartIndex is None or EndIndex is None:
        raise ValueError("Start or End index could not be found in the data series.")

    # Getting the OriginalSeries
    OriginalSeries = OriginalDataSeries[StartIndex:EndIndex+1, 3 + SeriesNum3]

    return OriginalSeries, StartIndex, EndIndex

# Example usage:
# OriginalDataSeries = np.array([
#     [2023, 6, 18, 0, 100],
#     [2023, 6, 18, 1, 101],
#     [2023, 6, 18, 2, 102],
#     # ... more data ...
#     [2023, 6, 19, 0, 200],
#     [2023, 6, 19, 1, 201],
#     [2023, 6, 19, 2, 202]
# ])
# SeriesNum3 = 0
# Res = 60
# StartYear = 2023
# EndYear = 2023
# StartMonth = 6
# EndMonth = 6
# StartDay = 18
# EndDay = 19
# StartTime = 0.0
# EndTime = 2.0
# OriginalSeries, StartIndex, EndIndex = date_time_series_slicer(OriginalDataSeries, SeriesNum3, Res, StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay, StartTime, EndTime)
# print("OriginalSeries:", OriginalSeries)
# print("StartIndex:", StartIndex)
# print("EndIndex:", EndIndex)


def date_time_series_slicer_pecan_street_data(OriginalDataSeries, SeriesNum3, Res, StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay, StartTime, EndTime):
    """
    Slice a time series based on specified start and end dates and times.

    Parameters:
    OriginalDataSeries (ndarray): Original data series with datetime information.
    SeriesNum3 (int): Column index for the data series to extract (0-based index).
    Res (int): Time resolution in minutes.
    StartYear (int): Start year.
    EndYear (int): End year.
    StartMonth (int): Start month.
    EndMonth (int): End month.
    StartDay (int): Start day.
    EndDay (int): End day.
    StartTime (float): Start time in decimal hours.
    EndTime (float): End time in decimal hours.

    Returns:
    tuple: Tuple containing:
        - OriginalSeries (ndarray or str): Extracted data series or 'Error' if an exception occurs.
        - StartIndex (int): Start index of the extracted data.
        - EndIndex (int): End index of the extracted data.
    """
    try:
        r, c = OriginalDataSeries.shape

        # Creating Day Time Vector based on the File Resolution
        DayVector = np.arange(0, 24, Res / 60)

        # Finding Time Value Indices within the Day Vector
        DiffStartTime = np.abs(StartTime - DayVector)
        IndexST = np.argmin(DiffStartTime)

        DiffEndTime = np.abs(EndTime - DayVector)
        IndexET = np.argmin(DiffEndTime)

        StartIndex = None
        EndIndex = None

        # Finding the Start Index
        for i in range(0, r, len(DayVector)):
            if (OriginalDataSeries[i, 2] == StartDay and
                OriginalDataSeries[i, 1] == StartMonth and
                OriginalDataSeries[i, 0] == StartYear):
                StartIndex = i + IndexST
                break

        # Finding the End Index
        for i in range(0, r, len(DayVector)):
            if (OriginalDataSeries[i, 2] == EndDay and
                OriginalDataSeries[i, 1] == EndMonth and
                OriginalDataSeries[i, 0] == EndYear):
                EndIndex = i + IndexET
                break

        if StartIndex is None or EndIndex is None:
            raise ValueError("Start or End index could not be found in the data series.")

        # Getting the OriginalSeries
        OriginalSeries = OriginalDataSeries[StartIndex:EndIndex+1, 3 + SeriesNum3]

        return OriginalSeries, StartIndex, EndIndex
    
    except Exception as e:
        print(f"DateTimeSlicer Function encountered an error: {e}")
        return 'Error', 0, 0

# Example usage:
# OriginalDataSeries = np.array([
#     [2023, 6, 18, 0, 100],
#     [2023, 6, 18, 1, 101],
#     [2023, 6, 18, 2, 102],
#     # ... more data ...
#     [2023, 6, 19, 0, 200],
#     [2023, 6, 19, 1, 201],
#     [2023, 6, 19, 2, 202]
# ])
# SeriesNum3 = 0
# Res = 60
# StartYear = 2023
# EndYear = 2023
# StartMonth = 6
# EndMonth = 6
# StartDay = 18
# EndDay = 19
# StartTime = 0.0
# EndTime = 2.0
# OriginalSeries, StartIndex, EndIndex = date_time_series_slicer_pecan_street_data(OriginalDataSeries, SeriesNum3, Res, StartYear, EndYear, StartMonth, EndMonth, StartDay, EndDay, StartTime, EndTime)
# print("OriginalSeries:", OriginalSeries)
# print("StartIndex:", StartIndex)
# print("EndIndex:", EndIndex)

def rows_cols_to_compute_data_cleaning(start_year, start_month, start_day, end_year, end_month, end_day, res, data_cols, date_time_cols):
    """
    Compute the total number of rows and columns needed for cleaning the data.

    Parameters:
    start_year (int): Start year of the data.
    start_month (int): Start month of the data.
    start_day (int): Start day of the data.
    end_year (int): End year of the data.
    end_month (int): End month of the data.
    end_day (int): End day of the data.
    res (int): Resolution of the data in minutes.
    data_cols (int): Number of data columns.
    date_time_cols (int): Number of date and time columns.

    Returns:
    tuple: Total rows, columns, and total days in the data set.
    """
    # Computing the total number of different years in the data
    num_of_years = end_year - start_year + 1

    # Computing the different year signature values
    years = [start_year + i for i in range(num_of_years)]

    # Finding the leap and non-leap years
    leap_years = [leap_year_finder(year) for year in years]

    # Initializing day counters
    a = 0
    b = 0
    c = np.zeros(num_of_years)

    # Computing number of days in the given data set
    for j in range(num_of_years):
        if j == 0:  # Days for start year
            if num_of_years == 1:
                start_day1, end_day1 = days_to_compute(leap_years[j], start_day, start_month, end_day, end_month)
                a = end_day1 - start_day1 + 1  # Total number of days
            else:
                start_day1, end_day1 = days_to_compute(leap_years[j], start_day, start_month, 31, 12)
                a = end_day1 - start_day1 + 1  # Total number of days
        elif j == num_of_years - 1:  # Days for end year
            start_day1, end_day1 = days_to_compute(leap_years[j], 1, 1, end_day, end_month)
            b = end_day1 - start_day1 + 1  # Total number of days
        else:  # Days for all other years
            start_day1, end_day1 = days_to_compute(leap_years[j], 1, 1, 31, 12)
            c[j] = end_day1 - start_day1 + 1  # Total number of days

    # Total number of days in the data set
    tot_days = a + b + sum(c)

    # Data points in one day (resolution has to be in minutes)
    data_points = 24 * (60 // res)

    # Total data points in the given data set i.e., the total number of rows
    rows = tot_days * data_points

    # Total number of columns in the data set
    cols = data_cols + date_time_cols

    return rows, cols, tot_days
    
# Example usage:
# rows, cols, tot_days = rows_cols_to_compute_data_cleaning(2020, 1, 1, 2020, 12, 31, 15, 5, 3)

def min_to_min_data_converter(data_cols, res_original, res_new, avg_or_add, headers):
    """
    Convert time series data from one resolution to another.

    Parameters:
    data_cols (int): Number of data columns in the input file.
    res_original (int): Original resolution of the data in minutes.
    res_new (int): New resolution of the data in minutes.
    avg_or_add (list): List of integers indicating whether to average (0) or add (1) for each data column.
    headers (int): 1 if the input file contains headers, 0 otherwise.

    Returns:
    pd.DataFrame: The processed data with new resolution.
    """
   

    root = tk.Tk()
    root.withdraw()

    # File selection dialog
    file_path = filedialog.askopenfilename(title="Select Raw Data File", filetypes=[("All files", "*.*")])
    processed_data = pd.read_excel(file_path, header=None)

    # Computing size of processed_data matrix
    row, col = processed_data.shape
    row_new = int(np.ceil(row * (res_original / res_new)))

    # Initializing the new processed data matrix
    processed_data1 = np.zeros((row_new, 4 + data_cols))

    # Computing number of rows to be averaged or added
    num_rows = res_new / res_original
    num_rows_frac = res_new % res_original

    if headers:
        data_file = pd.read_excel(file_path, header=None)
        header1 = data_file.iloc[0, 4:(4 + data_cols)].tolist()
        header = ['Day', 'Month', 'Year', 'Time'] + header1
    else:
        header = ['Day', 'Month', 'Year', 'Time']

    if num_rows_frac == 0:
        # Initializing index for processed_data1 matrix
        index1 = 1

        for i in range(1, row, int(num_rows)):
            processed_data1[0, :] = processed_data.iloc[0, :]

            if i == 1:
                continue

            for k in range(data_cols):
                indicator = avg_or_add[k]
                add = 0
                avg = np.zeros(int(num_rows))

                for j in range(int(num_rows)):
                    row_index = i + j - 1

                    if indicator == 1:
                        add += processed_data.iloc[row_index, k + 4]
                    elif indicator == 0:
                        avg[j] = processed_data.iloc[row_index, k + 4]

                processed_data1[index1, 0:4] = processed_data.iloc[row_index, 0:4]

                if indicator == 1:
                    processed_data1[index1, k + 4] = add
                elif indicator == 0:
                    processed_data1[index1, k + 4] = sum(avg) / num_rows

            index1 += 1

    elif num_rows_frac != 0:
        start_year = int(processed_data.iloc[0, 2])
        start_month = int(processed_data.iloc[0, 1])
        start_day = int(processed_data.iloc[0, 0])

        end_year = int(processed_data.iloc[row - 1, 2])
        end_month = int(processed_data.iloc[row - 1, 1])
        end_day = int(processed_data.iloc[row - 1, 0])

        rows1, cols1, tot_days = rows_cols_to_compute_data_cleaning(start_year, start_month, start_day, end_year, end_month, end_day, res_new, data_cols, 4)

        processed_data1 = np.zeros((rows1, cols1))
        date_time_matrix = start_end_calendar(start_year, start_month, start_day, tot_days, res_new, data_cols)
        
        processed_data1[:, 0:4] = date_time_matrix[:, 0:4]

        index1 = 0
        row_num_vector = []

        for i in range(1, row, int(num_rows)):
            index1 += 1
            row_num_vector.append(i)

            if i == 1:
                processed_data1[0, :] = processed_data.iloc[0, :]
                continue

            prev_row_num = row_num_vector[index1 - 2]
            int_frac_indicator2 = prev_row_num % 1

            if int_frac_indicator2 == 0:
                num_row_previous = prev_row_num + 1
                start_val_multiplier = 1
            else:
                num_row_previous = int(np.ceil(prev_row_num))
                start_val_multiplier = 1 - int_frac_indicator2

            int_frac_indicator1 = i % 1

            if int_frac_indicator1 == 0:
                num_row_next = i
                end_val_multiplier = 1
            else:
                num_row_next = int(np.ceil(i))
                end_val_multiplier = int_frac_indicator1

            actual_indices = list(range(num_row_previous, num_row_next + 1))

            for k in range(data_cols):
                indicator = avg_or_add[k]
                actual_values = processed_data.iloc[actual_indices, k + 4]

                if indicator == 1:
                    actual_values.iloc[0] *= start_val_multiplier
                    actual_values.iloc[-1] *= end_val_multiplier
                    addition_value = actual_values.sum()
                    processed_data1[index1, k + 4] = addition_value
                elif indicator == 0:
                    averaged_value = actual_values.sum() / len(actual_values)
                    processed_data1[index1, k + 4] = averaged_value

    filename = os.path.basename(file_path).split('_')[0]
    new_filename = f"{filename}_Converted_File_MinutesResolution_{res_original}-Mins_To_{res_new}-Mins.xlsx"

    with pd.ExcelWriter(new_filename) as writer:
        if headers:
            pd.DataFrame([header]).to_excel(writer, sheet_name='Sheet1', header=False, index=False, startrow=0)
        pd.DataFrame(processed_data1).to_excel(writer, sheet_name='Sheet1', header=False, index=False, startrow=1 if headers else 0)

    return pd.DataFrame(processed_data1)
    
def solar_pv_weather_data_cleaner_modified_for_pecan_street(res, data_cols, n, data_file):
    """
    Clean and format Solar PV weather data.

    Parameters:
    res (int): Time step of the data file in minutes.
    data_cols (int): Number of columns in data file which represents data, other than date and time columns.
    n (int): Number of points for averaging.
    data_file (pd.DataFrame): DataFrame containing the raw data.

    Returns:
    pd.DataFrame: Processed data.
    """
    # Finding Start and End Days of the Data Set
    start_month = data_file.iloc[0, 1]
    start_day = data_file.iloc[0, 0]
    start_year = data_file.iloc[0, 2]

    end_month = data_file.iloc[-1, 1]
    end_day = data_file.iloc[-1, 0]
    end_year = data_file.iloc[-1, 2]

    # Computing Rows and Columns for the Processed Data File
    rows, cols, tot_days = rows_cols_to_compute_data_cleaning(start_year, start_month, start_day, end_year, end_month, end_day, res, data_cols, 4)

    # Initializing Processed Data File to NaNs
    processed_data = np.full((rows, cols), np.nan)

    # Creating Date Time (Decimal Solar Time) Matrix
    date_time_matrix, _, time_t = start_end_calendar(start_year, start_month, start_day, tot_days, res, data_cols)
    time_t = time_t.T
    len_time_t = len(time_t)

    # Copying the DateTimeMatrix to the ProcessedData Matrix
    processed_data[:, :4] = date_time_matrix

    # Updating ProcessedData Data Columns
    for i in range(len(data_file)):
        month = data_file.iloc[i, 1]
        day = data_file.iloc[i, 0]
        year = data_file.iloc[i, 2]
        time_deci = data_file.iloc[i, 3]
        data_capture = data_file.iloc[i, 4:4+data_cols].values

        # Finding Corrected Time value for Time Deci
        difference = np.abs(time_deci - time_t)
        min_diff_idx = np.argmin(difference)
        corrected_time = time_t[min_diff_idx]

        # Computing Correct Index in ProcessedData Matrix
        for l in range(rows):
            if (processed_data[l, 0] == day and
                processed_data[l, 1] == month and
                processed_data[l, 2] == year and
                processed_data[l, 3] == corrected_time):
                break

        # Storing Data
        processed_data[l, 4:4+data_cols] = data_capture

    # N Point Average Method for Filling missing Data
    ra_n = np.zeros((n, data_cols))
    for i in range(rows):
        ra_counter = i % n
        for k in range(data_cols):
            if np.isnan(processed_data[i, k+4]):
                n_point_average_n = np.sum(ra_n[:, k]) / n
                processed_data[i, k+4] = n_point_average_n
            ra_n[ra_counter, k] = processed_data[i, k+4]

    for i in range(rows-1, -1, -1):
        ra_counter = i % n
        for k in range(data_cols):
            if np.isnan(processed_data[i, k+4]):
                n_point_average_n = np.sum(ra_n[:, k]) / n
                processed_data[i, k+4] = n_point_average_n
            ra_n[ra_counter, k] = processed_data[i, k+4]

    return pd.DataFrame(processed_data)
    
# Example usage:
# processed_data = solar_pv_weather_data_cleaner_modified_for_pecan_street(15, 5, 3, pd.read_excel('data.xlsx'))

###############################################################################################
# Module Functions : Date Time
###############################################################################################

def deci_to_hm(Td):
    """
    Convert decimal hours to hours, minutes, and seconds.

    Parameters:
    Td (array-like): Array of time in decimal hours.

    Returns:
    tuple: Tuple containing:
        - hr (ndarray): Hours.
        - min (ndarray): Minutes.
        - sec (ndarray): Seconds.
    """
    Td = np.array(Td)
    hr = np.fix(Td).astype(int)
    mmm = np.remainder(Td, 1)
    mm = mmm * 60
    min = np.fix(mm).astype(int)
    sss = np.remainder(mm, 1)
    ss = sss * 60
    sec = np.fix(ss).astype(int)

    return hr, min, sec

# Example usage:
# Td = [1.5, 2.75, 3.125]  # Example array of decimal hours
# hr, min, sec = deci_to_hm(Td)
# print("Hours:", hr)
# print("Minutes:", min)
# print("Seconds:", sec)

def hm_to_deci(hr, min, sec):
    """
    Convert hours, minutes, and seconds to decimal hours.

    Parameters:
    hr (int or array-like): Hours.
    min (int or array-like): Minutes.
    sec (int or array-like): Seconds.

    Returns:
    float or ndarray: Time in decimal hours.
    """
    hr = np.array(hr)
    min = np.array(min)
    sec = np.array(sec)
    
    MinD = min / 60
    SecD = sec / 3600
    Td = hr + MinD + SecD
    
    return Td

# Example usage:
# hr = [1, 2, 3]  # Example hours
# min = [30, 45, 15]  # Example minutes
# sec = [0, 30, 45]  # Example seconds
# Td = hm_to_deci(hr, min, sec)
# print("Decimal hours:", Td)

def leap_year_finder(year):
    """
    Determine if a given year is a leap year.

    Parameters:
    year (int): The year to check.

    Returns:
    int: 1 if leap year, 0 otherwise.
    """
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 1
    else:
        return 0        
        
def julian_day(day, month, year):
    """
    Calculate the Julian day number for a given date.

    Parameters:
    day (int): Day of the month.
    month (int): Month of the year.
    year (int): Year.

    Returns:
    int: Julian day number.
    """
    leap_year = leap_year_finder(year)
    
    if leap_year == 0:
        month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    elif leap_year == 1:
        month_days = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    else:
        raise ValueError("LeapYear should be 0 or 1")

    start_month_start_day = month_days[month - 1]
    n = start_month_start_day + day - 1

    return n
    
# Example usage:
# day = 18
# month = 6
# year = 2023
# julian_day_num = julian_day(day, month, year)
# print("Julian Day Number:", julian_day_num)

def days_to_compute(LeapYear, StartDay, StartMonth, EndDay, EndMonth):
    """
    Calculate the start and end days of the year based on whether it is a leap year
    and given start and end dates.

    Parameters:
    LeapYear (int): 0 for non-leap year, 1 for leap year.
    StartDay (int): Start day of the month.
    StartMonth (int): Start month.
    EndDay (int): End day of the month.
    EndMonth (int): End month.

    Returns:
    tuple: Tuple containing:
        - StartDay (int): Calculated start day of the year.
        - EndDay (int): Calculated end day of the year.
    """
    start_month_start_day = 0
    end_month_start_day = 0

    if LeapYear == 0:
        month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    elif LeapYear == 1:
        month_days = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    else:
        raise ValueError("LeapYear should be 0 or 1")

    start_month_start_day = month_days[StartMonth - 1]
    end_month_start_day = month_days[EndMonth - 1]

    StartDay = start_month_start_day + StartDay - 1
    EndDay = end_month_start_day + EndDay - 1

    return StartDay, EndDay

# Example usage:
# LeapYear = 0
# StartDay = 15
# StartMonth = 6
# EndDay = 20
# EndMonth = 7
# start_day, end_day = days_to_compute(LeapYear, StartDay, StartMonth, EndDay, EndMonth)
# print("StartDay:", start_day)
# print("EndDay:", end_day)

def start_end_calendar(StartYear, StartMonth, StartDay, TotDays, Res, DataCols):
    """
    Generate a date-time matrix from a given start date over a specified number of days with a given resolution.

    Parameters:
    StartYear (int): Starting year.
    StartMonth (int): Starting month.
    StartDay (int): Starting day.
    TotDays (int): Total number of days to generate.
    Res (int): Time resolution in minutes.
    DataCols (int): Number of additional data columns.

    Returns:
    tuple: Tuple containing:
        - DateTimeMatrix (ndarray): Matrix with date, time, and data columns.
        - TotDataPoints (int): Total number of data points.
        - Time (ndarray): Time vector for one day.
    """
    # Initialize start date
    SD = StartDay
    SM = StartMonth
    SY = StartYear

    # Compute number of data points in one day and total number of data points
    DayPoints = 24 * (60 // Res)
    TotDataPoints = TotDays * DayPoints
    Time = np.arange(0, 24, Res / 60)
    Time = Time.reshape(-1, 1)

    # Compute total number of columns (initial 4 columns for date-time signature)
    TotCols = DataCols + 4

    # Initialize DateTimeMatrix to zeros and count
    DateTimeMatrix = np.zeros((TotDataPoints, TotCols))
    Count = 0

    # Initialize month/year markers
    Tn, Th, Tl, Yr = 0, 0, 0, 0

    # Create DateTimeMatrix using a loop
    for i in range(TotDays):
        if i == 0:
            SY = SY
            SM = SM
            SD = SD
        elif SD == 31 and SM == 12:
            SY += 1
            SM = 1
            SD = 1
            Yr = 1

        LP = leap_year_finder(SY)

        if SD == 31:
            SD = 1
            SM += 1
            Tn = 1
        elif SD == 30 and SM in [4, 6, 9, 11]:
            SD = 1
            SM += 1
            Th = 1
        elif (SD == 28 and not LP and SM == 2) or (SD == 29 and LP and SM == 2):
            SD = 1
            SM += 1
            Tl = 1

        DayIncrement = (
            i != 0 and (
                (SD != 31 and SM in [1, 3, 5, 7, 8, 10, 12]) or
                (SD != 30 and SM in [4, 6, 9, 11]) or
                ((SD != 28 and not LP and SM == 2) or (SD != 29 and LP and SM == 2))
            )
        )

        if DayIncrement:
            if Tn == 1 or Th == 1 or Tl == 1 or Yr == 1:
                SD = 1
            else:
                SD += 1

        for k in range(len(Time)):
            Count += 1
            DateTimeMatrix[Count - 1, :] = [SD, SM, SY, Time[k, 0]] + [0] * DataCols

        Tn, Th, Tl, Yr = 0, 0, 0, 0

    return DateTimeMatrix, TotDataPoints, Time

# Example usage:
# StartYear = 2023
# StartMonth = 1
# StartDay = 1
# TotDays = 10
# Res = 60  # 1 hour resolution
# DataCols = 2
# DateTimeMatrix, TotDataPoints, Time = start_end_calendar(StartYear, StartMonth, StartDay, TotDays, Res, DataCols)
# print("DateTimeMatrix:\n", DateTimeMatrix)
# print("Total Data Points:", TotDataPoints)
# print("Time Vector:\n", Time)


    

###############################################################################################
# Module Functions : Automatic Email
###############################################################################################

def send_email_from_python(email_input_struct):
    """
    Send an email using Python.

    Parameters:
    email_input_struct (dict): Dictionary containing email details:
        - Email_Sender (str): Sender email address.
        - Email_Password_Sender (str): Sender email password.
        - SMTP_Server_Sender (str): SMTP server address.
        - Email_ReceiverList_Cell (list): List of receiver email addresses.
        - Email_Subject (str): Subject of the email.
        - Email_Text_Vector (str): Body text of the email.
        - Email_Attachment_Cell (list): List of file paths to attach.
    """
    # Email Sender Information
    email_sender = email_input_struct['Email_Sender']
    email_password_sender = email_input_struct['Email_Password_Sender']
    smtp_server_sender = email_input_struct['SMTP_Server_Sender']

    # Email Contents
    email_receiver_list = email_input_struct['Email_ReceiverList_Cell']
    email_subject = email_input_struct['Email_Subject']
    email_text = email_input_struct['Email_Text_Vector']
    email_attachments = email_input_struct['Email_Attachment_Cell']

    # Setting up the MIME
    message = MIMEMultipart()
    message['From'] = email_sender
    message['To'] = ', '.join(email_receiver_list)
    message['Subject'] = email_subject

    # Attach the body with the msg instance
    message.attach(MIMEText(email_text, 'plain'))

    # Attachments
    for file in email_attachments:
        # Open the file as binary mode
        with open(file, "rb") as attachment:
            # Add file as application/octet-stream
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {file}",
        )

        # Add attachment to message and convert message to string
        message.attach(part)

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Sending the Email
    with smtplib.SMTP_SSL(smtp_server_sender, 465, context=context) as server:
        server.login(email_sender, email_password_sender)
        server.sendmail(email_sender, email_receiver_list, message.as_string())

# Example usage:
# email_input_struct = {
#     'Email_Sender': 'sender@example.com',
#     'Email_Password_Sender': 'password',
#     'SMTP_Server_Sender': 'smtp.example.com',
#     'Email_ReceiverList_Cell': ['receiver1@example.com', 'receiver2@example.com'],
#     'Email_Subject': 'Test Email',
#     'Email_Text_Vector': 'This is a test email sent from Python.',
#     'Email_Attachment_Cell': ['path/to/attachment1', 'path/to/attachment2']
# }
# send_email_from_python(email_input_struct)


