### Pecan Street Load Data Processing Module ###

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

###############################################################################################
# Module Functions : For Data Preprocessing of PecanStreet Dataset
###############################################################################################

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

def pecan_street_low_to_high_res(original_resolution, new_resolution, processing_type, weather_data):
    """
    Convert weather data from low resolution to high resolution using zero-order hold.

    Parameters:
    original_resolution (int): Original resolution in minutes.
    new_resolution (int): New resolution in minutes.
    processing_type (int): Type of processing (1 for full file, 2 for user-defined).
    weather_data (numpy.ndarray): Weather data array with datetime and weather parameters.

    Returns:
    numpy.ndarray: Weather data array with the new resolution.
    """
    def deci_to_hms(deci_time):
        """Convert decimal time to hours, minutes, and seconds."""
        hours = int(deci_time)
        minutes = int((deci_time - hours) * 60)
        seconds = int((((deci_time - hours) * 60) - minutes) * 60)
        return hours, minutes, seconds

    def hms_to_deci(hour, minute, second):
        """Convert hours, minutes, and seconds to decimal time."""
        return hour + minute / 60 + second / 3600

    def datetime_series_slicer(weather_data, original_resolution, start_year, end_year, start_month, end_month, start_day, end_day, start_time, end_time):
        """Slice the datetime series based on the provided parameters."""
        start_date = datetime(start_year, start_month, start_day) + timedelta(hours=start_time)
        end_date = datetime(end_year, end_month, end_day) + timedelta(hours=end_time)
        timestamps = [datetime(year=int(row[2]), month=int(row[1]), day=int(row[0]), hour=deci_to_hms(row[3])[0], minute=deci_to_hms(row[3])[1], second=deci_to_hms(row[3])[2]).timestamp() for row in weather_data]
        mask = (np.array(timestamps) >= start_date.timestamp()) & (np.array(timestamps) <= end_date.timestamp())
        start_index = np.where(mask)[0][0]
        end_index = np.where(mask)[0][-1]
        return mask, start_index, end_index

    
    R, C = weather_data.shape
    FileRes = original_resolution
    Delta_T_Hour_OriginalRes = original_resolution / 60
    Delta_T_Hour_NewRes = new_resolution / 60

    weather_data[:, 4:] /= Delta_T_Hour_OriginalRes

    if processing_type == 1:
        StartDay, StartMonth, StartYear, StartTime = weather_data[0, :4]
        EndDay, EndMonth, EndYear, EndTime = weather_data[-1, :4]
    elif processing_type == 2:
        StartDay, StartMonth, StartYear, StartTime = 1, 1, 2017, 0
        EndDay, EndMonth, EndYear, EndTime = 31, 12, 2017, 23.5

    StartHr, StartMin, StartSec = deci_to_hms(StartTime)
    EndHr, EndMin, EndSec = deci_to_hms(EndTime)

    _, StartIndex, EndIndex = datetime_series_slicer(weather_data, original_resolution, 
                                                     int(StartYear), int(EndYear), int(StartMonth), 
                                                     int(EndMonth), int(StartDay), int(EndDay), 0, EndTime)

    Start_DateTime = datetime(int(StartYear), int(StartMonth), int(StartDay), StartHr, StartMin, StartSec)
    End_DateTime = datetime(int(EndYear), int(EndMonth), int(EndDay), EndHr, EndMin, EndSec)

    NewResolution_Duration = timedelta(minutes=new_resolution)
    DateTimeArray = [Start_DateTime + i * NewResolution_Duration for i in range(int((End_DateTime - Start_DateTime) / NewResolution_Duration) + 1)]
    
    NewResFile = np.zeros((len(DateTimeArray), C))
    Counter_OldTime = StartIndex

    for Counter_NewTime, CurrentDateTime_NewRes in enumerate(DateTimeArray):
        if (CurrentDateTime_NewRes.timestamp() <= datetime(int(weather_data[Counter_OldTime, 2]), int(weather_data[Counter_OldTime, 1]), int(weather_data[Counter_OldTime, 0]), *deci_to_hms(weather_data[Counter_OldTime, 3])).timestamp()):
            Day = CurrentDateTime_NewRes.day
            Month = CurrentDateTime_NewRes.month
            Year = CurrentDateTime_NewRes.year
            Hour = CurrentDateTime_NewRes.hour
            Minute = CurrentDateTime_NewRes.minute
            Second = CurrentDateTime_NewRes.second

            Time = hms_to_deci(Hour, Minute, Second)
            NewResFile[Counter_NewTime, :] = np.hstack(([Day, Month, Year, Time], weather_data[Counter_OldTime, 4:]))
        else:
            Counter_OldTime += 1
            if Counter_OldTime >= len(weather_data):
                break
            Day = CurrentDateTime_NewRes.day
            Month = CurrentDateTime_NewRes.month
            Year = CurrentDateTime_NewRes.year
            Hour = CurrentDateTime_NewRes.hour
            Minute = CurrentDateTime_NewRes.minute
            Second = CurrentDateTime_NewRes.second

            Time = hms_to_deci(Hour, Minute, Second)
            NewResFile[Counter_NewTime, :] = np.hstack(([Day, Month, Year, Time], weather_data[Counter_OldTime, 4:]))

    NewResFile[:, 4:] *= Delta_T_Hour_NewRes
    return NewResFile

def pecan_street_data_preprocessing(data_folder_path, file_name, original_resolution, new_resolution, processing_type, averaging_points):
    """
    Pecan Street Data Preprocessing: Converting Many Files to 1
    
    Parameters:
    data_folder_path (str): Path to the folder containing the data files.
    file_name (str): Name of the CSV file to process.
    original_resolution (int): Original resolution in minutes.
    new_resolution (int): New resolution in minutes.
    processing_type (int): Type of processing (1 for full file, 2 for user-defined).
    averaging_points (int): Number of averaging points.

    Example:
    >>> data_folder_path = 'C:/Users/ninad/Dropbox (UFL)/NinadGaikwad_PhD/Gaikwad_Research/Gaikwad_Research_Work/20_Gaikwad_SmartCommunity/data/RawFiles/15minute_data_newyork/'
    >>> file_name = '15minute_data_newyork.csv'
    >>> original_resolution = 15
    >>> new_resolution = 10
    >>> processing_type = 1
    >>> averaging_points = 2
    >>> pecan_street_data_preprocessing(data_folder_path, file_name, original_resolution, new_resolution, processing_type, averaging_points)
    """
    
    # Step 1: Reading CSV File
    data_full_path = data_folder_path + file_name
    actual_file = pd.read_csv(data_full_path)
    
    # Getting relevant columns of data in array format
    id_array = actual_file.iloc[:, 0].values
    date_time_string_array = actual_file.iloc[:, 1].values
    data_array = actual_file.iloc[:, 2:].values
    
    data_frame_row, data_frame_column = data_array.shape
    data_cols = data_frame_column - 2
    
    # Step 2: Changing Date-Time Stamp Columns for Utility
    date_time_stamp = np.zeros((len(date_time_string_array), 4))  # Initialization
    
    for i in range(len(date_time_string_array)):  # For each row in actual_file
        date_time_string = date_time_string_array[i]
        date_time_obj = datetime.strptime(date_time_string, '%Y-%m-%d %H:%M')
        
        year = date_time_obj.year
        month = date_time_obj.month
        day = date_time_obj.day
        hour = date_time_obj.hour
        minute = date_time_obj.minute
        
        time_deci = hour + minute / 60.0
        
        date_time_stamp[i, :4] = [day, month, year, time_deci]
        
        print(i)  # Debugger
    
    # Step 3: Grouping Data according to different houses
    unique_houses = np.unique(id_array)
    
    file_num = len(unique_houses)
    print(f"File_Num: {file_num}")  # Debugger
    
    file_num_current = 0  # Debugger
    
    for i in range(len(unique_houses)):  # For each house
        # Incrementing file_num_current
        file_num_current += 1
        print(f"FileNum_Current: {file_num_current}")  # Debugger
        
        # Finding indices for the current house
        current_house_indices = np.where(id_array == unique_houses[i])[0]
        
        # Creating current house dataframe
        current_house_dataframe = np.hstack((date_time_stamp[current_house_indices, :], data_array[current_house_indices, :]))
        
        # Converting NaNs to zeros
        current_house_dataframe[np.isnan(current_house_dataframe)] = 0
        
        # Clean data for any irregularities of date-time ordering and missing data (Negatives are not converted to 0s)
        processed_dataframe = solar_pv_weather_data_cleaner_modified_for_pecan_street(original_resolution, data_cols, averaging_points, current_house_dataframe)
        
        # Changing to required resolution
        current_house_dataframe_new_res_file = pecan_street_low_to_high_res(original_resolution, new_resolution, processing_type, processed_dataframe)
        
        # Current house name
        current_house_name = f"House_{i+1}_{original_resolution}minTo_{new_resolution}min.csv"
        
        # Saving house data in a CSV file
        np.savetxt(data_folder_path + current_house_name, current_house_dataframe_new_res_file, delimiter=",")


###############################################################################################
# Module Functions : Data Extraction from Preprocessed PecanStreet Dataset
###############################################################################################
    
def pecan_street_data_extractor(HEMSWeatherData_Input, PecanStreet_Data_FolderPath, N_House_Vector, Type, Data_MatFile_Name):
    """
    Pecan Street Data Extraction: Converting Many Files to 1
    
    :param HEMSWeatherData_Input: Dictionary containing weather data input parameters
    :param PecanStreet_Data_FolderPath: String path to the folder containing Pecan Street data files
    :param N_House_Vector: List of house numbers (if applicable)
    :param Type: Type of data to extract (if applicable)
    :param Data_MatFile_Name: Output file name for the combined data
    :return: 3D numpy array containing the combined data from all files
    
    Example usage:
    HEMSWeatherData_Input = {
        'StartMonth': 1,
        'StartDay': 1,
        'StartTime': '0.0',
        'EndMonth': 1,
        'EndDay': 1,
        'EndTime': '23.0'
    }
    PecanStreet_Data_FolderPath = 'path/to/data/folder'
    N_House_Vector = [2, 2, 2, 2]
    Type = 1
    Data_MatFile_Name = 'output.mat'
    
    output = pecan_street_data_extractor(HEMSWeatherData_Input, PecanStreet_Data_FolderPath, N_House_Vector, Type, Data_MatFile_Name)
    """
    
    def convert_decimal_time_to_hhmm(decimal_time):
        """
        Converts a decimal time (0-24) to hours and minutes.
        
        :param decimal_time: Decimal time value.
        :return: Tuple of hours and minutes.
        """
        hours = int(decimal_time)
        minutes = int((decimal_time - hours) * 60)
        return hours, minutes

    def date_time_series_slicer(df, start_year, end_year, start_month, end_month, start_day, end_day, start_time, end_time):
        """
        Slices the dataframe to get rows within the specified date-time range.
        
        :param df: DataFrame with separate columns for year, month, day, and decimal time.
        :param start_year: Start year for filtering.
        :param end_year: End year for filtering.
        :param start_month: Start month for filtering.
        :param end_month: End month for filtering.
        :param start_day: Start day for filtering.
        :param end_day: End day for filtering.
        :param start_time: Start time for filtering (float, 0-24).
        :param end_time: End time for filtering (float, 0-24).
        :return: Filtered DataFrame, start index, end index
        """
        start_hours, start_minutes = convert_decimal_time_to_hhmm(start_time)
        end_hours, end_minutes = convert_decimal_time_to_hhmm(end_time)

        start_datetime = datetime(start_year, start_month, start_day, start_hours, start_minutes)
        end_datetime = datetime(end_year, end_month, end_day, end_hours, end_minutes)

        df['datetime'] = pd.to_datetime(df[[0, 1, 2]].astype(str).agg('-'.join, axis=1)) + pd.to_timedelta(df[3], unit='h')

        mask = (df['datetime'] >= start_datetime) & (df['datetime'] <= end_datetime)
        filtered_df = df.loc[mask]

        if filtered_df.empty:
            return None, None, None

        start_index = filtered_df.index[0]
        end_index = filtered_df.index[-1]

        filtered_df = filtered_df.drop(columns=['datetime'])

        return filtered_df, start_index, end_index

    def rearrange_columns(data):
        """
        Rearranges the columns of the data matrix according to the specified priorities.
        
        :param data: 3D numpy array of data
        :return: 3D numpy array with columns rearranged
        """
        columns_priority_order = [
            0, 1, 2, 3,  # Date-Time
            66+4, 67+4, 13+4, 14+4, 15+4,  # Solar PV, Battery, EV
            61+4, 62+4, 25+4, 39+4, 374, 40+4, 38+4, 60+4, 71+4, 49+4, 53+4, 54+4,  # Level 1 Priority - Fridge, Freezer, Kitchen
            8+4, 9+4, 10+4, 11+4, 12+4,  # Level 2 Priority - Bedrooms
            47+4, 48+4, 50+4,  # Level 3 Priority - Living Rooms, Office Room
            17+4, 23+4, 64+4, 22+4, 59+4, 69+4, 74+4,  # Level 4 Priority - Clothes, Garbage Disposal, Pumps
            63+4, 6+4, 7+4, 19+4, 20+4, 21+4,  # Level 5 Priority - Security, Bathrooms, Dining Room, Dishwasher
            28+4, 29+4, 70+4, 65+4, 51+4, 52+4,  # Level 6 Priority - Remaining Rooms, Outside Lights
            5+4, 68+4, 75+4,  # Level 7 Priority - Aquarium, Lawn Sprinklers, Wine Cooler
            35+4, 58+4, 57+4, 36+4, 72+4, 73+4  # Level 8 Priority - Pool, Jacuzzi, Water Heater
        ]
        
        # Filter columns to only those that exist in the data
        columns_priority_order = [col for col in columns_priority_order if col < data.shape[1]]
        
        # Reorder columns for each file
        rearranged_data = data[:, columns_priority_order, :]
        
        return rearranged_data


    # Extracting required data from HEMSWeatherData_Input
    start_month = HEMSWeatherData_Input['StartMonth']
    start_day = HEMSWeatherData_Input['StartDay']
    start_time = float(HEMSWeatherData_Input['StartTime'])
    end_month = HEMSWeatherData_Input['EndMonth']
    end_day = HEMSWeatherData_Input['EndDay']
    end_time = float(HEMSWeatherData_Input['EndTime'])

    date_time_no_error_counter = 0
    all_files_correct_dates_data = []

    # Getting all files data from the specified folder
    for file_name in os.listdir(PecanStreet_Data_FolderPath):
        # Skip temporary or non-CSV files
        if re.match(r'^[a-zA-Z0-9_]+\w*$', file_name):
            full_path_name = os.path.join(PecanStreet_Data_FolderPath, file_name)
            try:
                # Read CSV file without headers
                data = pd.read_csv(full_path_name, header=None)

                start_year = data.iloc[0, 0]
                end_year = start_year

                # Slicing the data for the required date-time range
                sliced_data, start_index, end_index = date_time_series_slicer(data, start_year, end_year, start_month, end_month, start_day, end_day, start_time, end_time)

                if sliced_data is not None:
                    date_time_no_error_counter += 1
                    all_files_correct_dates_data.append(sliced_data.to_numpy())
            except Exception as e:
                print(f"Error processing {full_path_name}: {e}")

    if not all_files_correct_dates_data:
        print('Warning: Desired dates not found in files')
        return 'None'

    # Convert list of 2D arrays to a 3D array
    all_files_correct_dates_data = np.array(all_files_correct_dates_data)

    # Rearrange columns according to the specified priorities
    rearranged_data = rearrange_columns(all_files_correct_dates_data)
    
    # Number of files in the data
    _, _, file_num = rearranged_data.shape

    # Computing Info for PV File Indices
    house_pv_sum_array = np.sum(rearranged_data[:, 4:6, :], axis=(0, 1))
    house_pv_file_indices = np.where(house_pv_sum_array > 0)[0]

    # Computing Info for Bat File Indices
    house_bat_sum_array = np.sum(rearranged_data[:, 12, :], axis=0)
    house_bat_file_indices = np.where(house_bat_sum_array > 0)[0]

    # Computing Info for EV File Indices
    house_ev_sum_array = np.sum(rearranged_data[:, 13:15, :], axis=(0, 1))
    house_ev_file_indices = np.where(house_ev_sum_array > 0)[0]

    # PV_Bat_EV File Indices Intersection
    house_pv_bat_ev_file_indices_intersection = np.intersect1d(np.intersect1d(house_pv_file_indices, house_bat_file_indices), house_ev_file_indices)

    # PV_Bat_EV File Indices Union
    house_pv_bat_ev_file_indices_union = np.union1d(np.union1d(house_pv_file_indices, house_bat_file_indices), house_ev_file_indices)

    # Only None Indices
    house_only_none_file_indices = np.setdiff1d(np.arange(file_num), house_pv_bat_ev_file_indices_union)

    # Only PV Indices
    house_only_pv_file_indices = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.arange(file_num), house_only_none_file_indices), house_bat_file_indices), house_ev_file_indices)

    # Only Bat Indices
    house_only_bat_file_indices = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.arange(file_num), house_only_none_file_indices), house_pv_file_indices), house_ev_file_indices)

    # Only EV Indices
    house_only_ev_file_indices = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.arange(file_num), house_only_none_file_indices), house_pv_file_indices), house_bat_file_indices)

    # Only PV_Bat Indices
    house_only_pv_bat_file_indices = np.intersect1d(house_only_pv_file_indices, house_only_bat_file_indices)

    # Only PV_EV Indices
    house_only_pv_ev_file_indices = np.intersect1d(house_only_pv_file_indices, house_only_ev_file_indices)

    # Only Bat_EV Indices
    house_only_bat_ev_file_indices = np.intersect1d(house_only_bat_file_indices, house_only_ev_file_indices)

    # Getting required House Data - Based on Type of Community [PV, Bat, EV, None]
    length_house_vector = len(N_House_Vector)
    
    classified_data = {
        'PV_Bat_EV': rearranged_data[:, :, house_pv_bat_ev_file_indices_intersection],
        'PV_Bat': rearranged_data[:, :, house_only_pv_bat_file_indices],
        'PV_EV': rearranged_data[:, :, house_only_pv_ev_file_indices],
        'Bat_EV': rearranged_data[:, :, house_only_bat_ev_file_indices],
        'PV': rearranged_data[:, :, house_only_pv_file_indices],
        'Bat': rearranged_data[:, :, house_only_bat_file_indices],
        'EV': rearranged_data[:, :, house_only_ev_file_indices],
        'None': rearranged_data[:, :, house_only_none_file_indices],
    }
    rows, columns, _ = next(iter(classified_data.values())).shape

    if length_house_vector == 1:  # Single Large House
        if Type == 1:
            if classified_data['PV_Bat_EV'].size > 0:
                PecanStreet_Data_Output = classified_data['PV_Bat_EV'][:, :, 0]
            elif classified_data['Bat_EV'].size > 0:
                print("Warning: Bat_EV File instead of PV_Bat_EV")
                PecanStreet_Data_Output = classified_data['Bat_EV'][:, :, 0]
            elif classified_data['PV_EV'].size > 0:
                print("Warning: PV_EV File instead of PV_Bat_EV")
                PecanStreet_Data_Output = classified_data['PV_EV'][:, :, 0]
            elif classified_data['EV'].size > 0:
                print("Warning: EV File instead of PV_Bat_EV")
                PecanStreet_Data_Output = classified_data['EV'][:, :, 0]
            elif classified_data['PV_Bat'].size > 0:
                print("Warning: PV_Bat File instead of PV_Bat_EV")
                PecanStreet_Data_Output = classified_data['PV_Bat'][:, :, 0]
            elif classified_data['Bat'].size > 0:
                print("Warning: Bat File instead of PV_Bat_EV")
                PecanStreet_Data_Output = classified_data['Bat'][:, :, 0]
            elif classified_data['PV'].size > 0:
                print("Warning: PV File instead of PV_Bat_EV")
                PecanStreet_Data_Output = classified_data['PV'][:, :, 0]
            elif classified_data['None'].size > 0:
                print("Warning: None File instead of PV_Bat_EV")
                PecanStreet_Data_Output = classified_data['None'][:, :, 0]
            else:
                print("Warning: No File found for the desired dates and required type")
                PecanStreet_Data_Output = np.zeros((rows, columns))

        elif Type == 2:
            if classified_data['PV_Bat'].size > 0:
                PecanStreet_Data_Output = classified_data['PV_Bat'][:, :, 0]
            elif classified_data['PV_Bat_EV'].size > 0:
                print("Warning: PV_Bat_EV File instead of PV_Bat")
                PecanStreet_Data_Output = classified_data['PV_Bat_EV'][:, :, 0]
            elif classified_data['Bat_EV'].size > 0:
                print("Warning: Bat_EV File instead of PV_Bat")
                PecanStreet_Data_Output = classified_data['Bat_EV'][:, :, 0]
            elif classified_data['PV_EV'].size > 0:
                print("Warning: PV_EV File instead of PV_Bat")
                PecanStreet_Data_Output = classified_data['PV_EV'][:, :, 0]
            elif classified_data['Bat'].size > 0:
                print("Warning: Bat File instead of PV_Bat")
                PecanStreet_Data_Output = classified_data['Bat'][:, :, 0]
            elif classified_data['PV'].size > 0:
                print("Warning: PV File instead of PV_Bat")
                PecanStreet_Data_Output = classified_data['PV'][:, :, 0]
            elif classified_data['EV'].size > 0:
                print("Warning: EV File instead of PV_Bat")
                PecanStreet_Data_Output = classified_data['EV'][:, :, 0]
            elif classified_data['None'].size > 0:
                print("Warning: None File instead of PV_Bat")
                PecanStreet_Data_Output = classified_data['None'][:, :, 0]
            else:
                print("Warning: No File found for the desired dates and required type")
                PecanStreet_Data_Output = np.zeros((rows, columns))

        elif Type == 3:
            if classified_data['PV_EV'].size > 0:
                PecanStreet_Data_Output = classified_data['PV_EV'][:, :, 0]
            elif classified_data['PV_Bat_EV'].size > 0:
                print("Warning: PV_Bat_EV File instead of PV_EV")
                PecanStreet_Data_Output = classified_data['PV_Bat_EV'][:, :, 0]
            elif classified_data['Bat_EV'].size > 0:
                print("Warning: Bat_EV File instead of PV_EV")
                PecanStreet_Data_Output = classified_data['Bat_EV'][:, :, 0]
            elif classified_data['EV'].size > 0:
                print("Warning: EV File instead of PV_EV")
                PecanStreet_Data_Output = classified_data['EV'][:, :, 0]
            elif classified_data['PV'].size > 0:
                print("Warning: PV File instead of PV_EV")
                PecanStreet_Data_Output = classified_data['PV'][:, :, 0]
            elif classified_data['PV_Bat'].size > 0:
                print("Warning: PV_Bat File instead of PV_EV")
                PecanStreet_Data_Output = classified_data['PV_Bat'][:, :, 0]
            elif classified_data['Bat'].size > 0:
                print("Warning: Bat File instead of PV_EV")
                PecanStreet_Data_Output = classified_data['Bat'][:, :, 0]
            elif classified_data['None'].size > 0:
                print("Warning: None File instead of PV_EV")
                PecanStreet_Data_Output = classified_data['None'][:, :, 0]
            else:
                print("Warning: No File found for the desired dates and required type")
                PecanStreet_Data_Output = np.zeros((rows, columns))

        elif Type == 4:
            if classified_data['Bat_EV'].size > 0:
                PecanStreet_Data_Output = classified_data['Bat_EV'][:, :, 0]
            elif classified_data['PV_Bat_EV'].size > 0:
                print("Warning: PV_Bat_EV File instead of Bat_EV")
                PecanStreet_Data_Output = classified_data['PV_Bat_EV'][:, :, 0]
            elif classified_data['EV'].size > 0:
                print("Warning: EV File instead of Bat_EV")
                PecanStreet_Data_Output = classified_data['EV'][:, :, 0]
            elif classified_data['PV_EV'].size > 0:
                print("Warning: PV_EV File instead of Bat_EV")
                PecanStreet_Data_Output = classified_data['PV_EV'][:, :, 0]
            elif classified_data['PV_Bat'].size > 0:
                print("Warning: PV_Bat File instead of Bat_EV")
                PecanStreet_Data_Output = classified_data['PV_Bat'][:, :, 0]
            elif classified_data['Bat'].size > 0:
                print("Warning: Bat File instead of Bat_EV")
                PecanStreet_Data_Output = classified_data['Bat'][:, :, 0]
            elif classified_data['PV'].size > 0:
                print("Warning: PV File instead of Bat_EV")
                PecanStreet_Data_Output = classified_data['PV'][:, :, 0]
            elif classified_data['None'].size > 0:
                print("Warning: None File instead of Bat_EV")
                PecanStreet_Data_Output = classified_data['None'][:, :, 0]
            else:
                print("Warning: No File found for the desired dates and required type")
                PecanStreet_Data_Output = np.zeros((rows, columns))

        elif Type == 5:
            if classified_data['PV'].size > 0:
                PecanStreet_Data_Output = classified_data['PV'][:, :, 0]
            elif classified_data['PV_Bat_EV'].size > 0:
                print("Warning: PV_Bat_EV File instead of PV")
                PecanStreet_Data_Output = classified_data['PV_Bat_EV'][:, :, 0]
            elif classified_data['PV_Bat'].size > 0:
                print("Warning: PV_Bat File instead of PV")
                PecanStreet_Data_Output = classified_data['PV_Bat'][:, :, 0]
            elif classified_data['PV_EV'].size > 0:
                print("Warning: PV_EV File instead of PV")
                PecanStreet_Data_Output = classified_data['PV_EV'][:, :, 0]
            elif classified_data['Bat_EV'].size > 0:
                print("Warning: Bat_EV File instead of PV")
                PecanStreet_Data_Output = classified_data['Bat_EV'][:, :, 0]
            elif classified_data['Bat'].size > 0:
                print("Warning: Bat File instead of PV")
                PecanStreet_Data_Output = classified_data['Bat'][:, :, 0]
            elif classified_data['EV'].size > 0:
                print("Warning: EV File instead of PV")
                PecanStreet_Data_Output = classified_data['EV'][:, :, 0]
            elif classified_data['None'].size > 0:
                print("Warning: None File instead of PV")
                PecanStreet_Data_Output = classified_data['None'][:, :, 0]
            else:
                print("Warning: No File found for the desired dates and required type")
                PecanStreet_Data_Output = np.zeros((rows, columns))

        elif Type == 6:
            if classified_data['Bat'].size > 0:
                PecanStreet_Data_Output = classified_data['Bat'][:, :, 0]
            elif classified_data['PV_Bat_EV'].size > 0:
                print("Warning: PV_Bat_EV File instead of Bat")
                PecanStreet_Data_Output = classified_data['PV_Bat_EV'][:, :, 0]
            elif classified_data['PV_Bat'].size > 0:
                print("Warning: PV_Bat File instead of Bat")
                PecanStreet_Data_Output = classified_data['PV_Bat'][:, :, 0]
            elif classified_data['Bat_EV'].size > 0:
                print("Warning: Bat_EV File instead of Bat")
                PecanStreet_Data_Output = classified_data['Bat_EV'][:, :, 0]
            elif classified_data['PV_EV'].size > 0:
                print("Warning: PV_EV File instead of Bat")
                PecanStreet_Data_Output = classified_data['PV_EV'][:, :, 0]
            elif classified_data['PV'].size > 0:
                print("Warning: PV File instead of Bat")
                PecanStreet_Data_Output = classified_data['PV'][:, :, 0]
            elif classified_data['EV'].size > 0:
                print("Warning: EV File instead of Bat")
                PecanStreet_Data_Output = classified_data['EV'][:, :, 0]
            elif classified_data['None'].size > 0:
                print("Warning: None File instead of Bat")
                PecanStreet_Data_Output = classified_data['None'][:, :, 0]
            else:
                print("Warning: No File found for the desired dates and required type")
                PecanStreet_Data_Output = np.zeros((rows, columns))

        elif Type == 7:
            if classified_data['EV'].size > 0:
                PecanStreet_Data_Output = classified_data['EV'][:, :, 0]
            elif classified_data['PV_Bat_EV'].size > 0:
                print("Warning: PV_Bat_EV File instead of EV")
                PecanStreet_Data_Output = classified_data['PV_Bat_EV'][:, :, 0]
            elif classified_data['PV_EV'].size > 0:
                print("Warning: PV_EV File instead of EV")
                PecanStreet_Data_Output = classified_data['PV_EV'][:, :, 0]
            elif classified_data['Bat_EV'].size > 0:
                print("Warning: Bat_EV File instead of EV")
                PecanStreet_Data_Output = classified_data['Bat_EV'][:, :, 0]
            elif classified_data['PV_Bat'].size > 0:
                print("Warning: PV_Bat File instead of EV")
                PecanStreet_Data_Output = classified_data['PV_Bat'][:, :, 0]
            elif classified_data['PV'].size > 0:
                print("Warning: PV File instead of EV")
                PecanStreet_Data_Output = classified_data['PV'][:, :, 0]
            elif classified_data['Bat'].size > 0:
                print("Warning: Bat File instead of EV")
                PecanStreet_Data_Output = classified_data['Bat'][:, :, 0]
            elif classified_data['None'].size > 0:
                print("Warning: None File instead of EV")
                PecanStreet_Data_Output = classified_data['None'][:, :, 0]
            else:
                print("Warning: No File found for the desired dates and required type")
                PecanStreet_Data_Output = np.zeros((rows, columns))

        elif Type == 8:
            if classified_data['None'].size > 0:
                PecanStreet_Data_Output = classified_data['None'][:, :, 0]
            elif classified_data['PV_Bat_EV'].size > 0:
                print("Warning: PV_Bat_EV File instead of None")
                PecanStreet_Data_Output = classified_data['PV_Bat_EV'][:, :, 0]
            elif classified_data['PV_Bat'].size > 0:
                print("Warning: PV_Bat File instead of None")
                PecanStreet_Data_Output = classified_data['PV_Bat'][:, :, 0]
            elif classified_data['PV_EV'].size > 0:
                print("Warning: PV_EV File instead of None")
                PecanStreet_Data_Output = classified_data['PV_EV'][:, :, 0]
            elif classified_data['Bat_EV'].size > 0:
                print("Warning: Bat_EV File instead of None")
                PecanStreet_Data_Output = classified_data['Bat_EV'][:, :, 0]
            elif classified_data['PV'].size > 0:
                print("Warning: PV File instead of None")
                PecanStreet_Data_Output = classified_data['PV'][:, :, 0]
            elif classified_data['Bat'].size > 0:
                print("Warning: Bat File instead of None")
                PecanStreet_Data_Output = classified_data['Bat'][:, :, 0]
            elif classified_data['EV'].size > 0:
                print("Warning: EV File instead of None")
                PecanStreet_Data_Output = classified_data['EV'][:, :, 0]
            else:
                print("Warning: No File found for the desired dates and required type")
                PecanStreet_Data_Output = np.zeros((rows, columns))
                
        # Save the data to a .mat file
        scipy.io.savemat(Data_MatFile_Name, PecanStreet_Data_Output)


        return PecanStreet_Data_Output

    elif length_house_vector == 4:  # Smart Community - [N_PV_Bat, N_Bat, N_PV, N_None]
        N_House = sum(N_House_Vector)
        N1 = N_House_Vector[0]
        N2 = N1 + N_House_Vector[1]
        N3 = N2 + N_House_Vector[2]
        N4 = N_House

        Counter_SmartCommunity_1 = 0
        PecanStreet_Data_Output = np.zeros((rows, columns, N_House))

        for ii in range(N_House):
            if ii < N1:  # N_PV_Bat
                if classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N2:  # N_Bat
                if classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N3:  # N_PV
                if classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N4:  # N_None
                if classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))
        
        # Save the data to a .mat file
        scipy.io.savemat(Data_MatFile_Name, PecanStreet_Data_Output)
        
        return PecanStreet_Data_Output

    elif length_house_vector == 8:  # Smart Community - [N_PV_Bat_EV, N_PV_Bat, N_PV_EV, N_Bat_EV, N_PV, N_Bat, N_EV, N_None]
        N_House = sum(N_House_Vector)
        N1 = N_House_Vector[0]
        N2 = N1 + N_House_Vector[1]
        N3 = N2 + N_House_Vector[2]
        N4 = N3 + N_House_Vector[3]
        N5 = N4 + N_House_Vector[4]
        N6 = N5 + N_House_Vector[5]
        N7 = N6 + N_House_Vector[6]
        N8 = N7 + N_House_Vector[7]

        Counter_SmartCommunity_1 = 0
        PecanStreet_Data_Output = np.zeros((rows, columns, N_House))

        for ii in range(N_House):
            if ii < N1:  # N_PV_Bat_EV
                if classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of N_PV_Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of N_PV_Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of N_PV_Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of N_PV_Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of N_PV_Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of N_PV_Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of N_PV_Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N2:  # N_PV_Bat
                if classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of PV_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N3:  # N_PV_EV
                if classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of PV_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of PV_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of PV_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of PV_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of PV_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of PV_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of PV_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N4:  # N_Bat_EV
                if classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of Bat_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N5:  # N_PV
                if classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of N_PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of N_PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of N_PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of N_PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of N_PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of N_PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of N_PV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N6:  # N_Bat
                if classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of N_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of N_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of N_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of N_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of N_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of N_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of N_Bat")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N7:  # N_EV
                if classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of N_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of N_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of N_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of N_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of N_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of N_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: None File instead of N_EV")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

            elif ii < N8:  # N_None
                if classified_data['None'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['None'][:, :, ii % len(classified_data['None'])]
                elif classified_data['PV_Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat_EV File instead of N_None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat_EV'][:, :, ii % len(classified_data['PV_Bat_EV'])]
                elif classified_data['PV_Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_Bat File instead of N_None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_Bat'][:, :, ii % len(classified_data['PV_Bat'])]
                elif classified_data['PV_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV_EV File instead of N_None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV_EV'][:, :, ii % len(classified_data['PV_EV'])]
                elif classified_data['Bat_EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat_EV File instead of N_None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat_EV'][:, :, ii % len(classified_data['Bat_EV'])]
                elif classified_data['PV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: PV File instead of N_None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['PV'][:, :, ii % len(classified_data['PV'])]
                elif classified_data['Bat'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: Bat File instead of N_None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['Bat'][:, :, ii % len(classified_data['Bat'])]
                elif classified_data['EV'].size > 0:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: EV File instead of N_None")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = classified_data['EV'][:, :, ii % len(classified_data['EV'])]
                else:
                    Counter_SmartCommunity_1 += 1
                    print("Warning: No File found for the desired dates and required type")
                    PecanStreet_Data_Output[:, :, Counter_SmartCommunity_1 - 1] = np.zeros((rows, columns))

        # Save the data to a .mat file
        scipy.io.savemat(Data_MatFile_Name, PecanStreet_Data_Output)
        
        return PecanStreet_Data_Output

    else:
        raise ValueError("Invalid length of N_House_Vector")
    

    

    











