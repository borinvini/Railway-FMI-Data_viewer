# PARAMETERS
START_DATE = "2024-11-29" # YYYY-MM-DD
END_DATE = "2024-12-02" # YYYY-MM-DD
CATEGORY = "All Categories" 
OPERATOR = "All Operators"

# 
FMI_BBOX = "18,55,35,75" # Bounding box for Finland
#FMI_BBOX = "20.8,59.4,27.2,67.6"

# URLs for the Finnish Meteorological Institute API
FMI_OBSERVATIONS = "fmi::observations::weather::multipointcoverage"
FMI_EMS = "fmi::ef::stations"

# URLs for the Finnish Railway API
FIN_RAILWAY_BASE_URL = "https://rata.digitraffic.fi/api/v1"
FIN_RAILWAY_STATIONS = "/metadata/stations"
FIN_RAILWAY_TRAIN_CAT = "/metadata/train-categories"
FIN_RAILWAY_ALL_TRAINS = "/trains"
FIN_RAILWAY_TRAIN_TRACKING = "/train-tracking"

# CSVs
FOLDER_NAME = "data"
CSV_TRAIN_STATIONS = "train_stations.csv"
CSV_TRAIN_CATEGORIES = "train_categories.csv"
CSV_ALL_TRAINS = "all_trains_data.csv"
CSV_FMI = "fmi_weather_observations.csv"
CSV_FMI_EMS = "fmi_ems_stations.csv"
CSV_CLOSEST_EMS_TRAIN = "closest_ems_to_train_stations.csv"
CSV_MATCHED_DATA = "matched_data.csv"