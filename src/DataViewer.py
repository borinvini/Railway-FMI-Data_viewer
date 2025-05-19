from collections import defaultdict
import os
import re
import pandas as pd
import streamlit as st

from config.const import CSV_MATCHED_DATA, FOLDER_NAME, VIEWER_FOLDER_NAME

class DataViewer:
    def __init__(self):
        # Make date_dict instance-specific
        self.date_dict = defaultdict(list)

    def get_date_dict(self):
        """
        Return the date_dict containing available years and months.
        If empty, shows a warning in Streamlit.
        """
        if not self.date_dict:
            st.warning("⚠️ No date information available. Please load the data first.")
            return None
        
        return self.date_dict


    def has_data(self):
        """
        Check if the data folder exists and is not empty.
        Shows a warning in Streamlit if no data is available.
        """
        if not os.path.exists(VIEWER_FOLDER_NAME) or not os.listdir(VIEWER_FOLDER_NAME):
            st.warning("⚠️ No data available! Please fetch the data first.")
            return False
        return True
    
    def check_file_pattern(self, file_pattern, subfolder=None):
        """
        Check if there are files matching the specified pattern in the data folder.
        Shows the number of files and the total size in MB or GB in Streamlit.
        
        Args:
            file_pattern: The pattern to match files against
            subfolder: Optional subfolder within VIEWER_FOLDER_NAME to search in
        """
        try:
            search_dir = VIEWER_FOLDER_NAME
            if subfolder:
                search_dir = os.path.join(VIEWER_FOLDER_NAME, subfolder)
                
            if not os.path.exists(search_dir):
                st.warning(f"⚠️ Directory {search_dir} does not exist.")
                return
                
            matched_files = [
                file for file in os.listdir(search_dir)
                if file.startswith(file_pattern.replace('.csv', '')) and file.endswith('.csv')
            ]

            if matched_files:
                num_files = len(matched_files)
                total_size = sum(os.path.getsize(os.path.join(search_dir, file)) for file in matched_files)
                total_size_mb = total_size / (1024 * 1024)  # Convert bytes to MB
                
                # If size is larger than 1024 MB, display in GB
                if total_size_mb > 1024:
                    total_size_gb = total_size_mb / 1024  # Convert MB to GB
                    st.success(f"✅ Found **{num_files}** files matching `{file_pattern}` with total size of **{total_size_gb:.2f} GB**.")
                else:
                    st.success(f"✅ Found **{num_files}** files matching `{file_pattern}` with total size of **{total_size_mb:.2f} MB**.")
            else:
                st.warning(f"⚠️ No files found matching `{file_pattern}` in {search_dir}.")
        
        except Exception as e:
            st.error(f"❌ Error while checking file pattern: {e}")


    def get_date_range(self):
        """
        Check available data files and extract the date range.
        Shows the date range in Streamlit if files are found, otherwise shows a warning.
        """
        try:
            # Look in matched_data subfolder for consistency
            matched_data_dir = os.path.join(VIEWER_FOLDER_NAME, "matched_data")
            
            if not os.path.exists(matched_data_dir):
                st.warning(f"⚠️ Directory '{matched_data_dir}' does not exist.")
                return None, None
                
            matched_files = [
                file for file in os.listdir(matched_data_dir)
                if re.match(rf"{CSV_MATCHED_DATA.replace('.csv', '')}_(\d{{4}})_(\d{{2}}).csv", file)
            ]
            
            if matched_files:
                # Extract dates from filenames and store them in the instance-level dict
                for file in matched_files:
                    year, month = re.findall(r"(\d{4})_(\d{2})", file)[0]
                    year, month = int(year), int(month)

                    if month not in self.date_dict[year]:
                        self.date_dict[year].append(month)
                
                # Rest of the method remains the same
                # Sort the months for each year
                for year in self.date_dict:
                    self.date_dict[year] = sorted(self.date_dict[year])
                
                # Find the date range
                min_year, min_month = min((year, min(months)) for year, months in self.date_dict.items())
                max_year, max_month = max((year, max(months)) for year, months in self.date_dict.items())
                
                start_date = f"{min_year}-{min_month:02d}"
                end_date = f"{max_year}-{max_month:02d}"
                
                st.html(
                    f"""
                    <div style="
                        padding: 10px;
                        border-radius: 5px;
                        background-color: #e6f7ff;
                        color: #000000;
                        border: 1px solid #b8daff;
                        font-size: 16px;
                        ">
                        ✅ Data available from <b>{start_date}</b> to <b>{end_date}</b>
                    </div>
                    """
                )
                return start_date, end_date
            else:
                st.warning("⚠️ No matched data files found.")
                return None, None
        
        except Exception as e:
            st.error(f"❌ Error while processing date range: {e}")
            return None, None
        

    def load_csv(self, file_name, subfolder=None):
        """
        Load a CSV file into a Pandas DataFrame.
        Only displays a warning or error message if something goes wrong.
        
        Args:
            file_name: Name of the CSV file to load
            subfolder: Optional subfolder within VIEWER_FOLDER_NAME
        """
        if subfolder:
            file_path = os.path.join(VIEWER_FOLDER_NAME, subfolder, file_name)
        else:
            file_path = os.path.join(VIEWER_FOLDER_NAME, file_name)
            
        try:
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                st.warning(f"⚠️ File `{file_name}` not found in `{os.path.dirname(file_path)}`.")
                return None
        
        except Exception as e:
            st.error(f"❌ Error while loading file `{file_name}`: {e}")
            return None