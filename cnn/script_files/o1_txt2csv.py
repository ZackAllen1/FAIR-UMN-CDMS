"""
Saves the raw text file containing all 24 features,
row number and the ground truth y into a CSV file.
"""
import os
import pandas as pd

def process_line(line):
    info = []
    for item in line:
        if len(item) == 0:
            continue
        else:
            info.append(item)
    return info

def raw_txt_to_csv(raw_txt_filepath: str, extracted_csv_filepath: str):
    """
    This function takes the filepath in str format
    and returns a pd.DataFrame
    """
    combined_list = []
    header = None
    with open(raw_txt_filepath, "r") as f:
        
        for line in f:
            line = line.strip()
            check_line = line.replace("*", "")
            # print(line)
            
            # Skip any empty lines
            if check_line == "":
                continue
            
            # Skip the line separator
            if "==>" in line:
                continue
            
            line = check_line.split(' ')
            info = process_line(line)
            
            if header is None:
                header = info
                combined_list.append(info)
            elif info == header:
                continue
            else:
                combined_list.append(info)
            
        # Create the dataframe using first row as the column    
        result_df = pd.DataFrame(combined_list[1:], columns=combined_list[0])
        
        # Convert all string values to numeric
        result_df = result_df.apply(pd.to_numeric)
        # print(result_df.columns)
        
        # Save to csv
        result_df.to_csv(extracted_csv_filepath, index=False)

    print('>>>The raw txt has been processed successfully!')


if __name__ == "__main__":
    data_folder_name = "data"
    
    raw_txt_file = "data_all_v4.txt"
    raw_txt_filepath = f"../{data_folder_name}/{raw_txt_file}"
    
    csv_filename = "data_all_v4.csv"
    extracted_csv_filepath = f"../{data_folder_name}/{csv_filename}"
    
    raw_txt_to_csv(raw_txt_filepath = raw_txt_filepath, extracted_csv_filepath = extracted_csv_filepath)