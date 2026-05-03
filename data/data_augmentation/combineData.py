import sys
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) == 2:
        raise ValueError("Not enough arguments provided. Provide at least two files to combine. Format: python combineData.py <file1> <file2> [<file3> ...] [-o output_filepath]")
    dataframe = None
    output_path = "./data_combined.csv"
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-o':
            i += 1
            output_path = sys.argv[i]
        else:
            if dataframe is None:
                dataframe = pd.read_csv(sys.argv[i])
            else:
                file = pd.read_csv(sys.argv[i])
                dataframe = pd.concat([dataframe,file],axis=0,ignore_index=True)
        i += 1
    dataframe.to_csv(output_path,index=False)