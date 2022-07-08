from label_data import IMAGE_PATH, check_image_exists
from pandas import DataFrame, read_csv, concat

def return_valid_frame(dataframe):
        valid = DataFrame()
        for _, row in dataframe.iterrows():
                if check_image_exists(row.image_id):
                        valid = valid.append(row)
        return valid

# if __name__ == '__main__':
#         data = read_csv('real.csv')
#         print(len(data))
#         new_data = return_valid_frame(data)

#         print(len(new_data))
        # print(len(new_data.columns))
        # print(new_data.head())
