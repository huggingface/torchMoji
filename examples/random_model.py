# from label_data import check_image_exists
from pandas import DataFrame, read_csv, concat
import glob

def check_image_exists(unique_image_id,path):

    if glob.glob(path + f"/{unique_image_id}.png", recursive=True):
        return True
    
    return False


def return_valid_frame(dataframe, path):
        valid = DataFrame()
        for _, row in dataframe.iterrows():
                if check_image_exists(row.image_id, path):
                        valid = valid.append(row)

        # valid['2'] = 0
        valid['11'] = 0
        valid['48'] = 0

        new_cols = ['image_id', 'emoji']+[str(i) for i in range(64)]
        valid = valid[new_cols]

        return valid

# if __name__ == '__main__':
#         data = read_csv('real.csv')
#         print(len(data))
#         new_data = return_valid_frame(data)

#         print(len(new_data))
        # print(len(new_data.columns))
        # print(new_data.head())
