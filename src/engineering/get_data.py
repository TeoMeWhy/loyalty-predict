import dotenv
import shutil

dotenv.load_dotenv('../../.env')

from kaggle import api

datasets = [
    'teocalvo/teomewhy-loyalty-system',
    'teocalvo/teomewhy-education-platform',    
]

for d in datasets:
    dataset_name = d.split("teomewhy-")[-1]
    print(dataset_name)
    path = f'../../data/{dataset_name}/database.db'
    api.dataset_download_file(d, 'database.db')
    shutil.move("database.db", path)
