from config.config_loader import config_loader

config = config_loader()

collection_name = config['astra_db']['collection_name']

print(collection_name)

import os 
print(os.getcwd())


print(os.getenv('ASTRA_DB_API_ENDPOINT'))