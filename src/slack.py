import yaml
import json
import requests
import os

my_path = os.path.abspath(os.path.dirname(__file__))

#CONFIG_PATH = os.path.join(my_path, "../../config/config.yml")
CONFIG_PATH ="/media/ubuntu/Local Disk/MAJOR_PROJECT/REIA/config/config.yml"

with open(CONFIG_PATH,"r") as config_file:
	config = yaml.load(config_file)

def post_message(message):
	print(message)
	

def read_message():
	with open('/media/ubuntu/Local Disk/MAJOR_PROJECT/REIA/mqueue.txt', 'r') as f:
			first_line = f.readline()
	user_input = first_line.split(' ', 1)[1]
	return user_input

def upload_file(file_path):
	print(file_path)
	f = {'file': (file_path, open(file_path, 'rb'), {'Expires':'0'})}
	r = requests.post(url='https://slack.com/api/files.upload', data=
		{'token': config['slack']['slack_token'], 'channels': config['slack']['channel'], 'media': f}, 
		headers={'Accept': 'application/json'}, files=f)
	return r

def get_username(user_id):
	payload = {'token': config['slack']['slack_token'], 'user': user_id}
	r = requests.post(config['slack']['user_info'], params=payload)
	return r.json()['user']['name']