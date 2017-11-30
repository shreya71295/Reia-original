import yaml
import sys
import random
import nltk
import operator
import jellyfish as jf
import json
import requests
import os
import time
import signal
import subprocess
from nltk.tag import StanfordPOSTagger
from textblob.classifiers import NaiveBayesClassifier
#from execute import construct_command
from feedback import get_user_feedback
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from slack import post_message
from slack import read_message
from slack import upload_file
import os
import sys
import yaml
import json
import requests
from slack import post_message
from slack import read_message
from slack import upload_file

command = ""
pref = ""

my_path = os.path.abspath(os.path.dirname(__file__))

COMMAND_PATH = os.path.join(my_path, "../data/command.json")
#FILES_PATH = os.path.join(my_path, "../utilities/files")
FILES_PATH= "/media/ubuntu/Local Disk/MAJOR_PROJECT/REIA/utilities/files"
#my_path = os.path.abspath(os.path.dirname(__file__))

CONFIG_PATH = os.path.join(my_path, "../config/config.yml")
MAPPING_PATH = os.path.join(my_path, "../data/mapping.json")
TRAINDATA_PATH = os.path.join(my_path, "../data/traindata.txt")
LABEL_PATH = os.path.join(my_path, "../data/")

sys.path.insert(0, LABEL_PATH)
import trainlabel

with open(CONFIG_PATH,"r") as config_file:
	config = yaml.load(config_file)

os.environ['STANFORD_MODELS'] = config['tagger']['path_to_models']

exec_command = config['preferences']['execute']

def classify(text):
	X_train = np.array([line.rstrip('\n') for line in open(TRAINDATA_PATH)])
	y_train_text = trainlabel.y_train_text
	X_test = np.array([text])
	target_names = ['file', 'folder', 'network', 'system', 'general']

	lb = preprocessing.MultiLabelBinarizer()
	Y = lb.fit_transform(y_train_text)

	classifier = Pipeline([
		('vectorizer', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', OneVsRestClassifier(LinearSVC()))])

	classifier.fit(X_train, Y)
	predicted = classifier.predict(X_test)
	all_labels = lb.inverse_transform(predicted)

	for item, labels in zip(X_test, all_labels):
		return (', '.join(labels))

def suggestions(suggest_list):
	suggest = (sorted(suggest_list,reverse=True)[:5])
	return suggest

def consume_message():
	cmd = "sed -i -e \"1d\" /media/ubuntu/Local Disk/MAJOR_PROJECT/REIA/mqueue.txt"
	proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)  	
	(out, err) = proc.communicate()

GOOGLE_PATH = os.path.join(my_path, "../utilities/google")

def construct_command2(user_input,label,tokens,mapping,tags):
	sys.path.insert(0, FILES_PATH)
	from home_search import search_home
	with open(COMMAND_PATH,'r') as cmd:
		data = json.load(cmd)
	pref = data['network'][mapping]
	print("printing pref:", pref)
	if pref == 'ifconfig':
		command = "ifconfig"
		print("command is ", command)
		return(command,0)
	if pref == 'google':
		search_term = ""
		stoppers = ["google","for","web","search","on"]
		for word in tokens:
			if word not in stoppers:
				sys.path.insert(0, GOOGLE_PATH)
				from gsearch import call_search
				search_term += word+" "
				search_result = call_search(search_term)
				command = search_result
		return(command,0)
def construct_command(user_input,label,tokens,mapping,tags):
	print("in folder.py")
	sys.path.insert(0, FILES_PATH)
	from home_search import search_home
	with open(COMMAND_PATH,'r') as cmd:
		data = json.load(cmd)
	source = ""
	dest = ""
	name = ""
	pref = data['folder'][mapping]
	print("printing pref:", pref)
	
	if pref == 'mv' or pref == 'cp -r':
		for pos,tag in enumerate(tags):
			if (tag[1] == 'NNP' or tag[1] == 'NN') and tokens[pos-1] == 'from':
				source = multiple_paths(search_home(tag[0],"~"))
				name = tokens[pos-2]
				if(search_home(name,source) == ""):
					command = "Could not locate folder"
					return(command,0)				
			if (tag[1] == 'NNP' or tag[1] == 'NN') and tokens[pos-1] == 'to':
				dest = multiple_paths(search_home(tag[0],"~"))
			if('from' not in tokens and tokens[pos-1] == 'to'):
				name = multiple_paths(search_home(tokens[pos-2],"~"))
				dest = multiple_paths(search_home(tag[0]),"~")
			command = str(pref)+" "+source+"/"+name+" "+dest
		return(command,0)
	if pref == 'mkdir':
		print("Inside if condition")
		dest = ""
		name = ""
		for item in tags:
			print(item, " ")
		for pos,tag in enumerate(tags):
			if(tag[1] == 'NNP' or tag[1] == 'NN') and tokens[pos-1] == 'in':
				dest = multiple_paths(search_home(tag[0],"~"))
				if(dest == ""):
					command = "Could not locate folder in home directory"
					return(command,0)
			if(tag[1] == 'NNP' or tag[1] == 'NN') and tokens[pos-1] == 'folder':
				name = tag[0]
			command = str(pref)+" "+dest+"/"+name
		return(command,0)
	if pref == 'rm -rf':
		flag = 0
		for pos,tag in enumerate(tags):
			if (tag[1] == 'NNP' or tag[1] == 'NN') and tokens[pos-1] == 'from':
				flag = 1
				name = tokens[pos-2]
				dest = multiple_paths(search_home(tokens[pos],"~"))
		if flag == 0:
			for pos,tag in enumerate(tags):
				if(tag[1] == 'NNP' or tag[1] == 'NN') and tokens[pos-1] == 'folder':
					name = ""
					dest = multiple_paths(search_home(tag[0],"~"))
		command = str(pref)+" "+dest+"/"+name
		return(command,0)
	if pref == "ls":
		if "contents" in tokens:
			index = tokens.index("contents")
			if "folder" in tokens:
				indx = tokens.index("folder")
				name = tokens[indx+1]
			else:
				name = tokens[index+2]
			folder_path = multiple_paths(search_home(name,"~"))
		command = str(pref)+" "+folder_path
		return(command,0)
	if pref == "find -name":
		index = 0
		indx = 0
		if "folder" in tokens:
			index = tokens.index("folder")
		elif "directory" in tokens:
			index = tokens.index("directory")
		else:
			command = "Invalid Syntax!"
			return(command,0)
		if "in" in tokens:
			indx = tokens.index("in")
			name = tokens[index+1]
			dest = tokens[indx+1]
		else:
			name = tokens[index+1]
			dest= "~"
		print("Dest = "+multiple_paths(dest))
		folder_path = multiple_paths(search_home(name,multiple_paths(search_home(dest,"~"))))
		command = folder_path
		return(command,0)


with open('/media/ubuntu/Local Disk/MAJOR_PROJECT/REIA/mqueue.txt', 'r') as f:
	first_line = f.readline()

user_input = first_line.split(' ', 1)[1]
max_score = 0.1
map_val = ""
print("\nINPUT = ")
print(user_input)
label = classify(user_input)
suggest_list = []
suggest_message = ""
print("Classified as : "+str(label))
tokens = nltk.word_tokenize(user_input)
print(tokens)
st = StanfordPOSTagger(config['tagger']['model'],path_to_jar=config['tagger']['path'])
stanford_tag = st.tag(user_input.split())
print("Tags")
print(stanford_tag)
with open(MAPPING_PATH,'r') as data_file:    
	data = json.load(data_file)	
for i in data[label]:
	dist = jf.jaro_distance(unicode(str(user_input),encoding="utf-8"), unicode(str(i),encoding="utf-8"))
	suggest_list.append(tuple((dist,i)))
	print(dist)
	if(dist > max_score):
		max_score = dist
		map_val = i
if max_score < config['preferences']['similarity_threshold']:
	post_message("Sorry, I could not understand. Please rephrase and try again.")
	consume_message()
	if config['preferences']['suggestions'] == True:
		suggest = suggestions(suggest_list)
		post_message("Did you mean :")
		for i in suggest:
			suggest_message += (str(i[1])+"\n")
		post_message(suggest_message)
	   
print("\nMapped to : "+map_val)


#construct_command(user_input,label,tokens,map_val,stanford_tag,exec_command)


command,response = construct_command2(user_input,label,tokens,map_val,stanford_tag)
print(command)
print(response)