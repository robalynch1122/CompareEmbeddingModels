import pandas as pd
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import concurrent.futures
import time
import openai
openai.api_key = ""

# load models and tokenizers to be tested
MPNETmodel = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
MPNETtokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

BERTBasemodel = AutoModel.from_pretrained('bert-base-uncased')
BERTBasetokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_csv(r'SentimentAnalysis.csv')

# randomized the data and reset the index
df = df.sample(frac=1).reset_index(drop=True)

# split the data into train and test sets
test_set_fraction = 0.2
df_train = df[0:int(len(df)*(1-test_set_fraction))]
df_test = df[int(len(df)*(1-test_set_fraction)):]

# set the input and target variables (change these to match your data)
input_variable = 'text'
target_variable = 'category'

# function to get the embedding from a text using a model (this will be used by multi-threading to speed up
def get_embedding_from_text_model(text, model):
	if model=='MPNET':
		return MPNETmodel(**MPNETtokenizer(text, return_tensors='pt'))[0][0][0].detach().numpy()
	elif model=='BERTBase':
		return BERTBasemodel(**BERTBasetokenizer(text, return_tensors='pt'))[0][0][0].detach().numpy()
	elif model=='OpenAIAda':
		return list(openai.embeddings.create(input=text, model='text-embedding-ada-002').data[0].embedding)

# list of models to compare
models = ['MPNET', 'BERTBase','OpenAIAda']

# get the embeddings for the train data
for model in models:
	print(model)
	start = time.perf_counter()
	embeddings = []
	with concurrent.futures.ThreadPoolExecutor() as executor:
		results = list(tqdm(executor.map(get_embedding_from_text_model, df_train[input_variable], [model]*len(df_train[input_variable])), total=len(df_train[input_variable])))
		embeddings.extend(results)
	end = time.perf_counter()
	print(f'Finished in {end-start} seconds')
	df_train['embeddings' + model] = embeddings

# create an empty list to store the trained models
clf_list = []

# train one multi-class logistic regression model per embedding model to predict the label from the respective embeddings
for model in models:
	print(model)
	X_train = list(df_train['embeddings' + model])
	y_train = list(df_train[target_variable])
	clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

	# add the trained model to a list of models so we can use it later
	clf_list.append(clf)

# get the embeddings for the test data
for model in models:
	print(model)
	start = time.perf_counter()
	embeddings = []
	with concurrent.futures.ThreadPoolExecutor() as executor:
		results = list(tqdm(executor.map(get_embedding_from_text_model, df_test[input_variable], [model]*len(df_test[input_variable])), total=len(df_test[input_variable])))
		embeddings.extend(results)
	end = time.perf_counter()
	print(f'Finished in {end-start} seconds')
	df_test['embeddings' + model] = embeddings

# for each trained logistic model (and embedding model), predict the label for the test data and print the per category F1 score
for model, clf in zip(models, clf_list):
	print(model)
	y_test = list(df_test[target_variable])
	X_test = list(df_test['embeddings' + model])
	y_pred = clf.predict(X_test)
	print('Accuracy: ', accuracy_score(y_test, y_pred))
	print('F1 score: ', f1_score(y_test, y_pred, average='weighted'))
	print('F1 score for each category:')
	for category in df_test[target_variable].unique():
		print(category, f1_score(y_test, y_pred, average='weighted', labels=[category]))