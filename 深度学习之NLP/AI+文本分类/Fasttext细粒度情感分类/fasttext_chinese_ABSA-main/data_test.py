from util import *

train_data_df = load_data_from_csv('./data/sentiment_analysis_trainingset.csv')

# print(train_data_df)

content_train = train_data_df.iloc[:, 1]


# content_train = seg_words(content_train)
# print(content_train)

columns = train_data_df.columns.values.tolist()
print(columns[2:])