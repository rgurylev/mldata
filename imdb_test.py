from imdb import IMDB

data = IMDB()
data.load(path="E:\\dev\\PycharmProjects\\LearningPyTorch\\data\\imdb\\aclImdb.zip")
text_batch, label_batch, length_batch = next(iter( data.loader()))
print(text_batch)
print(label_batch)
print(length_batch)
print(text_batch.shape)
print (len(data.vocab))