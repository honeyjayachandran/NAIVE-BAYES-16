library(e1071)
library(gmodels)
sms <- read.csv(file = "C:\\Users\\Sony\\Downloads\\naive bayes assign-16\\sms_raw_NB.csv", stringsAsFactors = F)
str(sms)

sms$type <- as.factor(sms$type)
str(sms)
table(sms$type) #no. of ham and spam

library(tm)
#We need to build a corpus, which is a collection of documents, from the texts.
sms_corp <- VCorpus(VectorSource(sms$text))
sms_corp
inspect(sms_corp[5:10])
as.character(sms_corp[5])
lapply(sms_corp[1:4], as.character)

sms_corpus_clean <- tm_map(sms_corp,content_transformer(tolower)) #lowercase
as.character(sms_corp[[3]])
as.character(sms_corpus_clean[[3]])  
#We need to clean the corpus using tm_map():
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)# remove digits
sms_corpus_clean <- tm_map(sms_corpus_clean,removeWords, stopwords()) # removing stopwords
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation) # removing punctuations

install.packages("SnowballC")
library(SnowballC)
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)# removing whitespaces

lapply(sms_corp[1:4], as.character)
lapply(sms_corpus_clean[1:4], as.character)

#prepare document term matrix
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm
#splitting into test and train data
sms_dtm_train <- sms_dtm[1:4170, ]
sms_dtm_test <- sms_dtm[4171:5559, ]

sms_train_labels <- sms[1:4170, ]$type
sms_test_labels <- sms[4171:5559, ]$type


##checking the proportion of ham and spam in the training and test sets:
prop.table(table(sms_train_labels))#HAM=86.5% AND SPAM=13.5%
prop.table(table(sms_test_labels)) #TEST HAM=86.5% AND SPAM 13.5%

library(wordcloud)
windows()
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)#CREATING WORLD CLOUD

spam <- subset(sms, type == "spam")
ham <- subset(sms, type == "ham")

windows()
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

sms_freq_words <- findFreqTerms(sms_dtm_train, 10)
sms_dtm_freq_train<- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

#DTM only has numerics. Define a function which converts counts to yes/no factor, and apply it to our dtm train and test sets.
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

#applying the function to convert to factor to the dtm
# MARGIN = 1 is for rows, and 2 for columns
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,convert_counts) #applying to the dtm train data
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,convert_counts) #applying to the dtm test data

# building a Naive Bayes Model using dtm train data to predict whether the text is ham or spam
sms_nb <- sms_classifier <- naiveBayes(sms_train, sms_train_labels)  #building a model using the training
sms_test_pred <- predict(sms_nb, sms_test)  #predicting the type of text using the test data and the model we created

# create crosstables to see the number of accurate predictions and to get the accuracy of our Naive Bayes Model
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, 
           prop.t = FALSE, dnn = c('predicted', 'actual'))
t <- table(sms_test_pred, sms_test_labels)
t
confusionMatrix(t)
# accuracy 97.41%