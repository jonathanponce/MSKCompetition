library('tidytext')
library('tibble')
library('readr')
library('stringr')
library('SnowballC')
library('tidyr')
library('dplyr')
library('forcats')

#load and clean the data
train_txt_dump <- tibble(text = read_lines('./training_text/training_text', skip = 1))
train_txt <- train_txt_dump %>%
  separate(text, into = c("ID", "txt"), sep = "\\|\\|")
train_txt <- train_txt %>%
  mutate(ID = as.integer(ID))
train_txt <- train_txt %>%
  mutate(txt = gsub("-", "", txt))
train_txt <- train_txt %>%
  mutate(txt = gsub("_", "", txt))

test_txt_dump <- tibble(text = read_lines('./test_text/test_text', skip = 1))
test_txt <- test_txt_dump %>%
  separate(text, into = c("ID", "txt"), sep = "\\|\\|")
test_txt <- test_txt %>%
  mutate(ID = as.integer(ID))
test_txt <- test_txt %>%
  mutate(txt = gsub("-", "", txt))
test_txt <- test_txt %>%
  mutate(txt = gsub("_", "", txt))

train <- read_csv('./training_variants/training_variants')
test  <- read_csv('./test_variants/test_variants')
train <- train %>%
  mutate(Gene = factor(Gene),
         Variation = factor(Variation),
         Class = factor(Class))

test <- test %>%
  mutate(Gene = factor(Gene),
         Variation = factor(Variation))
#tokenize         
t1 <- train_txt %>% select(ID, txt) %>% unnest_tokens(word, txt)

data("stop_words")
my_stopwords <- data_frame(word = c(as.character(1:100),
                                    "fig", "figure", "et", "al", "table",
                                    "data", "analysis", "analyze", "study",
                                    "method", "result", "conclusion", "author",
                                    "find", "found", "show", "perform",
                                    "demonstrate", "evaluate", "discuss"))
my_not_stopwords <- data_frame(word = c(as.character(1:100),
                                    "no", "not"))                                    
stop_words <- stop_words %>%
  anti_join(my_not_stopwords, by = "word")
#remove stop words
t1 <- t1 %>%
  anti_join(stop_words, by = "word") %>%
  anti_join(my_stopwords, by = "word") %>%
  filter(str_detect(word, "[a-z]"))
  
#standarize data using wordstem
t1 <- t1 %>%
  mutate(word = wordStem(word))
foo <- train %>%
  select(ID, Class)
t1_class <- full_join(t1, foo, by = "ID")
frequency <-t1_class %>%
  count(Class, word)
  
#tf_idf
tf_idf <- frequency %>%
  bind_tf_idf(word, Class, n)
top <- tf_idf %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>%
  group_by(Class) %>%
  top_n(500, tf_idf) %>%
  ungroup() %>%
  select(word)

#save the new reformatted summarized text
top<-reshape2::melt(unique(top)) 
top<- top %>% mutate(word=sapply(word,as.character))

count=0+count(train_txt)
count=count$n[[1]]
count=count-1
write("ID,Text", file = "train_txt_new",append = TRUE, sep = "")
for (val in 0:count){
temp<-train_txt %>%
  filter(ID==val) %>% 
  select(txt) %>% 
  unnest_tokens(word, txt) %>%
  anti_join(stop_words, by = "word") %>%
  anti_join(my_stopwords, by = "word") %>%
  filter(str_detect(word, "[a-z]")) %>%
  mutate(word = wordStem(word)) %>%
  intersect(top)
temp<- paste(temp,collapse='')
temp<- gsub("\"","",temp) 
temp<- gsub(",","",temp)
temp <- gsub("c\\(","",temp)
temp <- gsub("\\)","",temp)
temp <- gsub("\\)","",temp)
temp <- gsub("\\n","",temp)
temp <-iconv(temp, "latin1", "ASCII", sub="")
temp<- paste(paste(val,"||", sep=""),temp, sep="")
write(temp, file = "train_txt_new",append = TRUE, sep = "")
print(val)
 }
 

t2 <- test_txt %>% select(ID, txt) %>% unnest_tokens(word, txt)
t1 <- t2 %>%
  anti_join(stop_words, by = "word") %>%
  anti_join(my_stopwords, by = "word") %>%
  filter(str_detect(word, "[a-z]"))
t1 <- t1 %>%
  mutate(word = wordStem(word))

count=0+count(test_txt)
count=count$n[[1]]
count=count-1
write("ID,Text", file = "test_txt_new",append = TRUE, sep = "")
for (val in 0:count){
temp<-test_txt %>%
  filter(ID==val) %>% 
  select(txt) %>% 
  unnest_tokens(word, txt) %>%
  anti_join(stop_words, by = "word") %>%
  anti_join(my_stopwords, by = "word") %>%
  filter(str_detect(word, "[a-z]")) %>%
  mutate(word = wordStem(word)) %>%
  intersect(top)
temp<- paste(temp,collapse='')
temp<- gsub("\"","",temp) 
temp<- gsub(",","",temp)
temp <- gsub("c\\(","",temp)
temp <- gsub("\\)","",temp)
temp <- gsub("\\)","",temp)
temp <- gsub("\\n","",temp)
temp <-iconv(temp, "latin1", "ASCII", sub="")
temp<- paste(paste(val,"||", sep=""),temp, sep="")
write(temp, file = "test_txt_new",append = TRUE, sep = "")
print(val)
 }
 