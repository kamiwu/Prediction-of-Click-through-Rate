# ISOM 674 Machine Learning Final Project
# Group 11: Kwaku Danso-Manu, Yan Li, Miller Page, Kami Wu

# load packages
if(!require("data.table")) { install.packages("data.table"); require("data.tables") }
if(!require("dplyr")) { install.packages("dplyr"); require("dplyr") }
if(!require("fastDummies")) { install.packages("fastDummies"); require("fastDummies") }
if(!require("caret")) { install.packages("caret"); require("caret") }
if(!require("stringr")) { install.packages("stringr"); require("stringr") }
if(!require("e1071")) { install.packages("e1071"); require("e1071") }
if(!require("tree")) { install.packages("tree"); require("tree") }
if(!require("rpart")) { install.packages("rpart"); require("rpart") }
if(!require("randomForest")) { install.packages("randomForest"); require("randomForest") }
if(!require("FNN")) { install.packages("FNN"); require("FNN") }
if(!require("glmnet")) { install.packages("glmnet"); require("glmnet") }
if(!require("MLmetrics")) { install.packages("MLmetrics"); require("MLmetrics") }

# load data sets
traindata <- fread("ProjectTrainingData.csv")
testdata <- fread("ProjectTestData.csv")
submission <- fread("ProjectSubmission-TeamX.csv")


# Part 1: Feature engineering ---------------------------------------------------------
# a. Transform variable "hour" 
## extract date and hour information from variable "hour"
traindata$hours <- str_sub(traindata$hour,-2,-1)
traindata$date <- str_sub(traindata$hour,-4,-3)
## trnasform hour into 8 date time intervals
traindata$time <- "beforedawn"
traindata$time[traindata$hours %in% c("03","04","05")] <- "dawn"
traindata$time[traindata$hours %in% c("06","07","08")] <- "morning"
traindata$time[traindata$hours %in% c("09","10","11")] <- "forenoon"
traindata$time[traindata$hours %in% c("12","13","14")] <- "noon"
traindata$time[traindata$hours %in% c("15","16","17")] <- "afternoon"
traindata$time[traindata$hours %in% c("18","19","20")] <- "night"
traindata$time[traindata$hours %in% c("21","22","23")] <- "midnight"
## transform date into weekday & weekend
traindata$day <- "weekday"
traindata$day[traindata$date %in% c(25,26)] <- "weekend"

## do the same thing for the test set
testdata$hours <- str_sub(testdata$hour,-2,-1)
testdata$date <- str_sub(testdata$hour,-4,-3)
testdata$time <- "beforedawn"
testdata$time[testdata$hours %in% c("03","04","05")] <- "dawn"
testdata$time[testdata$hours %in% c("06","07","08")] <- "morning"
testdata$time[testdata$hours %in% c("09","10","11")] <- "forenoon"
testdata$time[testdata$hours %in% c("12","13","14")] <- "noon"
testdata$time[testdata$hours %in% c("15","16","17")] <- "afternoon"
testdata$time[testdata$hours %in% c("18","19","20")] <- "night"
testdata$time[testdata$hours %in% c("21","22","23")] <- "midnight"
testdata$day <- "weekday"
testdata$day[testdata$date %in% c(25,26)] <- "weekend"

## delete useless variables
origin_train <- traindata[,-c("id","hour","date","hours")]
origin_test <- testdata[,-c("id","hour","date","hours")]


# b. Regroup categorical variables with number of categories above 10
## change site_id into 8 groups based on the clickrate of each site_id, set low-frequency site_id as "other" group
### count the frequency of each site_id
site_id_dist <- origin_train %>% count(site_id, sort = TRUE) 
### examine the basic statistics, find a rational threshold
summary(site_id_dist)
### base on the statistics value, 35 which is the median seems to be a rational threshold
site_id_highfreq <- site_id_dist %>% filter(n > 35) 
### calculate the click-through rate of high-frequency site_id
site_id_clickrate <- origin_train %>% filter(site_id %in% (site_id_highfreq$site_id)) %>% group_by(site_id) %>% summarize(clickrate = mean(click)) 
### assign new groups based on click-though rate
site_id_clickrate$group <- 0
site_id_clickrate$group[(site_id_clickrate$clickrate > 0.1) & (site_id_clickrate$clickrate <= 0.2)] <- 1
site_id_clickrate$group[(site_id_clickrate$clickrate > 0.2) & (site_id_clickrate$clickrate <= 0.3)] <- 2
site_id_clickrate$group[(site_id_clickrate$clickrate > 0.3) & (site_id_clickrate$clickrate <= 0.4)] <- 3
site_id_clickrate$group[(site_id_clickrate$clickrate > 0.4) & (site_id_clickrate$clickrate <= 0.5)] <- 4
site_id_clickrate$group[(site_id_clickrate$clickrate > 0.5) & (site_id_clickrate$clickrate <= 0.6)] <- 5
site_id_clickrate$group[(site_id_clickrate$clickrate > 0.6) & (site_id_clickrate$clickrate <= 0.7)] <- 6
site_id_clickrate$group[site_id_clickrate$clickrate > 0.7] <- 7
### create a new data set contains site_id and new groups for high-frequency categories
site_id_clickgroup <- site_id_clickrate %>% select(site_id, group)
### merge this new data set with original training data
train_cleansiteid <- left_join(origin_train, site_id_clickgroup, by = "site_id")
### delete original site_id, set low-frequency categories as "other"
train_cleansiteid <- train_cleansiteid %>% select(-site_id) %>% rename(site_id_group = group)
train_cleansiteid$site_id_group[is.na(train_cleansiteid$site_id_group)] <- "other"


## do the same thing for site_domain
site_domain_dist <- origin_train %>% count(site_domain, sort = TRUE)
summary(site_domain_dist)
### base on the statistics value, 62 which is the median seems to be a rational threshold
site_domain_highfreq <- site_domain_dist %>% filter(n > 62)
site_domain_clickrate <- origin_train %>% filter(site_domain %in% (site_domain_highfreq$site_domain)) %>% group_by(site_domain) %>% summarize(clickrate = mean(click))
site_domain_clickrate$group <- 0
site_domain_clickrate$group[(site_domain_clickrate$clickrate > 0.1) & (site_domain_clickrate$clickrate <= 0.2)] <- 1
site_domain_clickrate$group[(site_domain_clickrate$clickrate > 0.2) & (site_domain_clickrate$clickrate <= 0.3)] <- 2
site_domain_clickrate$group[(site_domain_clickrate$clickrate > 0.3) & (site_domain_clickrate$clickrate <= 0.4)] <- 3
site_domain_clickrate$group[(site_domain_clickrate$clickrate > 0.4) & (site_domain_clickrate$clickrate <= 0.5)] <- 4
site_domain_clickrate$group[(site_domain_clickrate$clickrate > 0.5) & (site_domain_clickrate$clickrate <= 0.6)] <- 5
site_domain_clickrate$group[(site_domain_clickrate$clickrate > 0.6) & (site_domain_clickrate$clickrate <= 0.7)] <- 6
site_domain_clickrate$group[site_domain_clickrate$clickrate > 0.7] <- 7

site_domain_clickgroup <- site_domain_clickrate %>% select(site_domain, group)

train_cleansite <- left_join(train_cleansiteid, site_domain_clickgroup, by = "site_domain")
train_cleansite <- train_cleansite %>% select(-site_domain) %>% rename(site_domain_group = group)
train_cleansite$site_domain_group[is.na(train_cleansite$site_domain_group)] <- "other"



## do the same thing for app id
app_id_dist <- origin_train %>% count(app_id, sort = TRUE)
summary(app_id_dist)
### base on the statistics value, 46 which is the third quartile seems to be a rational threshold
app_id_highfreq <- app_id_dist %>% filter(n > 46)
app_id_clickrate <- origin_train %>% filter(app_id %in% (app_id_highfreq$app_id)) %>% group_by(app_id) %>% summarize(clickrate = mean(click))
app_id_clickrate$group <- 0
app_id_clickrate$group[(app_id_clickrate$clickrate > 0.1) & (app_id_clickrate$clickrate <= 0.2)] <- 1
app_id_clickrate$group[(app_id_clickrate$clickrate > 0.2) & (app_id_clickrate$clickrate <= 0.3)] <- 2
app_id_clickrate$group[(app_id_clickrate$clickrate > 0.3) & (app_id_clickrate$clickrate <= 0.4)] <- 3
app_id_clickrate$group[(app_id_clickrate$clickrate > 0.4) & (app_id_clickrate$clickrate <= 0.5)] <- 4
app_id_clickrate$group[(app_id_clickrate$clickrate > 0.5) & (app_id_clickrate$clickrate <= 0.6)] <- 5
app_id_clickrate$group[(app_id_clickrate$clickrate > 0.6) & (app_id_clickrate$clickrate <= 0.7)] <- 6
app_id_clickrate$group[(app_id_clickrate$clickrate > 0.7) & (app_id_clickrate$clickrate <= 0.8)] <- 7
app_id_clickrate$group[(app_id_clickrate$clickrate > 0.8) & (app_id_clickrate$clickrate <= 0.9)] <- 8
app_id_clickrate$group[app_id_clickrate$clickrate > 0.9] <- 9

app_id_clickgroup <- app_id_clickrate %>% select(app_id, group)

train_cleanappid <- left_join(train_cleansite, app_id_clickgroup, by = "app_id")
train_cleanappid <- train_cleanappid %>% select(-app_id) %>% rename(app_id_group = group)
train_cleanappid$app_id_group[is.na(train_cleanappid$app_id_group)] <- "other"


## do the same thing forapp domain
app_domain_dist <- origin_train %>% count(app_domain, sort = TRUE)
summary(app_domain_dist)
### base on the statistics value, 50 which is the third quartile seems to be a rational threshold
app_domain_highfreq <- app_domain_dist %>% filter(n > 50)
app_domain_clickrate <- origin_train %>% filter(app_domain %in% (app_domain_highfreq$app_domain)) %>% group_by(app_domain) %>% summarize(clickrate = mean(click))

app_domain_clickrate$group <- 0
app_domain_clickrate$group[(app_domain_clickrate$clickrate > 0.1) & (app_domain_clickrate$clickrate <= 0.2)] <- 1
app_domain_clickrate$group[(app_domain_clickrate$clickrate > 0.2) & (app_domain_clickrate$clickrate <= 0.3)] <- 2
app_domain_clickrate$group[(app_domain_clickrate$clickrate > 0.3) & (app_domain_clickrate$clickrate <= 0.4)] <- 3
app_domain_clickrate$group[(app_domain_clickrate$clickrate > 0.4) & (app_domain_clickrate$clickrate <= 0.5)] <- 4
app_domain_clickrate$group[(app_domain_clickrate$clickrate > 0.5) & (app_domain_clickrate$clickrate <= 0.6)] <- 5
app_domain_clickrate$group[app_domain_clickrate$clickrate > 0.6] <- 6

app_domain_clickgroup <- app_domain_clickrate %>% select(app_domain, group)

train_cleanappid <- left_join(train_cleanappid, app_domain_clickgroup, by = "app_domain")
train_cleanappid <- train_cleanappid %>% select(-app_domain) %>% rename(app_domain_group = group)
train_cleanappid$app_domain_group[is.na(train_cleanappid$app_domain_group)] <- "other"




## do the same thing for device_id
device_id_dist <- origin_train %>% count(device_id, sort = TRUE)
summary(device_id_dist)
### base on the statistics value, use 100 as threshold since the third quartile is 2
device_id_highfreq <- device_id_dist %>% filter(n > 100)
device_id_clickrate <- origin_train %>% filter(device_id %in% (device_id_highfreq$device_id)) %>% group_by(device_id) %>% summarize(clickrate = mean(click))
device_id_clickrate$group <- 0
device_id_clickrate$group[(device_id_clickrate$clickrate > 0.1) & (device_id_clickrate$clickrate <= 0.2)] <- 1
device_id_clickrate$group[(device_id_clickrate$clickrate > 0.2) & (device_id_clickrate$clickrate <= 0.3)] <- 2
device_id_clickrate$group[(device_id_clickrate$clickrate > 0.3) & (device_id_clickrate$clickrate <= 0.4)] <- 3
device_id_clickrate$group[(device_id_clickrate$clickrate > 0.4) & (device_id_clickrate$clickrate <= 0.5)] <- 4
device_id_clickrate$group[(device_id_clickrate$clickrate > 0.5) & (device_id_clickrate$clickrate <= 0.6)] <- 5
device_id_clickrate$group[(device_id_clickrate$clickrate > 0.6) & (device_id_clickrate$clickrate <= 0.7)] <- 6
device_id_clickrate$group[(device_id_clickrate$clickrate > 0.7) & (device_id_clickrate$clickrate <= 0.8)] <- 7
device_id_clickrate$group[(device_id_clickrate$clickrate > 0.8) & (device_id_clickrate$clickrate <= 0.9)] <- 8
device_id_clickrate$group[device_id_clickrate$clickrate > 0.9] <- 9

device_id_clickgroup <- device_id_clickrate %>% select(device_id, group)

train_cleanapp <- left_join(train_cleanappid, device_id_clickgroup, by = "device_id")
train_cleanapp <- train_cleanapp %>% select(-device_id) %>% rename(device_id_group = group)
train_cleanapp$device_id_group[is.na(train_cleanapp$device_id_group)] <- "other"



## do the same thing for device_ip
device_ip_dist <- origin_train %>% count(device_ip, sort = TRUE)
summary(device_ip_dist)
### base on the statistics value, use 100 as threshold since the third quartile is 3
device_ip_highfreq <- device_ip_dist %>% filter(n > 100)
device_ip_clickrate <- origin_train %>% filter(device_ip %in% (device_ip_highfreq$device_ip)) %>% group_by(device_ip) %>% summarize(clickrate = mean(click))
device_ip_clickrate$group <- 0
device_ip_clickrate$group[(device_ip_clickrate$clickrate > 0.1) & (device_ip_clickrate$clickrate <= 0.2)] <- 1
device_ip_clickrate$group[(device_ip_clickrate$clickrate > 0.2) & (device_ip_clickrate$clickrate <= 0.3)] <- 2
device_ip_clickrate$group[(device_ip_clickrate$clickrate > 0.3) & (device_ip_clickrate$clickrate <= 0.4)] <- 3
device_ip_clickrate$group[(device_ip_clickrate$clickrate > 0.4) & (device_ip_clickrate$clickrate <= 0.5)] <- 4
device_ip_clickrate$group[(device_ip_clickrate$clickrate > 0.5) & (device_ip_clickrate$clickrate <= 0.6)] <- 5
device_ip_clickrate$group[(device_ip_clickrate$clickrate > 0.6) & (device_ip_clickrate$clickrate <= 0.7)] <- 6
device_ip_clickrate$group[(device_ip_clickrate$clickrate > 0.7) & (device_ip_clickrate$clickrate <= 0.8)] <- 7
device_ip_clickrate$group[(device_ip_clickrate$clickrate > 0.8) & (device_ip_clickrate$clickrate <= 0.9)] <- 8
device_ip_clickrate$group[device_ip_clickrate$clickrate > 0.9] <- 9

device_ip_clickgroup <- device_ip_clickrate %>% select(device_ip, group)

train_clean <- left_join(train_cleanapp, device_ip_clickgroup, by = "device_ip")
train_clean <- train_clean %>% select(-device_ip) %>% rename(device_ip_group = group)
train_clean$device_ip_group[is.na(train_clean$device_ip_group)] <- "other"



## do the same thing for device_model
device_model_dist <- origin_train %>% count(device_model, sort = TRUE)
summary(device_model_dist)
### base on the statistics value, 67 which is the median seems to be a rational threshold
device_model_highfreq <- device_model_dist %>% filter(n > 67)
device_model_clickrate <- origin_train %>% filter(device_model %in% (device_model_highfreq$device_model)) %>% group_by(device_model) %>% summarize(clickrate = mean(click))
device_model_clickrate$group <- 0
device_model_clickrate$group[(device_model_clickrate$clickrate > 0.1) & (device_model_clickrate$clickrate <= 0.2)] <- 1
device_model_clickrate$group[(device_model_clickrate$clickrate > 0.2) & (device_model_clickrate$clickrate <= 0.3)] <- 2
device_model_clickrate$group[(device_model_clickrate$clickrate > 0.3) & (device_model_clickrate$clickrate <= 0.4)] <- 3
device_model_clickrate$group[(device_model_clickrate$clickrate > 0.4) & (device_model_clickrate$clickrate <= 0.5)] <- 4
device_model_clickrate$group[(device_model_clickrate$clickrate > 0.5) & (device_model_clickrate$clickrate <= 0.6)] <- 5
device_model_clickrate$group[(device_model_clickrate$clickrate > 0.6) & (device_model_clickrate$clickrate <= 0.7)] <- 6
device_model_clickrate$group[(device_model_clickrate$clickrate > 0.7) & (device_model_clickrate$clickrate <= 0.8)] <- 7
device_model_clickrate$group[(device_model_clickrate$clickrate > 0.8) & (device_model_clickrate$clickrate <= 0.9)] <- 8
device_model_clickrate$group[device_model_clickrate$clickrate > 0.9] <- 9

device_model_clickgroup <- device_model_clickrate %>% select(device_model, group)

train_clean <- left_join(train_clean, device_model_clickgroup, by = "device_model")
train_clean <- train_clean %>% select(-device_model) %>% rename(device_model_group = group)
train_clean$device_model_group[is.na(train_clean$device_model_group)] <- "other"



## do the same thing for C14
C14_dist <- origin_train %>% count(C14, sort = TRUE)
summary(C14_dist)
### base on the statistics value, 77 which is the first quartile seems to be a rational threshold
C14_highfreq <- C14_dist %>% filter(n > 77)
C14_clickrate <- origin_train %>% filter(C14 %in% (C14_highfreq$C14)) %>% group_by(C14) %>% summarize(clickrate = mean(click))
C14_clickrate$group <- 0
C14_clickrate$group[(C14_clickrate$clickrate > 0.1) & (C14_clickrate$clickrate <= 0.2)] <- 1
C14_clickrate$group[(C14_clickrate$clickrate > 0.2) & (C14_clickrate$clickrate <= 0.3)] <- 2
C14_clickrate$group[(C14_clickrate$clickrate > 0.3) & (C14_clickrate$clickrate <= 0.4)] <- 3
C14_clickrate$group[(C14_clickrate$clickrate > 0.4) & (C14_clickrate$clickrate <= 0.5)] <- 4
C14_clickrate$group[(C14_clickrate$clickrate > 0.5) & (C14_clickrate$clickrate <= 0.6)] <- 5
C14_clickrate$group[(C14_clickrate$clickrate > 0.6) & (C14_clickrate$clickrate <= 0.7)] <- 6
C14_clickrate$group[(C14_clickrate$clickrate > 0.7) & (C14_clickrate$clickrate <= 0.8)] <- 7
C14_clickrate$group[(C14_clickrate$clickrate > 0.8) & (C14_clickrate$clickrate <= 0.9)] <- 8
C14_clickrate$group[C14_clickrate$clickrate > 0.9] <- 9

C14_clickgroup <- C14_clickrate %>% select(C14, group)

train_clean <- left_join(train_clean, C14_clickgroup, by = "C14")
train_clean <- train_clean %>% select(-C14) %>% rename(C14_group = group)
train_clean$C14_group[is.na(train_clean$C14_group)] <- "other"



## do the same thing for C17
C17_dist <- origin_train %>% count(C17, sort = TRUE)
summary(C17_dist)
### base on the statistics value, use 100 as threshold since the first quartile is 2750
C17_highfreq <- C17_dist %>% filter(n > 100)
C17_clickrate <- origin_train %>% filter(C17 %in% (C17_highfreq$C17)) %>% group_by(C17) %>% summarize(clickrate = mean(click))
C17_clickrate$group <- 0
C17_clickrate$group[(C17_clickrate$clickrate > 0.1) & (C17_clickrate$clickrate <= 0.2)] <- 1
C17_clickrate$group[(C17_clickrate$clickrate > 0.2) & (C17_clickrate$clickrate <= 0.3)] <- 2
C17_clickrate$group[(C17_clickrate$clickrate > 0.3) & (C17_clickrate$clickrate <= 0.4)] <- 3
C17_clickrate$group[(C17_clickrate$clickrate > 0.4) & (C17_clickrate$clickrate <= 0.5)] <- 4
C17_clickrate$group[(C17_clickrate$clickrate > 0.5) & (C17_clickrate$clickrate <= 0.6)] <- 5
C17_clickrate$group[(C17_clickrate$clickrate > 0.6) & (C17_clickrate$clickrate <= 0.7)] <- 6
C17_clickrate$group[(C17_clickrate$clickrate > 0.7) & (C17_clickrate$clickrate <= 0.8)] <- 7
C17_clickrate$group[(C17_clickrate$clickrate > 0.8) & (C17_clickrate$clickrate <= 0.9)] <- 8
C17_clickrate$group[C17_clickrate$clickrate > 0.9] <- 9

C17_clickgroup <- C17_clickrate %>% select(C17, group)

train_clean <- left_join(train_clean, C17_clickgroup, by = "C17")
train_clean <- train_clean %>% select(-C17) %>% rename(C17_group = group)
train_clean$C17_group[is.na(train_clean$C17_group)] <- "other"



## do the same thing for C19
C19_dist <- origin_train %>% count(C19, sort = TRUE)
summary(C19_dist)
### base on the statistics value, use 100 as threshold since the first quartile is 13375
C19_highfreq <- C19_dist %>% filter(n > 100)
C19_clickrate <- origin_train %>% filter(C19 %in% (C19_highfreq$C19)) %>% group_by(C19) %>% summarize(clickrate = mean(click))
C19_clickrate$group <- 0
C19_clickrate$group[(C19_clickrate$clickrate > 0.1) & (C19_clickrate$clickrate <= 0.2)] <- 1
C19_clickrate$group[(C19_clickrate$clickrate > 0.2) & (C19_clickrate$clickrate <= 0.3)] <- 2
C19_clickrate$group[(C19_clickrate$clickrate > 0.3) & (C19_clickrate$clickrate <= 0.4)] <- 3
C19_clickrate$group[(C19_clickrate$clickrate > 0.4) & (C19_clickrate$clickrate <= 0.5)] <- 4
C19_clickrate$group[(C19_clickrate$clickrate > 0.5) & (C19_clickrate$clickrate <= 0.6)] <- 5
C19_clickrate$group[(C19_clickrate$clickrate > 0.6) & (C19_clickrate$clickrate <= 0.7)] <- 6
C19_clickrate$group[(C19_clickrate$clickrate > 0.7) & (C19_clickrate$clickrate <= 0.8)] <- 7
C19_clickrate$group[(C19_clickrate$clickrate > 0.8) & (C19_clickrate$clickrate <= 0.9)] <- 8
C19_clickrate$group[C19_clickrate$clickrate > 0.9] <- 9

C19_clickgroup <- C19_clickrate %>% select(C19, group)

train_clean <- left_join(train_clean, C19_clickgroup, by = "C19")
train_clean <- train_clean %>% select(-C19) %>% rename(C19_group = group)
train_clean$C19_group[is.na(train_clean$C19_group)] <- "other"



## do the same thing for C20
C20_dist <- origin_train %>% count(C20, sort = TRUE)
summary(C20_dist)
### base on the statistics value, use 100 as threshold since the first quartile is 2610
C20_highfreq <- C20_dist %>% filter(n > 100)
C20_clickrate <- origin_train %>% filter(C20 %in% (C20_highfreq$C20)) %>% group_by(C20) %>% summarize(clickrate = mean(click))
C20_clickrate$group <- 0
C20_clickrate$group[(C20_clickrate$clickrate > 0.1) & (C20_clickrate$clickrate <= 0.2)] <- 1
C20_clickrate$group[(C20_clickrate$clickrate > 0.2) & (C20_clickrate$clickrate <= 0.3)] <- 2
C20_clickrate$group[(C20_clickrate$clickrate > 0.3) & (C20_clickrate$clickrate <= 0.4)] <- 3
C20_clickrate$group[(C20_clickrate$clickrate > 0.4) & (C20_clickrate$clickrate <= 0.5)] <- 4
C20_clickrate$group[(C20_clickrate$clickrate > 0.5) & (C20_clickrate$clickrate <= 0.6)] <- 5
C20_clickrate$group[(C20_clickrate$clickrate > 0.6) & (C20_clickrate$clickrate <= 0.7)] <- 6
C20_clickrate$group[(C20_clickrate$clickrate > 0.7) & (C20_clickrate$clickrate <= 0.8)] <- 7
C20_clickrate$group[(C20_clickrate$clickrate > 0.8) & (C20_clickrate$clickrate <= 0.9)] <- 8
C20_clickrate$group[C20_clickrate$clickrate > 0.9] <- 9

C20_clickgroup <- C20_clickrate %>% select(C20, group)

train_clean <- left_join(train_clean, C20_clickgroup, by = "C20")
train_clean$group[train_clean$C20 == -1] <- "other"
train_clean <- train_clean %>% select(-C20) %>% rename(C20_group = group)
train_clean$C20_group[is.na(train_clean$C20_group)] <- "other"

View(head(train_clean))



## do the same thing for C21
C21_dist <- origin_train %>% count(C21, sort = TRUE)
summary(C21_dist)
### base on the statistics value, use 500 as threshold since the lowest frequency is 447
C21_highfreq <- C21_dist %>% filter(n > 500)
C21_clickrate <- origin_train %>% filter(C21 %in% (C21_highfreq$C21)) %>% group_by(C21) %>% summarize(clickrate = mean(click))
C21_clickrate$group <- 0
C21_clickrate$group[(C21_clickrate$clickrate > 0.1) & (C21_clickrate$clickrate <= 0.2)] <- 1
C21_clickrate$group[(C21_clickrate$clickrate > 0.2) & (C21_clickrate$clickrate <= 0.3)] <- 2
C21_clickrate$group[(C21_clickrate$clickrate > 0.3) & (C21_clickrate$clickrate <= 0.4)] <- 3
C21_clickrate$group[(C21_clickrate$clickrate > 0.4) & (C21_clickrate$clickrate <= 0.5)] <- 4
C21_clickrate$group[(C21_clickrate$clickrate > 0.5) & (C21_clickrate$clickrate <= 0.6)] <- 5
C21_clickrate$group[(C21_clickrate$clickrate > 0.6) & (C21_clickrate$clickrate <= 0.7)] <- 6
C21_clickrate$group[(C21_clickrate$clickrate > 0.7) & (C21_clickrate$clickrate <= 0.8)] <- 7
C21_clickrate$group[(C21_clickrate$clickrate > 0.8) & (C21_clickrate$clickrate <= 0.9)] <- 8
C21_clickrate$group[C21_clickrate$clickrate > 0.9] <- 9

C21_clickgroup <- C21_clickrate %>% select(C21, group)

train_clean <- left_join(train_clean, C21_clickgroup, by = "C21")
train_clean <- train_clean %>% select(-C21) %>% rename(C21_group = group)
train_clean$C21_group[is.na(train_clean$C21_group)] <- "other"

View(head(train_clean))



## do the same thing for site category
site_category_dist <- origin_train %>% count(site_category, sort = TRUE)
summary(site_category_dist)
### base on the statistics value, use 100 as threshold since the first quartile is 302
site_category_highfreq <- site_category_dist %>% filter(n > 100)
site_category_clickrate <- origin_train %>% filter(site_category %in% (site_category_highfreq$site_category)) %>% group_by(site_category) %>% summarize(clickrate = mean(click))
site_category_clickrate$group <- 0
site_category_clickrate$group[(site_category_clickrate$clickrate > 0.1) & (site_category_clickrate$clickrate <= 0.2)] <- 1
site_category_clickrate$group[(site_category_clickrate$clickrate > 0.2) & (site_category_clickrate$clickrate <= 0.3)] <- 2
site_category_clickrate$group[(site_category_clickrate$clickrate > 0.3) & (site_category_clickrate$clickrate <= 0.4)] <- 3
site_category_clickrate$group[(site_category_clickrate$clickrate > 0.4) & (site_category_clickrate$clickrate <= 0.5)] <- 4
site_category_clickrate$group[(site_category_clickrate$clickrate > 0.5) & (site_category_clickrate$clickrate <= 0.6)] <- 5
site_category_clickrate$group[(site_category_clickrate$clickrate > 0.6) & (site_category_clickrate$clickrate <= 0.7)] <- 6
site_category_clickrate$group[(site_category_clickrate$clickrate > 0.7) & (site_category_clickrate$clickrate <= 0.8)] <- 7
site_category_clickrate$group[(site_category_clickrate$clickrate > 0.8) & (site_category_clickrate$clickrate <= 0.9)] <- 8
site_category_clickrate$group[site_category_clickrate$clickrate > 0.9] <- 9

site_category_clickgroup <- site_category_clickrate %>% select(site_category, group)

train_clean <- left_join(train_clean, site_category_clickgroup, by = "site_category")
train_clean <- train_clean %>% select(-site_category) %>% rename(site_category_group = group)
train_clean$site_category_group[is.na(train_clean$site_category_group)] <- "other"

View(head(train_clean))



## do the same thing for app category
app_category_dist <- origin_train %>% count(app_category, sort = TRUE)
summary(app_category_dist)
### base on the statistics value, use 100 as threshold since the first quartile is 11 while the median is 918
app_category_highfreq <- app_category_dist %>% filter(n > 100)
app_category_clickrate <- origin_train %>% filter(app_category %in% (app_category_highfreq$app_category)) %>% group_by(app_category) %>% summarize(clickrate = mean(click))
app_category_clickrate$group <- 0
app_category_clickrate$group[(app_category_clickrate$clickrate > 0.1) & (app_category_clickrate$clickrate <= 0.2)] <- 1
app_category_clickrate$group[(app_category_clickrate$clickrate > 0.2) & (app_category_clickrate$clickrate <= 0.3)] <- 2
app_category_clickrate$group[(app_category_clickrate$clickrate > 0.3) & (app_category_clickrate$clickrate <= 0.4)] <- 3
app_category_clickrate$group[(app_category_clickrate$clickrate > 0.4) & (app_category_clickrate$clickrate <= 0.5)] <- 4
app_category_clickrate$group[(app_category_clickrate$clickrate > 0.5) & (app_category_clickrate$clickrate <= 0.6)] <- 5
app_category_clickrate$group[(app_category_clickrate$clickrate > 0.6) & (app_category_clickrate$clickrate <= 0.7)] <- 6
app_category_clickrate$group[(app_category_clickrate$clickrate > 0.7) & (app_category_clickrate$clickrate <= 0.8)] <- 7
app_category_clickrate$group[(app_category_clickrate$clickrate > 0.8) & (app_category_clickrate$clickrate <= 0.9)] <- 8
app_category_clickrate$group[app_category_clickrate$clickrate > 0.9] <- 9

app_category_clickgroup <- app_category_clickrate %>% select(app_category, group)

train_clean <- left_join(train_clean, app_category_clickgroup, by = "app_category")
train_clean <- train_clean %>% select(-app_category) %>% rename(app_category_group = group)
train_clean$app_category_group[is.na(train_clean$app_category_group)] <- "other"

View(head(train_clean))




### do the same thing for test data, also label new categories as "other"
### merge a data set data set contains site_id and new groups for high-frequency categories with original test data
test_cleansiteid <- left_join(origin_test, site_id_clickgroup, by = "site_id")
### delete original site_id, set low-frequency categories and new categories as "other"
test_cleansiteid <- test_cleansiteid %>% select(-site_id) %>% rename(site_id_group = group)
test_cleansiteid$site_id_group[is.na(test_cleansiteid$site_id_group)] <- "other"

test_cleansite <- left_join(test_cleansiteid, site_domain_clickgroup, by = "site_domain")
test_cleansite <- test_cleansite %>% select(-site_domain) %>% rename(site_domain_group = group)
test_cleansite$site_domain_group[is.na(test_cleansite$site_domain_group)] <- "other"

test_cleanappid <- left_join(test_cleansite, app_id_clickgroup, by = "app_id")
test_cleanappid <- test_cleanappid %>% select(-app_id) %>% rename(app_id_group = group)
test_cleanappid$app_id_group[is.na(test_cleanappid$app_id_group)] <- "other"

test_cleanappid <- left_join(test_cleanappid, app_domain_clickgroup, by = "app_domain")
test_cleanappid <- test_cleanappid %>% select(-app_domain) %>% rename(app_domain_group = group)
test_cleanappid$app_domain_group[is.na(test_cleanappid$app_domain_group)] <- "other"

test_cleanapp <- left_join(test_cleanappid, device_id_clickgroup, by = "device_id")
test_cleanapp <- test_cleanapp %>% select(-device_id) %>% rename(device_id_group = group)
test_cleanapp$device_id_group[is.na(test_cleanapp$device_id_group)] <- "other"

test_clean <- left_join(test_cleanapp, device_ip_clickgroup, by = "device_ip")
test_clean <- test_clean %>% select(-device_ip) %>% rename(device_ip_group = group)
test_clean$device_ip_group[is.na(test_clean$device_ip_group)] <- "other"

test_clean <- left_join(test_clean, device_model_clickgroup, by = "device_model")
test_clean <- test_clean %>% select(-device_model) %>% rename(device_model_group = group)
test_clean$device_model_group[is.na(test_clean$device_model_group)] <- "other"

test_clean <- left_join(test_clean, C14_clickgroup, by = "C14")
test_clean <- test_clean %>% select(-C14) %>% rename(C14_group = group)
test_clean$C14_group[is.na(test_clean$C14_group)] <- "other"

test_clean <- left_join(test_clean, C17_clickgroup, by = "C17")
test_clean <- test_clean %>% select(-C17) %>% rename(C17_group = group)
test_clean$C17_group[is.na(test_clean$C17_group)] <- "other"

test_clean <- left_join(test_clean, C19_clickgroup, by = "C19")
test_clean <- test_clean %>% select(-C19) %>% rename(C19_group = group)
test_clean$C19_group[is.na(test_clean$C19_group)] <- "other"

test_clean <- left_join(test_clean, C20_clickgroup, by = "C20")
test_clean$group[test_clean$C20 == -1] <- "other"
test_clean <- test_clean %>% select(-C20) %>% rename(C20_group = group)
test_clean$C20_group[is.na(test_clean$C20_group)] <- "other"

test_clean <- left_join(test_clean, C21_clickgroup, by = "C21")
test_clean <- test_clean %>% select(-C21) %>% rename(C21_group = group)
test_clean$C21_group[is.na(test_clean$C21_group)] <- "other"

test_clean <- left_join(test_clean, site_category_clickgroup, by = "site_category")
test_clean <- test_clean %>% select(-site_category) %>% rename(site_category_group = group)
test_clean$site_category_group[is.na(test_clean$site_category_group)] <- "other"

test_clean <- left_join(test_clean, app_category_clickgroup, by = "app_category")
test_clean <- test_clean %>% select(-app_category) %>% rename(app_category_group = group)
test_clean$app_category_group[is.na(test_clean$app_category_group)] <- "other"


# Part 2: Data sampling and splitting ------------------------------------------
## random sample 1,000,000 observations 5 times using different random seeds 
set.seed(11)
trainval_1 <- train_clean[sample(nrow(train_clean), 1000000), ]
##do a 60-40 split, create training and validation sets
TrainInd <- ceiling(nrow(trainval_1)*0.6)
random <- sample(nrow(trainval_1))
trainval_1 <- trainval_1[random,]
trainval_1[] <- lapply(trainval_1, factor) # factorize each variable
train_1 <- trainval_1[1:TrainInd,]
valid_1 <- trainval_1[(TrainInd+1):nrow(trainval_1),]


set.seed(12)
trainval_2 <- train_clean[sample(nrow(train_clean), 1000000), ]

random <- sample(nrow(trainval_2))
trainval_2 <- trainval_2[random,]
trainval_2[] <- lapply(trainval_2, factor)
train_2 <- trainval_2[1:TrainInd,]
valid_2 <- trainval_2[(TrainInd+1):nrow(trainval_2),]


set.seed(13)
trainval_3 <- train_clean[sample(nrow(train_clean), 1000000), ]

random <- sample(nrow(trainval_3))
trainval_3 <- trainval_3[random,]
trainval_3[] <- lapply(trainval_3, factor)
train_3 <- trainval_3[1:TrainInd,]
valid_3 <- trainval_3[(TrainInd+1):nrow(trainval_3),]


set.seed(14)
trainval_4 <- train_clean[sample(nrow(train_clean), 1000000), ]

random <- sample(nrow(trainval_4))
trainval_4 <- trainval_4[random,]
trainval_4[] <- lapply(trainval_4, factor)
train_4 <- trainval_4[1:TrainInd,]
valid_4 <- trainval_4[(TrainInd+1):nrow(trainval_4),]


set.seed(15)
trainval_5 <- train_clean[sample(nrow(train_clean), 1000000), ]

random <- sample(nrow(trainval_5))
trainval_5 <- trainval_5[random,]
trainval_5[] <- lapply(trainval_5, factor)
train_5 <- trainval_5[1:TrainInd,]
valid_5 <- trainval_5[(TrainInd+1):nrow(trainval_5),]

## ransom sanple another 2,000,000 observations for validating ensemble models
set.seed(26)
valid_6 <- train_clean[sample(nrow(train_clean), 2000000), ] # for ensemble method




# Part 3: Modeling - Decision Tree --------------------------------------
## create formula
Vars <- names(train_1)
BigFm <- paste(Vars[1],"~",paste(Vars[2:24],collapse=" + "),sep=" ")
BigFm <- as.formula(BigFm)

## using tree package gave us error, use rpart instead
## first sample
### create weights based on the frequency of class 0 and 1
positiveWeight = nrow(train_1) / (nrow(subset(train_1, click == 1)))
negativeWeight = nrow(train_1) / (nrow(subset(train_1, click == 0)))
modelWeights <- ifelse(train_1$click== 1, positiveWeight, negativeWeight)
### build a tree model
tc <- rpart.control(maxdepth = 10, xval = 5, cp = 0)
out <- rpart(BigFm, data=train_1, method = "class", control = tc, weights = modelWeights)
### find the best complexity parameter
bestcp <- out$cptable[which.min(out$cptable[,"xerror"]),"CP"]
### prune the tree based on it
model_tree_1 <- prune(out,cp=bestcp)
### predict on validation set and calculate the Log Loss
ypred <- predict(model_tree_1,newdata=valid_1[,2:24],type="prob")
ypred <- ypred[,2]
logloss_tree_1 <- LogLoss(ypred, (as.integer(valid_1$click)-1)) # as.integer will cause factorized variable become 1 & 2, need to -1
bestcp
logloss_tree_1 # Log Loss equals to 0.6025587

## second sample
positiveWeight = nrow(train_2) / (nrow(subset(train_2, click == 1)))
negativeWeight = nrow(train_2) / (nrow(subset(train_2, click == 0)))
modelWeights <- ifelse(train_2$click== 1, positiveWeight, negativeWeight)
tc <- rpart.control(maxdepth = 10, xval = 5, cp = 0)
out <- rpart(BigFm, data=train_2, method = "class", control = tc, weights = modelWeights)
bestcp <- out$cptable[which.min(out$cptable[,"xerror"]),"CP"]
model_tree_2 <- prune(out,cp=bestcp)
ypred <- predict(model_tree_2,newdata=valid_2[,2:24],type="prob")
ypred <- ypred[,2]
logloss_tree_2 <- LogLoss(ypred, (as.integer(valid_2$click)-1)) 
bestcp
logloss_tree_2 # Log Loss equals to 0.6037556


## third sample
positiveWeight = nrow(train_3) / (nrow(subset(train_3, click == 1)))
negativeWeight = nrow(train_3) / (nrow(subset(train_3, click == 0)))
modelWeights <- ifelse(train_3$click== 1, positiveWeight, negativeWeight)
tc <- rpart.control(maxdepth = 10, xval = 5, cp = 0)
out <- rpart(BigFm, data=train_3, method = "class", control = tc, weights = modelWeights)
bestcp <- out$cptable[which.min(out$cptable[,"xerror"]),"CP"]
model_tree_3 <- prune(out,cp=bestcp)
ypred <- predict(model_tree_3,newdata=valid_3[,2:24],type="prob")
ypred <- ypred[,2]
logloss_tree_3 <- LogLoss(ypred, (as.integer(valid_3$click)-1)) 
bestcp
logloss_tree_3 # Log Loss equals to 0.6036102


## fourth sample
positiveWeight = nrow(train_4) / (nrow(subset(train_4, click == 1)))
negativeWeight = nrow(train_4) / (nrow(subset(train_4, click == 0)))
modelWeights <- ifelse(train_4$click== 1, positiveWeight, negativeWeight)
tc <- rpart.control(maxdepth = 10, xval = 5, cp = 0)
out <- rpart(BigFm, data=train_4, method = "class", control = tc, weights = modelWeights)
bestcp <- out$cptable[which.min(out$cptable[,"xerror"]),"CP"]
model_tree_4 <- prune(out,cp=bestcp)
ypred <- predict(model_tree_4,newdata=valid_4[,2:24],type="prob")
ypred <- ypred[,2]
logloss_tree_4 <- LogLoss(ypred, (as.integer(valid_4$click)-1)) 
bestcp
logloss_tree_4 # Log Loss equals to 0.6026439


## fifth sample
positiveWeight = nrow(train_5) / (nrow(subset(train_5, click == 1)))
negativeWeight = nrow(train_5) / (nrow(subset(train_5, click == 0)))
modelWeights <- ifelse(train_5$click== 1, positiveWeight, negativeWeight)
tc <- rpart.control(maxdepth = 10, xval = 5, cp = 0)
out <- rpart(BigFm, data=train_5, method = "class", control = tc, weights = modelWeights)
bestcp <- out$cptable[which.min(out$cptable[,"xerror"]),"CP"]
model_tree_5 <- prune(out,cp=bestcp)
ypred <- predict(model_tree_5,newdata=valid_5[,2:24],type="prob")
ypred <- ypred[,2]
logloss_tree_5 <- LogLoss(ypred, (as.integer(valid_5$click)-1)) 
bestcp
logloss_tree_5 # Log Loss equals to 0.6042286


## ensemble models
### combine 5 validation data together for ensemble models
valid_all <- rbindlist(list(valid_1, valid_2, valid_3, valid_4, valid_5))

### a. use mean
#### use 5 tree models to predict valid_all, get 5 columns of probability
tree_pred_ensemble <- data.frame(c(1:nrow(valid_all)))
for (i in 1:5) {
  model <- get(paste0("model_tree_", i))
  ypred <- predict(model, newdata = valid_all[,2:24], type="prob")
  agent <- ypred[,2]
  tree_pred_ensemble <- cbind.data.frame(tree_pred_ensemble, agent)
}
colnames(tree_pred_ensemble) <- c("num", "m1", "m2", "m3", "m4", "m5")
#### calculate the mean probability
tree_pred_ensemble <- tree_pred_ensemble %>% mutate(avgprob = (m1 + m2 + m3 + m4 + m5) / 5)
#### calculate the Log Loss
logloss_tree_mean <- LogLoss(tree_pred_ensemble$avgprob, (as.integer(valid_all$click)-1))
logloss_tree_mean # Log Loss equals to 0.6014383

### b. use median
#### calculate the median of probabilities
tree_pred_ensemble$medianprob <- apply(tree_pred_ensemble[,2:6], 1, median) 
#### calculate the Log Loss
logloss_tree_median <- LogLoss(tree_pred_ensemble$medianprob, (as.integer(valid_all$click)-1)) 
logloss_tree_median # Log Loss equals to 0.6026196

### c. use logistic regression
#### use valid_all to train the model
model_tree_ensemble_logreg <- glm(valid_all$click ~ m1 + m2 + m3 + m4 + m5, data = tree_pred_ensemble, family = binomial(link='logit'))
#### use valid_6 to validate the model, calculate the predicted probability from 5 tree models
tree_pred_ensemble_valid <- data.frame(c(1:nrow(valid_6)))
for (i in 1:5) {
     model <- get(paste0("model_tree_", i))
     ypred <- predict(model, newdata = valid_6[,2:24], type="prob")
     agent <- ypred[,2]
     tree_pred_ensemble_valid <- cbind.data.frame(tree_pred_ensemble_valid, agent)
}
colnames(tree_pred_ensemble_valid) <- c("num", "m1", "m2", "m3", "m4", "m5")
#### make prediction
ypred <- predict(model_tree_ensemble_logreg, newdata = tree_pred_ensemble_valid[, 2:6], type="response")
#### calculate the Log Loss
logloss_tree_logreg <- LogLoss(ypred, (as.integer(valid_6$click)-1)) 
logloss_tree_logreg # Log Loss equals to 0.4033911

# the logistic regression ensemble model is the best model for decision tree


# Part 4: Modeling - Random Forest --------------------------------------
## first model
set.seed(11)
### build a for loop to find the best "mtry"
logloss_rf_1 <- c()
for (k in c(5:15, 25)) {
  out3 <- randomForest(BigFm,data=train_1, mtry=k, ntree=100)
  ypred <- predict(out3,newdata=valid_1[,2:24],type="prob")
  ypred <- ypred[,2]
  logloss_rf_1[length(logloss_rf_1)+1] <- LogLoss(ypred, (as.integer(valid_1$click)-1))
}
logloss_rf_1 
which.min(logloss_rf_1) # 3.37425 as the lowest Log Loss, mtry = 11
bestk <- which.min(logloss_rf_1) + 4
### fit the model with the best "mtry"
set.seed(11)
model_rf_1 <- randomForest(BigFm,data=train_1, mtry=bestk,ntree=100)


## second model
set.seed(12)
logloss_rf_2 <- c()
### to save time, only loop mtry from 10 to 15 based on the outcome of model 1
for (k in c(10:15)) {
  out3 <- randomForest(BigFm,data=train_2, mtry=k,ntree=100)
  ypred <- predict(out3,newdata=valid_2[,2:24],type="prob")
  ypred <- ypred[,2]
  logloss_rf_2[length(logloss_rf_2)+1] <- LogLoss(ypred, (as.integer(valid_2$click)-1))
}
logloss_rf_2
which.min(logloss_rf_2) # 3.33316 as the lowest Log Loss, mtry = 14
bestk <- which.min(logloss_rf_2) + 9
set.seed(12)
model_rf_2 <- randomForest(BigFm,data=train_2, mtry=bestk,ntree=100)


## third model
set.seed(13)
logloss_rf_3 <- c()
for (k in c(10:15)) {
  out3 <- randomForest(BigFm,data=train_3, mtry=k,ntree=100)
  ypred <- predict(out3,newdata=valid_3[,2:24],type="prob")
  ypred <- ypred[,2]
  logloss_rf_3[length(logloss_rf_3)+1] <- LogLoss(ypred, (as.integer(valid_3$click)-1))
}
logloss_rf_3
which.min(logloss_rf_3) # 3.245467 as the lowest Log Loss, mtry = 14
bestk <- which.min(logloss_rf_3) + 9
set.seed(13)
model_rf_3 <- randomForest(BigFm,data=train_3, mtry=bestk,ntree=100)


## fourth model
set.seed(14)
logloss_rf_4 <- c()
for (k in c(10:15)) {
  out3 <- randomForest(BigFm,data=train_4, mtry=k,ntree=100)
  ypred <- predict(out3,newdata=valid_4[,2:24],type="prob")
  ypred <- ypred[,2]
  logloss_rf_4[length(logloss_rf_4)+1] <- LogLoss(ypred, (as.integer(valid_4$click)-1))
}
logloss_rf_4
which.min(logloss_rf_4) # 3.315264 as the lowest Log Loss, mtry = 14
bestk <- which.min(logloss_rf_4) + 9
set.seed(14)
model_rf_4 <- randomForest(BigFm,data=train_4, mtry=bestk,ntree=100)


## fifth model
set.seed(15)
logloss_rf_5 <- c()
for (k in c(10:15)) {
  out3 <- randomForest(BigFm,data=train_5, mtry=k,ntree=100)
  ypred <- predict(out3,newdata=valid_5[,2:24],type="prob")
  ypred <- ypred[,2]
  logloss_rf_5[length(logloss_rf_5)+1] <- LogLoss(ypred, (as.integer(valid_5$click)-1))
}
logloss_rf_5
which.min(logloss_rf_5)  # 3.382308 as the lowest Log Loss, mtry = 14
bestk <- which.min(logloss_rf_5) + 9
set.seed(15)
model_rf_5 <- randomForest(BigFm,data=train_5, mtry=bestk,ntree=100)



## ensemble models (similar to what we did for the tree ensemble models)
### a. use mean
rf_pred_ensemble <- data.frame(c(1:nrow(valid_all)))
for (i in 1:5) {
  model <- get(paste0("model_rf_", i))
  ypred <- predict(model, newdata = valid_all[,2:24], type="prob")
  agent <- ypred[,2]
  rf_pred_ensemble <- cbind.data.frame(rf_pred_ensemble, agent)
}
colnames(rf_pred_ensemble) <- c("num", "m1", "m2", "m3", "m4", "m5")
rf_pred_ensemble <- rf_pred_ensemble %>% mutate(avgprob = (m1 + m2 + m3 + m4 + m5) / 5)
logloss_rf_mean <- LogLoss(rf_pred_ensemble$avgprob, (as.integer(valid_all$click)-1))
logloss_rf_mean # Log Loss equals to 2.538491

#### b. use median
rf_pred_ensemble$medianprob <- apply(rf_pred_ensemble[,2:6], 1, median) 
logloss_rf_median <- LogLoss(rf_pred_ensemble$medianprob, (as.integer(valid_all$click)-1)) 
logloss_rf_median # Log Loss equals to 3.279562

#### c. use logistic regression
model_rf_ensemble_logreg <- glm(valid_all$click ~ m1 + m2 + m3 + m4 + m5, data = rf_pred_ensemble, family = binomial(link='logit'))

rf_pred_ensemble_valid <- data.frame(c(1:nrow(valid_6)))
for (i in 1:5) {
  model <- get(paste0("model_rf_", i))
  ypred <- predict(model, newdata = valid_6[,2:24], type="prob")
  agent <- ypred[,2]
  rf_pred_ensemble_valid <- cbind.data.frame(rf_pred_ensemble_valid, agent)
}
colnames(rf_pred_ensemble_valid) <- c("num", "m1", "m2", "m3", "m4", "m5")
ypred <- predict(model_rf_ensemble_logreg, newdata = rf_pred_ensemble_valid[, 2:6], type="response")
logloss_rf_logreg <- LogLoss(ypred, (as.integer(valid_6$click)-1)) 
logloss_rf_logreg # Log Loss equals to 0.433572

#  the logistic regression ensemble model is the best model for random forest


# Part 5: Modeling - Naive Bayes ----------------------------------------------------------
## use tree model to determine important features for naive bayes, first tree model is the best tree model, use it to calculate the importance scores of variables
import_tree_1 <- as.data.frame(varImp(model_tree_1))
## sort the table based on importance score
import_tree_1$variable <- rownames(import_tree_1)
sort_import_tree_1 <- import_tree_1[order(-import_tree_1$Overall),]
sort_import_tree_1

## first model
### build a for loop to feed different numbers of features into the model
logloss_nb_1 <- c()
for (i in 1:10) {
  variable <- sort_import_tree_1$variable[1:i]
  train_1_nb <- train_1 %>% select(click, variable)
  valid_1_nb <- valid_1 %>% select(click, variable)
  model <- naiveBayes(click ~., data=train_1_nb)
  ypred <- predict(model, newdata = valid_1_nb, type="raw")
  ypred <- ypred[,2]
  logloss_nb_1[length(logloss_nb_1)+1] <- LogLoss(ypred, (as.integer(valid_1$click)-1))
}
logloss_nb_1 # 0.4245828 as the lowest Log Loss when i = 1, the more the variables the bigger the Log Loss
model_nb_1 <- naiveBayes(click ~ C14_group, data=train_1_nb)


## second model
logloss_nb_2 <- c()
### to save time, only loop 3 times based on the outcome of model 1
for (i in 1:3) {
  variable <- sort_import_tree_1$variable[1:i]
  train_2_nb <- train_2 %>% select(click, variable)
  valid_2_nb <- valid_2 %>% select(click, variable)
  model <- naiveBayes(click ~., data=train_2_nb)
  ypred <- predict(model, newdata = valid_2_nb, type="raw")
  ypred <- ypred[,2]
  logloss_nb_2[length(logloss_nb_2)+1] <- LogLoss(ypred, (as.integer(valid_2$click)-1))
}
logloss_nb_2 # 0.424109 as the lowest Log Loss when i = 1
model_nb_2 <- naiveBayes(click ~ C14_group, data=train_2_nb)


## third model
logloss_nb_3 <- c()
for (i in 1:3) {
  variable <- sort_import_tree_1$variable[1:i]
  train_3_nb <- train_3 %>% select(click, variable)
  valid_3_nb <- valid_3 %>% select(click, variable)
  model <- naiveBayes(click ~., data=train_3_nb)
  ypred <- predict(model, newdata = valid_3_nb, type="raw")
  ypred <- ypred[,2]
  logloss_nb_3[length(logloss_nb_3)+1] <- LogLoss(ypred, (as.integer(valid_3$click)-1))
}
logloss_nb_3  # 0.4238987 as the lowest Log Loss when i = 1
model_nb_3 <- naiveBayes(click ~ C14_group, data=train_3_nb)


### fourth model
logloss_nb_4 <- c()
for (i in 1:3) {
  variable <- sort_import_tree_1$variable[1:i]
  train_4_nb <- train_4 %>% select(click, variable)
  valid_4_nb <- valid_4 %>% select(click, variable)
  model <- naiveBayes(click ~., data=train_4_nb)
  ypred <- predict(model, newdata = valid_4_nb, type="raw")
  ypred <- ypred[,2]
  logloss_nb_4[length(logloss_nb_4)+1] <- LogLoss(ypred, (as.integer(valid_4$click)-1))
}
logloss_nb_4 # 0.4226488 as the lowest Log Loss when i = 1
model_nb_4 <- naiveBayes(click ~ C14_group, data=train_4_nb) 


### fifth model
logloss_nb_5 <- c()
for (i in 1:3) {
  variable <- sort_import_tree_1$variable[1:i]
  train_5_nb <- train_5 %>% select(click, variable)
  valid_5_nb <- valid_5 %>% select(click, variable)
  model <- naiveBayes(click ~., data=train_5_nb)
  ypred <- predict(model, newdata = valid_5_nb, type="raw")
  ypred <- ypred[,2]
  logloss_nb_5[length(logloss_nb_5)+1] <- LogLoss(ypred, (as.integer(valid_5$click)-1))
}
logloss_nb_5 # 0.4248007 as the lowest Log Loss when i = 1
model_nb_5 <- naiveBayes(click ~ C14_group, data=train_5_nb) 


## ensemble models (similar to what we did for the tree ensemble models)
### a. use mean
nb_pred_ensemble <- data.frame(c(1:nrow(valid_all)))
for (i in 1:5) {
  model <- get(paste0("model_nb_", i))
  ypred <- predict(model, newdata = valid_all[,2:24], type="raw")
  agent <- ypred[,2]
  nb_pred_ensemble <- cbind.data.frame(nb_pred_ensemble, agent)
}
colnames(nb_pred_ensemble) <- c("num", "m1", "m2", "m3", "m4", "m5")
nb_pred_ensemble <- nb_pred_ensemble %>% mutate(avgraw = (m1 + m2 + m3 + m4 + m5) / 5)
logloss_nb_mean <- LogLoss(nb_pred_ensemble$avgraw, (as.integer(valid_all$click)-1))
logloss_nb_mean # Log Loss equals to 0.4240002 

### b. use median
nb_pred_ensemble$medianraw <- apply(nb_pred_ensemble[,2:6], 1, median) 
logloss_nb_median <- LogLoss(nb_pred_ensemble$medianraw, (as.integer(valid_all$click)-1)) 
logloss_nb_median # Log Loss equals to 0.4240014

### c. use logistic regression
model_nb_ensemble_logreg <- glm(valid_all$click ~ m1 + m2 + m3 + m4 + m5, data = nb_pred_ensemble, family = binomial(link='logit'))

nb_pred_ensemble_valid <- data.frame(c(1:nrow(valid_6)))
for (i in 1:5) {
  model <- get(paste0("model_nb_", i))
  ypred <- predict(model, newdata = valid_6[,2:24], type="raw")
  agent <- ypred[,2]
  nb_pred_ensemble_valid <- cbind.data.frame(nb_pred_ensemble_valid, agent)
}
colnames(nb_pred_ensemble_valid) <- c("num", "m1", "m2", "m3", "m4", "m5")
ypred <- predict(model_nb_ensemble_logreg, newdata = nb_pred_ensemble_valid[, 2:6], type="response")
logloss_nb_logreg <- LogLoss(ypred, (as.integer(valid_6$click)-1)) 
logloss_nb_logreg # Log Loss equals to 0.42483



# the fourth model is the best model for naive bayes



# Part 6: Modeling - Logistic regression ---------------------------------------------

grid <- 10^seq(5,-3,length=200) #set a grid of lambdas to search the best lambda

## 1. Lasso regression
### define a lasso regression function - input: train data, valid data, and grid; output: best model on valid data with best lambda, logloss and predictions from the best model
logit_lasso <- function(train_i, valid_i, grid){
  XTrain <- model.matrix(click ~ . ,train_i)[,-1]
  XVal <- model.matrix(click ~ . ,valid_i)[,-1]
  YTrain <- train_i$click
  YVal <- valid_i$click
  out <- glmnet(XTrain,YTrain,alpha=1,lambda=grid,thresh=1e-12,family="binomial") # using glmnet to run logistic regression with L1 regularization(lasso); glmnet does the dummy transformation itself
  YHat <- predict(out,newx=XVal,type = "response")
  logloss<-apply(YHat,2,FUN=LogLoss,(as.integer(YVal)-1))
  wh <- which.min(logloss)# find the best lambda with the lowest logloss
  best_lambda <- out$lambda[wh]
  best_out <- glmnet(XTrain,YTrain,alpha=1,lambda=best_lambda,thresh=1e-12,family="binomial")
  best_YHat <- predict(best_out,newx=XVal,type = "response")
  best_logloss <-apply(best_YHat,2,FUN=LogLoss,(as.integer(YVal)-1))
  list_output <- list("best_lambda" = best_lambda , "best_out" = best_out, "best_logloss" = best_logloss, "best_yhat" = best_YHat)
  return(list_output)
}

### run the function on 5 samples
logit_lasso_1 <- logit_lasso(train_1, valid_1, grid)
logit_lasso_2 <- logit_lasso(train_2, valid_2, grid)
logit_lasso_3 <- logit_lasso(train_3, valid_3, grid)
logit_lasso_4 <- logit_lasso(train_4, valid_4, grid) #best one, with Log Loss of 0.4009114
logit_lasso_5 <- logit_lasso(train_5, valid_5, grid)


## 2. Ridge regression
### define a ridge regression function - input: train data, valid data, and grid; output: best model on valid data with best lambda, logloss and predictions from the best model
logit_ridge <- function(train_i, valid_i, grid){
  XTrain <- model.matrix(click ~ . ,train_i)[,-1]
  XVal <- model.matrix(click ~ . ,valid_i)[,-1]
  YTrain <- train_i$click
  YVal <- valid_i$click
  out <- glmnet(XTrain,YTrain,alpha=0,lambda=grid,thresh=1e-12,family="binomial") # using glmnet to run logistic regression with L2 regularization(ridge); glmnet does the dummy transformation itself
  YHat <- predict(out,newx=XVal,type = "response")
  logloss <-apply(YHat,2,FUN=LogLoss,(as.integer(YVal)-1))
  wh <- which.min(logloss)
  best_lambda <- out$lambda[wh]
  best_out <- glmnet(XTrain,YTrain,alpha=0,lambda=best_lambda,thresh=1e-12,family="binomial")
  best_YHat <- predict(best_out,newx=XVal,type = "response")
  best_logloss <-apply(best_YHat,2,FUN=LogLoss,(as.integer(YVal)-1))
  list_output <- list("best_lambda" = best_lambda , "best_out" = best_out, "best_logloss" = best_logloss, "best_yhat" = best_YHat)
  return(list_output)
}

### run the function on 5 samples
logit_ridge_1 <- logit_ridge(train_1, valid_1, grid)
logit_ridge_2 <- logit_ridge(train_2, valid_2, grid)
logit_ridge_3 <- logit_ridge(train_3, valid_3, grid)
logit_ridge_4 <- logit_ridge(train_4, valid_4, grid) #best one, with Log Loss of 0.4001927
logit_ridge_5 <- logit_ridge(train_5, valid_5, grid)


## 3. Ridge regression with only importance features determined by Lasso models
### use the "varimp" function from "caret" to compute the importance of variables
import_feature <- function(model, lambda){
  import_glm <- as.data.frame(varImp(model, lambda = lambda))
  import_glm$variable <- rownames(import_glm)
  return(import_glm[order(-import_glm$Overall),])
}

### compute the importance of the five lasso regression models
feature_1 <- import_feature(logit_lasso_1$best_out, logit_lasso_1$best_lambda)
feature_2 <- import_feature(logit_lasso_2$best_out, logit_lasso_2$best_lambda)
feature_3 <- import_feature(logit_lasso_3$best_out, logit_lasso_3$best_lambda)
feature_4 <- import_feature(logit_lasso_4$best_out, logit_lasso_4$best_lambda)
feature_5 <- import_feature(logit_lasso_5$best_out, logit_lasso_5$best_lambda)

### define important features as the features that explain more than 0 variance of the data and the feature space decreases from around 130 to around 75
import_feature_1 <- feature_1[feature_1$Overall > 0,]
import_feature_2 <- feature_2[feature_2$Overall > 0,]
import_feature_3 <- feature_3[feature_3$Overall > 0,]
import_feature_4 <- feature_4[feature_4$Overall > 0,]
import_feature_5 <- feature_5[feature_5$Overall > 0,]

### create a function that transform variables into dummies and the names of the variables are consistent with the ones given by glmnet 
dummy_transformation <- function(data_i){
  data_i_dum <- dummy_cols(data_i[,-1], select_columns = NULL, remove_first_dummy = TRUE, remove_most_frequent_dummy = FALSE)[,-c(1:23)] #doesnt have device_type 2 and app_category_group_other so 137 variables in total
  new_colnames_i <- gsub("\\_(?=[^_]*$)", "", colnames(data_i_dum), perl = TRUE)
  colnames(data_i_dum) <- new_colnames_i
  data_i_dum <- cbind(data_i[,1], data_i_dum)
  return(data_i_dum) 
}

### transform the 5 samples into dummies
train_1_dum <- dummy_transformation(train_1)
train_2_dum <- dummy_transformation(train_2)
train_3_dum <- dummy_transformation(train_3)
train_4_dum <- dummy_transformation(train_4)
train_5_dum <- dummy_transformation(train_5)

valid_1_dum <- dummy_transformation(valid_1)
valid_2_dum <- dummy_transformation(valid_2)
valid_3_dum <- dummy_transformation(valid_3)
valid_4_dum <- dummy_transformation(valid_4)
valid_5_dum <- dummy_transformation(valid_5)
valid_6_dum <- dummy_transformation(valid_6)

### trim down the feature space of the 5 samples using only important features
train_1_trim <- train_1_dum %>% select_if(colnames(train_1_dum) %in% c('click', rownames(import_feature_1)))
train_2_trim <- train_2_dum %>% select_if(colnames(train_2_dum) %in% c('click', rownames(import_feature_2)))
train_3_trim <- train_3_dum %>% select_if(colnames(train_3_dum) %in% c('click', rownames(import_feature_3)))
train_4_trim <- train_4_dum %>% select_if(colnames(train_4_dum) %in% c('click', rownames(import_feature_4)))
train_5_trim <- train_5_dum %>% select_if(colnames(train_5_dum) %in% c('click', rownames(import_feature_5)))

valid_1_trim <- valid_1_dum %>% select_if(colnames(valid_1_dum) %in% c('click', rownames(import_feature_1)))
valid_2_trim <- valid_2_dum %>% select_if(colnames(valid_2_dum) %in% c('click', rownames(import_feature_2)))
valid_3_trim <- valid_3_dum %>% select_if(colnames(valid_3_dum) %in% c('click', rownames(import_feature_3)))
valid_4_trim <- valid_4_dum %>% select_if(colnames(valid_4_dum) %in% c('click', rownames(import_feature_4)))
valid_5_trim <- valid_5_dum %>% select_if(colnames(valid_5_dum) %in% c('click', rownames(import_feature_5)))

### make sure the training data and the valid data have exactly the same variables
train_1_trim <- train_1_trim %>% select_if(colnames(train_1_trim) %in% colnames(valid_1_trim))
train_2_trim <- train_2_trim %>% select_if(colnames(train_2_trim) %in% colnames(valid_2_trim))
train_3_trim <- train_3_trim %>% select_if(colnames(train_3_trim) %in% colnames(valid_3_trim))
train_4_trim <- train_4_trim %>% select_if(colnames(train_4_trim) %in% colnames(valid_4_trim))
train_5_trim <- train_5_trim %>% select_if(colnames(train_5_trim) %in% colnames(valid_5_trim))

valid_1_trim <- valid_1_trim %>% select_if(colnames(valid_1_trim) %in% colnames(train_1_trim))
valid_2_trim <- valid_2_trim %>% select_if(colnames(valid_2_trim) %in% colnames(train_2_trim))
valid_3_trim <- valid_3_trim %>% select_if(colnames(valid_3_trim) %in% colnames(train_3_trim))
valid_4_trim <- valid_4_trim %>% select_if(colnames(valid_4_trim) %in% colnames(train_4_trim))
valid_5_trim <- valid_5_trim %>% select_if(colnames(valid_5_trim) %in% colnames(train_5_trim))

### rerun the ridge regression using the data that only has important features - the logloss increased
logit_ridge_trim_1 <- logit_ridge(train_1_trim, valid_1_trim, grid) # Log Loss = 0.4207586
logit_ridge_trim_2 <- logit_ridge(train_2_trim, valid_2_trim, grid) # Log Loss = 0.4128881 
logit_ridge_trim_3 <- logit_ridge(train_3_trim, valid_3_trim, grid) # Log Loss = 0.4499932 
logit_ridge_trim_4 <- logit_ridge(train_4_trim, valid_4_trim, grid) # Log Loss = 0.4257033
logit_ridge_trim_5 <- logit_ridge(train_5_trim, valid_5_trim, grid) # Log Loss = 0.4171194


## ensemble models
### a. logreg - lasso 5 - run a logistic regression on the 5 columns of predictions made by 5 lasso regressions
valid_all <- rbind.data.frame(valid_1, valid_2, valid_3, valid_4, valid_5)
XVal_all <- model.matrix(click ~ . ,valid_all)[,-1]
XVal_6 <- model.matrix(click ~ . ,valid_6)[,-1]

logit_lasso_1_yhat <- predict(logit_lasso_1$best_out, newx = XVal_all, type = "response")
logit_lasso_2_yhat <- predict(logit_lasso_2$best_out, newx = XVal_all, type = "response")
logit_lasso_3_yhat <- predict(logit_lasso_3$best_out, newx = XVal_all, type = "response")
logit_lasso_4_yhat <- predict(logit_lasso_4$best_out, newx = XVal_all, type = "response")
logit_lasso_5_yhat <- predict(logit_lasso_5$best_out, newx = XVal_all, type = "response")

logit_lasso_1_yhat_valid_6 <- predict(logit_lasso_1$best_out, newx = XVal_6, type = "response")
logit_lasso_2_yhat_valid_6 <- predict(logit_lasso_2$best_out, newx = XVal_6, type = "response")
logit_lasso_3_yhat_valid_6 <- predict(logit_lasso_3$best_out, newx = XVal_6, type = "response")
logit_lasso_4_yhat_valid_6 <- predict(logit_lasso_4$best_out, newx = XVal_6, type = "response")
logit_lasso_5_yhat_valid_6 <- predict(logit_lasso_5$best_out, newx = XVal_6, type = "response")

lasso_yhat_ensemble <- cbind.data.frame(valid_all$click, pred_1 = logit_lasso_1_yhat, pred_2 = logit_lasso_2_yhat, pred_3 = logit_lasso_3_yhat, pred_4 = logit_lasso_4_yhat, pred_5 = logit_lasso_5_yhat)
colnames(lasso_yhat_ensemble) <- c('click', "pred_1", "pred_2", "pred_3", "pred_4", "pred_5")
lasso_yhat_ensemble_valid_6 <- cbind.data.frame("click" = valid_6$click, "pred_1" = logit_lasso_1_yhat_valid_6, "pred_2" = logit_lasso_2_yhat_valid_6, "pred_3" = logit_lasso_3_yhat_valid_6, "pred_4" = logit_lasso_4_yhat_valid_6, "pred_5" = logit_lasso_5_yhat_valid_6)
colnames(lasso_yhat_ensemble_valid_6) <- c('click', "pred_1", "pred_2", "pred_3", "pred_4", "pred_5")

lasso_ensemble_logreg <- glm(click ~ pred_1 + pred_2 + pred_3 + pred_4 + pred_5, data = lasso_yhat_ensemble, family = binomial(link='logit'))
lasso_ensemble_logreg_yhat <- predict(lasso_ensemble_logreg, newdata = lasso_yhat_ensemble_valid_6, type="response")
lasso_ensemble_logreg_logloss <- LogLoss(lasso_ensemble_logreg_yhat, (as.integer(valid_6$click)-1)) 
lasso_ensemble_logreg_logloss #0.4074958

### b. logreg - ridge 5 - run a logistic regression on the 5 columns of predictions made by 5 ridge regressions(using all features)
logit_ridge_1_yhat <- predict(logit_ridge_1$best_out, newx = XVal_all, type = "response")
logit_ridge_2_yhat <- predict(logit_ridge_2$best_out, newx = XVal_all, type = "response")
logit_ridge_3_yhat <- predict(logit_ridge_3$best_out, newx = XVal_all, type = "response")
logit_ridge_4_yhat <- predict(logit_ridge_4$best_out, newx = XVal_all, type = "response")
logit_ridge_5_yhat <- predict(logit_ridge_5$best_out, newx = XVal_all, type = "response")

logit_ridge_1_yhat_valid_6 <- predict(logit_ridge_1$best_out, newx = XVal_6, type = "response")
logit_ridge_2_yhat_valid_6 <- predict(logit_ridge_2$best_out, newx = XVal_6, type = "response")
logit_ridge_3_yhat_valid_6 <- predict(logit_ridge_3$best_out, newx = XVal_6, type = "response")
logit_ridge_4_yhat_valid_6 <- predict(logit_ridge_4$best_out, newx = XVal_6, type = "response")
logit_ridge_5_yhat_valid_6 <- predict(logit_ridge_5$best_out, newx = XVal_6, type = "response")

ridge_yhat_ensemble <- cbind.data.frame(valid_all$click, logit_ridge_1_yhat, logit_ridge_2_yhat, logit_ridge_3_yhat, logit_ridge_4_yhat, logit_ridge_5_yhat)
colnames(ridge_yhat_ensemble) <- c('click', "pred_1", "pred_2", "pred_3", "pred_4", "pred_5")
ridge_yhat_ensemble_valid_6 <- cbind.data.frame(valid_6$click, logit_ridge_1_yhat_valid_6, logit_ridge_2_yhat_valid_6, logit_ridge_3_yhat_valid_6, logit_ridge_4_yhat_valid_6, logit_ridge_5_yhat_valid_6)
colnames(ridge_yhat_ensemble_valid_6) <- c('click', "pred_1", "pred_2", "pred_3", "pred_4", "pred_5")

ridge_ensemble_logreg <- glm(click ~ pred_1 + pred_2 + pred_3 + pred_4 + pred_5, data = ridge_yhat_ensemble, family = binomial(link='logit'))
ridge_ensemble_logreg_yhat <- predict(ridge_ensemble_logreg, newdata = ridge_yhat_ensemble_valid_6, type="response")
ridge_ensemble_logreg_logloss <- LogLoss(ridge_ensemble_logreg_yhat, (as.integer(valid_6$click)-1)) 
ridge_ensemble_logreg_logloss #0.4069258

### c. logreg - lasso 5 + ridge 5 - run a logistic regression on the 10 columns of predictions made by 5 ridge and 5 lasso regressions(using all features)
landr_yhat_ensemble <- cbind.data.frame(valid_all$click, lasso_yhat_ensemble[,-1], ridge_yhat_ensemble[,-1])
landr_yhat_ensemble_valid_6 <- cbind.data.frame(valid_6$click, lasso_yhat_ensemble_valid_6[,-1], ridge_yhat_ensemble_valid_6[,-1])
colnames(landr_yhat_ensemble) <- c('click','pred_1','pred_2','pred_3','pred_4','pred_5','pred_6','pred_7','pred_8','pred_9','pred_10')
colnames(landr_yhat_ensemble_valid_6) <- c('click','pred_1','pred_2','pred_3','pred_4','pred_5','pred_6','pred_7','pred_8','pred_9','pred_10')

landr_ensemble_logreg <- glm(click ~ pred_1 + pred_2 + pred_3 + pred_4 + pred_5 + pred_6 + pred_7 + pred_8 + pred_9 + pred_10, data = landr_yhat_ensemble, family = binomial(link='logit'))
landr_ensemble_logreg_yhat <- predict(landr_ensemble_logreg, newdata = landr_yhat_ensemble_valid_6, type="response")
landr_ensemble_logreg_logloss <- LogLoss(landr_ensemble_logreg_yhat, (as.integer(valid_6$click)-1)) 
landr_ensemble_logreg_logloss #0.4068811


## now try using same validation data to evaluate all of the models to better compare - the resuls are similar as the previous models using different validation data to validate
YVal_6 <- valid_6$click
same_lasso_1_yhat <- predict(logit_lasso_1$best_out,newx=XVal_6,type = "response")
same_lasso_1_logloss <-apply(same_lasso_1_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))
same_lasso_2_yhat <- predict(logit_lasso_2$best_out,newx=XVal_6,type = "response")
same_lasso_2_logloss <-apply(same_lasso_2_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))
same_lasso_3_yhat <- predict(logit_lasso_3$best_out,newx=XVal_6,type = "response")
same_lasso_3_logloss <-apply(same_lasso_3_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))
same_lasso_4_yhat <- predict(logit_lasso_4$best_out,newx=XVal_6,type = "response")
same_lasso_4_logloss <-apply(same_lasso_4_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))
same_lasso_5_yhat <- predict(logit_lasso_5$best_out,newx=XVal_6,type = "response")
same_lasso_5_logloss <-apply(same_lasso_5_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))

same_ridge_1_yhat <- predict(logit_ridge_1$best_out,newx=XVal_6,type = "response")
same_ridge_1_logloss <-apply(same_ridge_1_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))
same_ridge_2_yhat <- predict(logit_ridge_2$best_out,newx=XVal_6,type = "response")
same_ridge_2_logloss <-apply(same_ridge_2_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))
same_ridge_3_yhat <- predict(logit_ridge_3$best_out,newx=XVal_6,type = "response")
same_ridge_3_logloss <-apply(same_ridge_3_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))
same_ridge_4_yhat <- predict(logit_ridge_4$best_out,newx=XVal_6,type = "response")
same_ridge_4_logloss <-apply(same_ridge_4_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))
same_ridge_5_yhat <- predict(logit_ridge_5$best_out,newx=XVal_6,type = "response")
same_ridge_5_logloss <-apply(same_ridge_5_yhat,2,FUN=LogLoss,(as.integer(YVal_6)-1))


# the fourth ridge model with all features is the best model for logistic regression


# Part 7: Modeling - K-nearest neighbors -----------------------------------------------------
## select top 25 important variables
vars_wanted <- c("site_id_group6","app_id_group6","site_id_group5","site_id_group4","app_id_group8","site_id_group3","app_id_group3","C14_group5","app_id_group2", "app_id_group5","device_ip_group7","C14_group3","C14_group4","device_model_group2","C14_group2", "app_id_group1","C21_group2", "site_domain_group2","C19_group2","device_ip_group3","C17_group2","device_model_group3","app_id_group9","C17_group5","site_domain_group3")
train_1_knn <- train_1_dum %>% select(vars_wanted)
valid_1_knn <- valid_1_dum %>% select(vars_wanted)
train_2_knn <- train_2_dum %>% select(vars_wanted)
valid_2_knn <- valid_2_dum %>% select(vars_wanted)
train_3_knn <- train_3_dum %>% select(vars_wanted)
valid_3_knn <- valid_3_dum %>% select(vars_wanted)
train_4_knn <- train_4_dum %>% select(vars_wanted)
valid_4_knn <- valid_4_dum %>% select(vars_wanted)
train_5_knn <- train_5_dum %>% select(vars_wanted)
valid_5_knn <- valid_5_dum %>% select(vars_wanted)
valid_6_knn <- valid_6_dum %>% select(vars_wanted)


## first model
set.seed(11)
logloss_knn_1 <- c()
### build a for loop that loop over every odd value of k from 11 to 99
for (i in seq(11,99,2)) {
  out1 <- knn(train=train_1_knn,test=valid_1_knn,cl = factor(train_1$click),prob = TRUE,k=i)
  ypred1 <- matrix(attr(out1,'prob'))
  logloss_knn_1[length(logloss_knn_1)+1] <- LogLoss(ypred1, (as.integer(valid_1$click)-1))
}
### print out the log loss for all the differnt k's, the best k and the log loss associated with best k
logloss_knn_1
bestk1 <- which.min(logloss_knn_1)*2 + 5
bestk1
logloss_knn_1[(bestk1-5)/2] # 1.802202 as the lowest Log Loss
### build new model with best k for the ensemble models
model_knn_1 <-knn(train=train_1_knn,test=valid_1_knn,cl = factor(train_1$click),prob = TRUE,k=bestk)



## second model
set.seed(12)
logloss_knn_2 <- c()
for (i in seq(11,99,2)) {
  out2 <- knn(train=train_2_knn,test=valid_2_knn,cl = factor(train_2$click),prob = TRUE,k=i)
  ypred2 <- matrix(attr(out2,'prob'))
  logloss_knn_2[length(logloss_knn_2)+1] <- LogLoss(ypred2, (as.integer(valid_2$click)-1))
}
### print out the log loss for all the differnt k's, the best k and the log loss associated with best k
logloss_knn_2
bestk2 <- which.min(logloss_knn_2)*2 + 5
bestk2
logloss_knn_2[(bestk2-5)/2] # 1.836988 as the lowest Log Loss
### build new model with best k for the ensemble models
model_knn_2 <-knn(train=train_2_knn,test=valid_2_knn,cl = factor(train_2$click),prob = TRUE,k=bestk)



## third model
set.seed(13)
logloss_knn_3 <- c()
for (i in seq(11,99,2)) {
  out3 <- knn(train=train_3_knn,test=valid_3_knn,cl = factor(train_3$click),prob = TRUE,k=i)
  ypred3 <- matrix(attr(out3,'prob'))
  logloss_knn_3[length(logloss_knn_3)+1] <- LogLoss(ypred3, (as.integer(valid_3$click)-1))
}
### print out the log loss for all the differnt k's, the best k and the log loss associated with best k
logloss_knn_3
bestk3 <- which.min(logloss_knn_3)*2 + 5
bestk3
logloss_knn_3[(bestk3-5)/2] # 1.813879 as the lowest Log Loss
### build new model with best k for the ensemble models
model_knn_3 <-knn(train=train_3_knn,test=valid_3_knn,cl = factor(train_3$click),prob = TRUE,k=bestk)



## fourth model
set.seed(14)
logloss_knn_4 <- c()
for (i in seq(11,99,2)) {
  out4 <- knn(train=train_4_knn,test=valid_4_knn,cl = factor(train_4$click),prob = TRUE,k=i)
  ypred4 <- matrix(attr(out4,'prob'))
  logloss_knn_4[length(logloss_knn_4)+1] <- LogLoss(ypred4, (as.integer(valid_4$click)-1))
}
### print out the log loss for all the differnt k's, the best k and the log loss associated with best k
logloss_knn_4
bestk4 <- which.min(logloss_knn_4)*2 + 5
bestk4
logloss_knn_4[(bestk4-5)/2] # 1.793094 as the lowest Log Loss
### build new model with best k for the ensemble models
model_knn_4 <-knn(train=train_4_knn,test=valid_4_knn,cl = factor(train_4$click),prob = TRUE,k=bestk)



## fifth model
set.seed(15)
logloss_knn_5 <- c()
for (i in seq(11,99,2)) {
  out5 <- knn(train=train_5_knn,test=valid_5_knn,cl = factor(train_5$click),prob = TRUE,k=i)
  ypred5 <- matrix(attr(out5,'prob'))
  logloss_knn_5[length(logloss_knn_5)+1] <- LogLoss(ypred5, (as.integer(valid_5$click)-1))
}
### print out the log loss for all the differnt k's, the best k and the log loss associated with best k
logloss_knn_5
bestk5 <- which.min(logloss_knn_5)*2 + 5
bestk5
logloss_knn_5[(bestk5-5)/2] # 1.748267 as the lowest Log Loss
### build new model with best k for the ensemble models
model_knn_5 <-knn(train=train_5_knn,test=valid_5_knn,cl = factor(train_5$click),prob = TRUE,k=bestk)



## ensemble methods
### a. use mean
knn_pred_ensemble <- data.frame(c(1:nrow(valid_all)))
for (i in 1:5) {
  model_a <- get(paste0("model_knn_", i))
  ypred_a <- matrix(attr(model_a,'prob'))
  agent <- ypred_a
  knn_pred_ensemble <- cbind.data.frame(knn_pred_ensemble, agent)
}
colnames(knn_pred_ensemble) <- c("num", "m1", "m2", "m3", "m4", "m5")
knn_pred_ensemble <- knn_pred_ensemble %>% mutate(avgprob = (m1 + m2 + m3 + m4 + m5) / 5)
logloss_knn_mean <- LogLoss(knn_pred_ensemble$avgprob, (as.integer(valid_all$click)-1))
logloss_knn_mean # Log Loss equals to 1.568061

### b. use median
knn_pred_ensemble$medianprob <- apply(knn_pred_ensemble[,2:6], 1, median) 
logloss_knn_median <- LogLoss(knn_pred_ensemble$medianprob, (as.integer(valid_all$click)-1)) 
logloss_knn_median # Log Loss equals to 1.727933

### c. use logistic regression
model_knn_ensemble_logreg <- glm(valid_all$click ~ m1 + m2 + m3 + m4 + m5, data = knn_pred_ensemble, family = binomial(link='logit'))
knn_pred_ensemble_valid <- data.frame(c(1:nrow(valid_6)))
for (i in 1:5) {
  model_c <- get(paste0("model_knn_", i))
  ypred_c <- matrix(attr(model_c,'prob'))
  knn_pred_ensemble_valid <- cbind.data.frame(knn_pred_ensemble_valid, agent)
}
colnames(knn_pred_ensemble_valid) <- c("num", "m1", "m2", "m3", "m4", "m5")
ypred_d <- predict(model_knn_ensemble_logreg, newdata = knn_pred_ensemble_valid[, 2:6], type="response")
logloss_knn_logreg <- LogLoss(ypred_d, (as.integer(valid_6$click)-1)) 
logloss_knn_logreg # Log Loss equals to 0.4888089


# the logistic regression ensemble model is the best model for Knn


# Part 8: Modeling - Neural network -----------------------------------------------------
## this part will be done by Python, we write out the dumminized data as well as the trimmed data with only the important features for neural network
write.csv(train_1_dum, file = "train_1_dum.csv", row.names = FALSE)
write.csv(train_2_dum, file = "train_2_dum.csv", row.names = FALSE)
write.csv(train_3_dum, file = "train_3_dum.csv", row.names = FALSE)
write.csv(train_4_dum, file = "train_4_dum.csv", row.names = FALSE)
write.csv(train_5_dum, file = "train_5_dum.csv", row.names = FALSE)
write.csv(valid_1_dum, file = "valid_1_dum.csv", row.names = FALSE)
write.csv(valid_2_dum, file = "valid_2_dum.csv", row.names = FALSE)
write.csv(valid_3_dum, file = "valid_3_dum.csv", row.names = FALSE)
write.csv(valid_4_dum, file = "valid_4_dum.csv", row.names = FALSE)
write.csv(valid_5_dum, file = "valid_5_dum.csv", row.names = FALSE)
write.csv(valid_6_dum, file = "valid_6_dum.csv", row.names = FALSE)

write.csv(train_1_trim, file = "train_1_trim.csv", row.names = FALSE)
write.csv(train_2_trim, file = "train_2_trim.csv", row.names = FALSE)
write.csv(train_3_trim, file = "train_3_trim.csv", row.names = FALSE)
write.csv(train_4_trim, file = "train_4_trim.csv", row.names = FALSE)
write.csv(train_5_trim, file = "train_5_trim.csv", row.names = FALSE)
write.csv(valid_1_trim, file = "valid_1_trim.csv", row.names = FALSE)
write.csv(valid_2_trim, file = "valid_2_trim.csv", row.names = FALSE)
write.csv(valid_3_trim, file = "valid_3_trim.csv", row.names = FALSE)
write.csv(valid_4_trim, file = "valid_4_trim.csv", row.names = FALSE)
write.csv(valid_5_trim, file = "valid_5_trim.csv", row.names = FALSE)
write.csv(valid_6_trim, file = "valid_6_trim.csv", row.names = FALSE)


# Part 9: Modeling - Ensemble methods ----------------------------------------------------
## ensemble the best tree and the best ridge since their Log Loss are below 0.41
### make the predictions of the ensembled tree on valid_all and valid_6 data
tree_yhat_ensemble <- predict(model_tree_ensemble_logreg, newdata = tree_pred_ensemble[, 2:6], type="response")
tree_yhat_ensemble_valid_6 <- predict(model_tree_ensemble_logreg, newdata = tree_pred_ensemble_valid[, 2:6], type="response")

### a. use mean
#### combine the predctions from the best tree and the best ridge together
treeandridge_yhat_ensemble <- cbind.data.frame(valid_all$click, logit_ridge_4_yhat[,1], tree_yhat_ensemble)
treeandridge_yhat_ensemble_valid_6 <- cbind.data.frame(valid_6$click, logit_ridge_4_yhat_valid_6[,1], tree_yhat_ensemble_valid_6)
colnames(treeandridge_yhat_ensemble) <- c('click','pred_1','pred_2')
colnames(treeandridge_yhat_ensemble_valid_6) <- c('click','pred_1','pred_2')

treeandridge_yhat_ensemble <- treeandridge_yhat_ensemble %>% mutate(avgprob = (pred_1 + pred_2) / 2)
logloss_treeandridge_mean <- LogLoss(treeandridge_yhat_ensemble$avgprob, (as.integer(valid_6$click)-1))
logloss_treeandridge_mean # Log Loss = 0.5161907

#### b. use median (same as average when there are only two predictions)

#### c. use logistic regression
treeandridge_ensemble_logreg <- glm(click ~ pred_1 + pred_2, data = treeandridge_yhat_ensemble, family = binomial(link='logit'))
treeandridge_ensemble_logreg_yhat <- predict(treeandridge_ensemble_logreg, newdata = treeandridge_yhat_ensemble_valid_6, type="response")
treeandridge_ensemble_logreg_logloss <- LogLoss(treeandridge_ensemble_logreg_yhat, (as.integer(valid_6$click)-1)) 
treeandridge_ensemble_logreg_logloss  # Log Loss = 0.4048252


# Part 10: Prediction ---------------------------------------------------------------
## factorize every column in test data
test_clean$click <- 0
test_clean <- test_clean %>% select(click, everything())
test_clean[] <- lapply(test_clean, factor)

## use the best ridge model to do prediction
model <- logit_ridge_4$best_out
XTest <- model.matrix(click ~ . ,test_clean)[,-1]
best_ridge_yhat <- predict(model,newx=XTest,type = "response")

## write out the outcome
submission$`P(click)` <- best_ridge_yhat
write.csv(submission, file = "ProjectSubmission-Team11.csv", row.names = FALSE)
