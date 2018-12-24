
##loading the required packages
library(MASS)
library(caret)
library(data.table)
library(gains)
library(lift)

##loading the data into the dataframe

spambase.df <- fread('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')


##loading the names into the vector

spambase.names <- c("word_freq_make","word_freq_address","word_freq_all","word_freq_3d","word_freq_our","word_freq_over",          
                    "word_freq_remove","word_freq_internet","word_freq_order","word_freq_mail","word_freq_receive","word_freq_will",          
                    "word_freq_people","word_freq_report","word_freq_addresses","word_freq_free","word_freq_business","word_freq_email",         
                    "word_freq_you","word_freq_credit","word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp",            
                    "word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs","word_freq_telnet","word_freq_857",           
                    "word_freq_data","word_freq_415","word_freq_85","word_freq_technology","word_freq_1999","word_freq_parts",         
                    "word_freq_pm","word_freq_direct","word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project",       
                    "word_freq_re","word_freq_edu","word_freq_table","word_freq_conference","char_freq_;","char_freq_(",             
                    "char_freq_[","char_freq_!","char_freq_$","char_freq_#","capital_run_length_average",  
                    "capital_run_length_longest","capital_run_length_total","spam")

## allocating the colnames to the dataframe

colnames(spambase.df) <- spambase.names

##Normalizing the data as they have different units of measurement
spam.norm.df  <- preProcess(spambase.df[,-58], method = c("center", "scale"))
spambase.norm <- predict(spam.norm.df, spambase.df)
#-----------------------------------------------------------------------------------------------------#


##Spam class avg and non spam avg for predictor variables
spam_avg <- aggregate(spambase.norm[,1:57] , list(spambase.norm$spam), mean)
View(spam_avg)

##Taking difference btw Spam class avg and non spam avg for predictor variables
diff_spam_avg <- abs(spam_avg[1,] - spam_avg[2,])
View(diff_spam_avg)

##Picking top 10 predictor variables
highest_diff <-sort(diff_spam_avg[,-1],decreasing = TRUE)[,1:10]
View(highest_diff)

##Top 10 predictor variables
predicted_var <- names(highest_diff[,1:10])
predicted_var

##The Top 10 predictors are the following:
#"word_freq_your"           "word_freq_000"            "word_freq_remove"         "char_freq_$"             
#"word_freq_you"            "word_freq_free"           "word_freq_business"       "word_freq_hp"            
#"capital_run_length_total" "word_freq_our"

#-------------------------------------------------------------------------------------------------------------#

##Reducing the dataset 
red_spambase.df <-spambase.norm[,..predicted_var]
red_spambase.df <- cbind.data.frame(red_spambase.df,spambase.norm[,58])
View(red_spambase.df)

##set the seed
set.seed(2)

#Partitioning the data set into training and validation 

training.index <- createDataPartition(red_spambase.df$spam, p = 0.8, list = FALSE)

spambase.train <- red_spambase.df[training.index, ]

spambase.valid <- red_spambase.df[-training.index, ]


## Performing the linear discriminant analysis
lda2 <- lda(spam ~., data = spambase.train)
lda2

#-------------------------------------------------------------------------------------------------------------#


pred.test <- predict(lda2, spambase.valid)


##Obtaining the confusion matrix
confusionMatrix(as.factor(pred.test$class),as.factor(spambase.valid$spam))

##from the result accuracy obtained is 84.02%


##Plotting cumulative lift chart

Post_test_var <- as.data.frame(pred.test$posterior)

Post_test_var <- cbind(Post_test_var,spambase.valid$spam)

Post_test_var <- Post_test_var[,-1]

colnames(Post_test_var) <- c("Prediction","Actual")

## Sorting the data
Top10_gain <- Post_test_var[order(Post_test_var$Prediction, decreasing = T),]

## Plotting lift chart
lift_chart <- lift(relevel(as.factor(Actual), ref="1") ~ Prediction,data = Top10_gain)
xyplot(lift_chart, plot = "gain")

##Converting into numeric 
Top10_gain$Actual <- as.numeric(Top10_gain$Actual)

##Obtain Gain
gain <- gains(Top10_gain$Actual, Top10_gain$Prediction)

##Plotting decile wise lift chart
##We can see that the model categorises 2.5 times better than selecting any 10% random categorization.

barplot(gain$mean.resp / mean(Top10_gain$Actual), names.arg = gain$depth, xlab = "Percentile",
        ylab = "Mean Response", main = "Decile chart", ylim = c(0,3.0))

#-------------------------------------------------------------------------------------------------------------#








