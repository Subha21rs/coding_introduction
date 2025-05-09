
# Packages are codes (similar to python module) that add new functions to R
# Base packages are already installed, but they need to be loaded
# There are some third party packages, that needs to be installed to use.
# We can get more packages from CRAN (https://cran.r-project.org/)
# To install a R package you can run the r code
# install.packages("package_name")
# To load the packages in r to use, run
# library(package_name)


####### DATA TYPES
n1 <- 15
typeof(n1)

n2 <- 1.5
typeof(n2)

c1 <- 'c'
typeof(c1)

c2 <- 'a string'
typeof(c2)

l1 <- TRUE
typeof(l1)

### vectors and basic operations
v= c(10,20,30,40)
is.vector(v)
mean(v)
sum(v)
length(v)
u=v[v>20]
print(u)

## matrix
m1=matrix(c(1,2,3,4), nrow=2,byrow=TRUE)

## data frame
vNumeric <- c(1,2,3)
vCharacter <- c('a','b','c')
vLogical <- c(T,F,T)

dfa <- cbind(vNumeric,vCharacter,vLogical)
df= as.data.frame(dfa)

### sequence generation
x1= seq(10)
print(x1)

(x2= seq(30,0, by=-3))

####### Input, Output statement
var1= readline('Enter a number:')
var1=as.numeric(var1)
var2=readline('Enter another number:')
var2=as.numeric(var2)
print(var1+var2)

############### Plots
library(datasets)
head(iris)
summary(iris)

plot(iris$Petal.Length)
plot(iris$Species,iris$Petal.Width)
plot(iris$Petal.Length,iris$Petal.Width)
plot(iris)

# plot with customization
plot(iris$Petal.Length,iris$Petal.Width,col='red',pch=20,main='petal length vs petal width',
     xlab='Petal Length', ylab='Petal Width')

### Histogram
hist(iris$Sepal.Length)
hist(iris$Sepal.Width)

# histogram for each species using customization
par(mfrow=c(3,1))
hist(iris$Petal.Width[iris$Species=='setosa'], xlim=c(0,3), breaks=9, main='Setosa',
     xlab='', col='red')

hist(iris$Petal.Width[iris$Species=='versicolor'], xlim=c(0,3), breaks=9, main='Versicolor',
     xlab='', col='purple')

hist(iris$Petal.Width[iris$Species=='virginica'], xlim=c(0,3), breaks=9, main='Virginica',
     xlab='', col='green')

par(mfrow=c(1,1))

## barplots 
head(mtcars)
cylinder= table(mtcars$cyl)
barplot(cylinder)

### more plots in one figure
?lynx
head(lynx)

hist(lynx, breaks=14, freq=FALSE, col='blue',main='Lynx data',
     xlab='Number of Lynx Trapped')
# add a normal distribution
curve(dnorm(x,mean=mean(lynx),sd=sd(lynx)), col='black',lwd=2, add=TRUE)
# add kernel density estimates
lines(density(lynx), col='red',lwd=2)

######### Importing data
data<- read.csv('data/example_data.csv')
head(data)

######## Linear regression
?USJudgeRatings
head(USJudgeRatings)
data <- USJudgeRatings

reg1= lm(RTEN~., data)
summary(reg1)


######## Functions in R
squre <- function(x){
  y <- x^2
  return (y)
}

squre(3)

##### for loop
for (i in 1:5){
  print(i)
}

sum_val <- 0
for (i in 1:100) {
  sum_val <- sum_val + i
}
print(sum_val)

####### if-else statement
x <- -10
if (x>0){
  print('Positive Number')
}else if (x<0){
  print('Negative number')
}else{
  print(paste(x, "is zero"))
}

# vectorized ifelse()
x <- c(10,-5,0,4)
result <- ifelse(x>0, 'Positive', 'Not positive')
print(result)

#### apply function
mat <- matrix(1:9, nrow = 3)
print(mat)
apply(mat, 1, sum) # row sum
apply(mat, 2, mean) #   column means

### lapply() and sapply()
lst <- list(a = 1:5, b = 6:10)
lapply(lst, mean)
sapply(lst, mean)


####### plot using ggplot2
library(ggplot2)
df <- data.frame(Name = c("A", "B", "C", "D"),
                 Score = c(85, 90, 78, 92),
                 Group = c("X", "X", "Y", "Y"))
ggplot(df, aes(x = Group, y = Score, fill = Name)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_classic()

data<- read.csv('data/example_data.csv')
x <- seq(0, 500, length.out = 32)
member=data[,33]
data=data[,-33]
cl1=colMeans( data[member==1,] )
cl2=colMeans( data[member==2,] )
cl3=colMeans( data[member==3,] )
cl4=colMeans( data[member==4,] )
cl5=colMeans( data[member==5,] )
cl6=colMeans( data[member==6,] )

cl <- c(cl1,cl2,cl3,cl4,cl5,cl6)
x_rep= rep(x, times=6)
class <- rep(c('class1','class2','class3','class4','class5','class6'), each=32)
df_plot <- data.frame(x=x_rep,y=cl,class=class)

plott = ggplot(df_plot, aes(x = x, y = y, color = class)) +
  geom_line() +
  #geom_point(size=3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(x = "Time (ms)", y = "Amplitude", color =NULL, shape=NULL) +
  scale_color_manual(values = c("class1" = "blue","class2" = "cyan","class3" = "azure4",
                                "class4" = "magenta","class5" = "green","class5" = "red")) +
  #scale_shape_manual(values = c("PO" = 16,   # solid circle
                                # "NB" = 17,        # solid triangle
                                # "ZIP" = 15,       # solid square
                                # "ZINB" = 3,       # plus
                                # "HP" = 4, # cross
                                # "HNB" = 8   # star
  #)) +
  theme_minimal()+
  theme(
    axis.title.x = element_text(size = 30),   # X label size
    axis.title.y = element_text(size = 30),   # Y label size
    axis.text.x = element_text(size = 18),    # X tick label size
    axis.text.y = element_text(size = 18),     # Y tick label size
    legend.text = element_text(size = 25),
    legend.title = element_blank(),
    panel.grid.minor.x = element_blank()
  )
plott
##ggsave("different_plots.pdf", plot = plott, width = 10, height = 6)

############ 3D plots using plotly
#install.packages("plotly")
library(plotly)

# Scatter plot
# Sample 3D data (from iris)
plot_ly(data = iris, 
        x = ~Sepal.Length, 
        y = ~Petal.Length, 
        z = ~Sepal.Width, 
        color = ~Species, 
        colors = c("red", "green", "blue"),
        type = "scatter3d", 
        mode = "markers",
        marker = list(size = 5, opacity = 0.8)) %>%
  layout(title = "3D Scatterplot of Iris Data",
         scene = list(
           xaxis = list(title = 'Sepal Length'),
           yaxis = list(title = 'Petal Length'),
           zaxis = list(title = 'Sepal Width')
         ))

# surface plot
x <- seq(-10, 10, length.out = 50)
y <- seq(-10, 10, length.out = 50)
z <- outer(x, y, function(x, y) sin(sqrt(x^2 + y^2)))  # z = sin(sqrt(x² + y²))

# Plot the surface
plot_ly(x = ~x, y = ~y, z = ~z) %>%
  add_surface(colorscale = 'Viridis') %>%
  layout(title = "3D Surface Plot of sin(sqrt(x² + y²))",
         scene = list(
           xaxis = list(title = "X"),
           yaxis = list(title = "Y"),
           zaxis = list(title = "Z")
         ))


##### simulation
#set.seed(13)
t_vals= rt(1000, df=10)
hist(t_vals, breaks=30,main='t-distribution')

# from normal distribution
x <- rnorm(1000, mean = 10, sd = 2)
hist(x, breaks = 30, main = "Normal distribution", col = "blue")

# from multivariate normal
mu <- c(5,10)
sigma <- matrix(c(4,2,2,3), nrow=2)
sample = mvrnorm(n=1000, mu=mu,Sigma=sigma)

head(sample)

####### Hypothesis testing
t_test= t.test(mpg ~ am, data = mtcars) 
t_test
wilcox.test(mpg ~ am, data = mtcars)

####### ANOVA
library(dplyr)
head(mtcars)
# one way anova
aov1 <- aov(mpg ~ factor(gear), data=mtcars) 
summary(aov1)

# two way anova
aov2 <- aov(mpg ~ factor(gear) * factor(am),data=mtcars) 
summary(aov2) 

############Logistic regression
data(iris)
iris_bin= subset(iris, Species!='setosa')
iris_bin$Species <- factor(iris_bin$Species)
set.seed(1)
train_index<-sample( seq_len(nrow(iris_bin)), size=0.7*nrow(iris_bin))
train_data <- iris_bin[train_index, ]
test_data  <- iris_bin[-train_index, ]

model <- glm(Species ~ .,
             data = train_data, family = binomial)
summary(model)

pred_prob<-predict(model,newdata = test_data,type='response')

actual <- ifelse(test_data$Species == "virginica", 1, 0)
predicted <- ifelse(pred_prob>0.5, 1, 0)
table(Predicted=predicted,Actual=actual)
acc=mean(predicted==actual)
print(acc)

####### PCA
data(iris)
iris_numeric <- iris[, 1:4]

pca <- prcomp(iris_numeric, scale. = TRUE)
summary(pca)

plot(pca$x[, 1], pca$x[, 2],
     col = as.numeric(iris$Species),
     pch = 19,
     xlab = "PC1",
     ylab = "PC2",
     main = "PCA: PC1 vs PC2")

legend("topright", legend = levels(iris$Species),
       col = 1:3, pch = 19,cex=0.7)

############ LDA
library(MASS)

set.seed(123)

n <- 100
mu1 <- c(1, 2)  
mu2 <- c(3, 4)
sigma <- matrix(c(1, 0.5, 0.5, 1), nrow = 2)

class1 <- mvrnorm(n, mu = mu1, Sigma = sigma)
class2 <- mvrnorm(n, mu = mu2, Sigma = sigma)
data <- rbind(class1, class2)

labels <- factor(c(rep(0, n), rep(1, n)))

df <- data.frame(data, class = labels)
names(df) <- c("X1", "X2", "Class")

set.seed(123)

train_indices <- sample(1:nrow(df), size = 0.7 * nrow(df))

trainData <- df[train_indices, ]
testData <- df[-train_indices, ]

lda_model <- lda(Class ~., data = trainData)

lda_pred <- predict(lda_model, testData)

predicted_class <- lda_pred$class
(confusion_matrix <- table(predicted_class, testData$Class))
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))




##########################################
#### Some questions ######################
# 1. Create a vector of integers from 1 to 10. Find the sum, mean, and variance of this vector.
# 2. Create a data frame with the following columns: Name, Age, Gender, and Height. Fill it with data for 5 people and display the data frame.
# 3. Create a sequence of numbers from 5 to 50 with a step size of 5, and extract the numbers that are divisible by 10.
# 4. Use the iris dataset to select all rows where the species is setosa and the sepal length is greater than 5.
# 5. Write a function to check if a number is prime or not.
# 6. Create a data frame with two columns: Height and Weight, representing the height and weight of 10 individuals. Calculate the Body Mass Index (BMI) for each individual and add it as a new column to the data frame.
# 7. Perform a one-sample t-test to determine if the average mpg of cars in the mtcars dataset is different from 20.
# 8. Fit a linear regression model to predict mpg from wt (weight) in the mtcars dataset. Plot the regression line on a scatter plot.
# 9. Perform a multiple linear regression with mpg as the dependent variable and wt, hp, and qsec as independent variables in the mtcars dataset.
# 10. Simulate 1000 samples from a normal distribution with mean = 50 and standard deviation = 10. Plot the histogram and overlay the theoretical normal curve.


