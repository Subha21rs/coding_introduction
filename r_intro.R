
# Packages are codes (similar to python module) that add new functions to R
# Base packages are already installed, but they need to be loaded
# There are some third party packages, that needs to be installed to use.
# We can get more packages from CRAN (https://cran.r-project.org/)
# To install a R package you can run the r code
# install.packages("package_name")
# To load the packages in r to use, run
# library(package_name)


##### YET TO CODE
# 1. ggplot


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


######### Importing data
data<- read.csv('/home/kd/Downloads/AIS_2025_code_session/example_data.csv')

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

