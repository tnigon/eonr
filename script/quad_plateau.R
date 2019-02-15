library(nls2)
library(MASS)
library(easynls)
library(DescTools)
sns17w_pre <- read.csv("G:/SOIL/GIS/SNS/eonr/sns17w_pre.csv")
x <- sns17w_pre$rate_n_applied_lbac
y <- sns17w_pre$yld_grain_dry_buac
data0 <- data.frame(x = x, y = y)

sns15w_pre <- read.csv("G:/SOIL/GIS/SNS/eonr/sns15w_pre.csv")
x <- sns15w_pre$rate_n_applied_lbac
y <- sns15w_pre$yld_grain_dry_buac
data15wpre <- data.frame(x = sns17w_pre$rate_n_applied_lbac, y = sns17w_pre$grtn)

ratio <- 0.1
nlsfit_r <- nls2(formula='y ~ th11-(2*th2*th12-ratio) * x + th12 * x^2',
                 data=data15wpre,
                 start=list(th11=150, th2=150, th12=-0.01))
summary(nlsfit_r)
confint(nlsfit_r, 'th2', level=0.9)
profile(nlsfit_r, 'th2')


#'th2 = -beta1/2*th12'
#'
'y ~ (a + b * x + c * I(x^2)) * (x <= -0.5 * b/c) + (a + I(-b^2/(4 * c))) * (x > -0.5 * b/c)'
'
a = th11
b = - (2*th2*th12-ratio)
c = th12
'
' * (x <= -0.5*-(2*th2*th12-ratio)/th12)'
' + (th11+I((2*th2*th12-ratio)^2/(4*th12)))'
' * (x > -0.5*-(2*th2*th12-ratio)/th12)'

' + (th11+I((2*th2*th12-ratio)^2/(4*th12))) * (x > -0.5*-(2*th2*th12-ratio)/th12)'
nls_qp <- nls2(formula='y~(th11 - (2*th2*th12-ratio)*x + th12*I(x^2)) * (x <= -0.5*-(2*th2*th12-ratio)/th12) + (th11+I((2*th2*th12-ratio)^2/(4*th12))) * (x > -0.5*-(2*th2*th12-ratio)/th12)',
               data=data0,
               start=list(th11=160, th2=150, th12=-0.01))
summary(nls_qp)
confint(nls_qp, 'th2', level=0.9)


x <- sns17w_pre$rate_n_applied_lbac
y <- sns17w_pre$grtn
data1 <- data.frame(x = x, y = y)
plot(data.1)

nls_q <- nls2(formula='y ~ a + b*x + c*x^2',
              data=data0,
              start=list(a=150, b=1, c=-0.01))
summary(nls_q)
confint(nls_q, 'c', level=0.9)

nls_qp <- nls2(formula='y ~ (a + b * x + c * I(x^2)) * (x <= -0.5 * b/c) + (a + I(-b^2/(4 * c))) * (x > -0.5 * b/c)',
              data=data15wpre,
              start=list(a=150, b=3, c=-0.01))
summary(nls_qp)
confint(nls_qp, 'c', level=0.9)

nls_qp <- nls2(formula='y ~ (a + b * x + c * I(x^2)) * (x <= -0.5 * b/c) + (a + I(-b^2/(4 * c))) * (x > -0.5 * b/c)',
               data=data.1,
               start=list(a=600, b=3, c=-0.01))
summary(nls_qp)
confint(nls_qp, 'c', level=0.9)
profile(nls_qp, 'b')

ex1 <- read.csv("G:/SOIL/GIS/SNS/eonr/ex1.csv")
data_ex <- data.frame(x = ex1$x, y = ex1$y)
ex1 <- nls2(formula='y ~ t1*x + t2*x + t4*exp(t3*x)',
            data=data_ex,
            start=list(t1=-0.05, t2=1.04, t3=-0.74, t4=-0.51))
m <- nls(y ~ t1*x + t2*x + t4*exp(t3*x),
         data=data_ex,
         start=list(t1=-0.05, t2=1.04, t3=-0.74, t4=-0.51),
         trace=T)
summary(ex1)
confint(ex1, 'c', level=0.9)
profile(ex1, 'b')


nls_q_easy = nlsfit(data1, model=2)
nls_q_easy$Parameters

nls_lp_easy = nlsfit(data1, model=3)
nls_lp_easy$Parameters

nls_qp_easy = nlsfit(data15wpre, model=4)
nls_qp_easy$Parameters
# confint(nls_qp_easy, 'b', level=0.9)


bid <- c(1,5,10,20,30,40,50,75,100,150,200)
n <- c(31,29,27,25,23,21,19,17,15,15,15)
y <- c(0,3,6,7,9,13,17,12,11,14,13)
m1 <- glm(y/n~log(bid),weights=n,family=binomial)

k <- 200
b2 <- seq(.7,2,length=k)
w <- rep(0,k)
for(i in 1:k){
  mm <- glm(y/n~1,offset=b2[i]*log(bid),weights=n,family=binomial)
  w[i] <- logLik(mm)
  }
plot(b2,w,type="l",ylab="Profile log-likelihood",cex.lab=1.3,cex.axis=1.3)
abline(h=logLik(m1)-qchisq(.95,1)/2,lty=2)

rate_data <- c(233, 234, 235, 236, 237, 238, 239)
val <- c(0, 0.1, 0.3, 0.6, 1.05, 1.65, 2.4)
data2 <- data.frame(x = rate_data, y = val)

A <- structure(list(rate_data <- c(233, 234, 235, 236, 237, 238, 239), 
                    val <- c(0, 0.1, 0.3, 0.6, 1.05, 1.65, 2.4)), .Names = c("rate_data", "val"), class = "data.frame")

exponential.model <- lm(log(rate_data)~ val)
summary(exponential.model)

Counts.exponential2 <- exp(predict(exponential.model,list(rate_data=seq(230, 239, 1))))


library(boot)
st <- cbind(data15wpre, fit = fitted(nls_qp))
bf <- function(rs, i) { st$y <-st$fit + rs[i]
  coef(nls(y ~ th11-(2*th2*th12-ratio)*x + th12*x^2,st, start = coef(nls_qp)))}

bf <- function(rs, i) { st$y <-st$fit + rs[i]
  coef(nls2(formula='y ~ (a + b * x + c * I(x^2)) * (x <= -0.5 * b/c) + (a + I(-b^2/(4 * c))) * (x > -0.5 * b/c)',st, start = coef(nls_qp)))}

rs <- scale(resid(nls_qp), scale = F) # remove the mean
nlsfit_boot <- boot(rs, bf, R = 9999)
boot.ci(nlsfit_boot, index=2, type = c("norm", "basic", "perc", "bca"), conf = 0.9)




