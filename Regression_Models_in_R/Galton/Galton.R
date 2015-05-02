library("UsingR")
data(galton)
par(mfrow<-c(1,2))
x<-galton$parent
y<-galton$child
pdf("Galton.pdf")
plot(x,y,xlab="Child's height",
     ylab="Parent's height",
     cex=0.5,
     bg="lightblue",
     col="black",
     main="Children vs parent's height distribution")
hist(galton$child,col="blue",breaks=50)
hist(galton$parent,col="blue",breaks=50)
par(mfrow<-c(1,1))
b1<-cor(y,x)*sd(y)/sd(x)
b0<-mean(y)-b1*mean(x)
print("Result of calculation is : ");
print(c(b0,b1));
print("Result of Regression is : ");
print(coef(lm(y~x)));
plot(x,y,xlab="Fathers' Height"
     ,ylab="Sons' Height",
     bg="lightblue",
     col="black",
     frame=F,
     cex=0.5,
     main="Linear Regressed Curve")
abline(lm(y~x),lwd=2,col=2)
e<-resid(lm(y~x))
plot(x,e,xlab="Father's Height",
     ylab="Residuals",
     col="red",
     frame=T,
     main="Residuals vs Parent Height")
abline(h=0)
#REgression for degree 2
fit<-lm(y~poly(x,2),data=Galton)
print(coef(summary(fit)))
htlim=range(x)
print(htlim)
hgrid<-seq(from=htlim[1],to=htlim[2])
pred<-predict(fit,newdata=data.frame(x=hgrid),se=T)
se.band=cbind(pred$fit+2*pred$se.fit,pred$fit-2*pred$se.fit)
#print(se.band)
plot(x,y,xlim=htlim,xlab="Fathers' Height"
     ,ylab="Sons' Height",
     bg="lightblue",
     col="black",
     frame=F,
     cex=0.5,
     main="Polynomial Regressed Curve Degree=2")
#print(length(hgrid))
#print(length(pred$fit))
lines(hgrid,pred$fit,lwd=2,col=2)
matlines(hgrid,se.band,lwd=1,col="blue",lty=3)

#Regression for degree 3
fit<-lm(y~poly(x,3),data=Galton)
pred<-predict(fit,newdata=data.frame(x=hgrid),se=T)
se.band=cbind(pred$fit+2*pred$se.fit,pred$fit-2*pred$se.fit)
#print(se.band)
plot(x,y,xlim=htlim,xlab="Fathers' Height"
     ,ylab="Sons' Height",
     bg="lightblue",
     col="black",
     frame=F,
     cex=0.5,
     main="Polynomial Regressed Curve Degree=3")
lines(hgrid,pred$fit,lwd=2,col=2)
matlines(hgrid,se.band,lwd=1,col="blue",lty=3)

#Regression for degree 4
fit<-lm(y~poly(x,4),data=Galton)
pred<-predict(fit,newdata=data.frame(x=hgrid),se=T)
se.band=cbind(pred$fit+2*pred$se.fit,pred$fit-2*pred$se.fit)
plot(x,y,xlim=htlim,xlab="Fathers' Height"
     ,ylab="Sons' Height",
     bg="lightblue",
     col="black",
     frame=F,
     cex=0.5,
     main="Polynomial Regressed Curve Degree=4")
lines(hgrid,pred$fit,lwd=2,col=2)
matlines(hgrid,se.band,lwd=1,col="blue",lty=3)

#Regression for degree 5
fit<-lm(y~poly(x,5),data=Galton)
pred<-predict(fit,newdata=data.frame(x=hgrid),se=T)
se.band=cbind(pred$fit+2*pred$se.fit,pred$fit-2*pred$se.fit)
#print(se.band)
plot(x,y,xlim=htlim,xlab="Fathers' Height"
     ,ylab="Sons' Height",
     bg="lightblue",
     col="black",
     frame=F,
     cex=0.5,
     main="Polynomial Regressed Curve Degree=5")
lines(hgrid,pred$fit,lwd=2,col=2)
matlines(hgrid,se.band,lwd=1,col="blue",lty=3)
dev.off()