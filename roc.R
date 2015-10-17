#!usr/bin/Rscript
# ROC analysis

# stole this code with some slight change from corey chiver (https://gist.github.com/cjbayesian/6921118)
## Discrete integration for AUC calc
## Δx.y1 + 1/2Δx.Δy  <- summation of trapezoids
disc_integrate <- function(x,y)
	{
	dint_inner = function(i) {
		( x[i]-x[i-1] ) * y[i-1] + 0.5 * ( x[i]-x[i-1] ) * ( y[i]-y[i-1] )  	 
		}
    f <- cbind(x,y)
   	## Sort by x, then by y (ascending)
   	f <- f[ order( f[,1], f[,2] ), ] 
   	dint <- 0
   	x <- f[,1]
   	y <- f[,2]
   	dint <- sapply( 2:length(x), dint_inner ) 
   	   
   	dint <- sum(dint)
   	return(dint)
	}

ROC = function(predictions) {
	# predictions is a 4-column df of (1) prediction scores and (2) true classifications, with annotations in 3 and 4.
	# only need first 2 cols
	predictions = data.matrix( predictions[,1:2] )
	class(predictions) = 'numeric'
	predictions = predictions[ order(predictions[,1], decreasing=TRUE), ]
	truePos = c()
	falsePos = c()
	totalTrue = sum( predictions[,2] )	# assumed to be 0/1, with 1=true
	
	for (i in 1:nrow(predictions)) {
#		if (i %% 1000 ==0) {
#			print(i)
#			}
		truePos[i] = sum(predictions[1:i, 2])
		falsePos[i] = length( which(predictions[1:i,2]==0) )
	#	print(truePos[i])
	#	print(falsePos[i])
		}
	
	AUC = disc_integrate( falsePos/max(falsePos), truePos/max(truePos) )
	
	# false and true pos
	par( mfrow=c(2,1) )
	plot( falsePos, truePos, type='l', xlab = 'false positives', ylab = 'true positives', main = paste('AUC =', AUC), ylim = c(0, totalTrue), xlim = c(0, nrow(predictions) - totalTrue) )	
	abline( a = 0, b = totalTrue /( nrow(predictions) - totalTrue ), lty= 3 )
	# precision and recall
	plot( truePos/totalTrue, truePos / (truePos+falsePos), type='l', xlab = 'recall', ylab='precision', ylim=c(0,1), xlim = c(0,1) )
	#abline(a=1,b=-1,lty=3)
	
	return( cbind(falsePos,truePos) )

	}


ROC_cat = function(predictions) {
	# predictions is a 4-column df of (1) prediction scores and (2) true classifications, with annotations in 3 and 4.
	# only need first 2 cols
	predictions = data.matrix(predictions[,1:2])
	class(predictions) = 'numeric'
	predictions = predictions[order(predictions[,1],decreasing=TRUE),]
	truePos = c()
	falsePos = c()
	trueCalls = rep(0,length(predictions))
	totalTrue = length( which(predictions[,1] == predictions[,2]) )	# changed to accommodate multi-label
	
	for (i in 1:nrow(predictions)) {
		if (i %% 1000 ==0) {
			print(i)
			}
		truePos[i] = sum(predictions[1:i, 2])
		falsePos[i] = length(which(predictions[1:i,2]==0))
	#	print(truePos[i])
	#	print(falsePos[i])
		}
	
	AUC = disc_integrate(falsePos/max(falsePos),truePos/max(truePos))
	
	# false and true pos
	par(mfrow=c(2,1))
	plot(falsePos,truePos,type='l',xlab = 'false positives',ylab='true positives',main=paste('AUC =',AUC),ylim=c(0,totalTrue),xlim = c(0,nrow(predictions)-totalTrue))	
	abline(a=0,b=totalTrue/(nrow(predictions)-totalTrue),lty=3)
	# precision and recall
	plot(truePos/totalTrue,truePos/(truePos+falsePos),type='l',xlab = 'recall',ylab='precision',ylim=c(0,1),xlim = c(0,1)	)
	#abline(a=1,b=-1,lty=3)
	
	return(cbind(falsePos,truePos))
	}


plot_roc = function(poses, AUC=NA) {
	par(mfrow=c(1,1))
	if (is.na(AUC)) {
		plot(poses[,1]/max(poses[,1]),poses[,2]/max(poses[,2]),type='l',xlab = 'false positive rate',ylab='true positive rate',ylim=c(0,1),xlim=c(0,1))
		abline(a=0,b=1,lty=3)
		} else {
		plot(poses[,1]/max(poses[,1]),poses[,2]/max(poses[,2]),type='l',xlab = 'false positive rate',ylab='true positive rate',ylim=c(0,1),xlim=c(0,1),main=paste('AUC =',AUC))	
		abline(a=0,b=1,lty=3)
		}
	}