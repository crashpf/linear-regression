  ########### Linear Regression ###########

  #random lr dataset#
  data = read.csv("../Linear-Regression/data/test.csv")
  head(data)

  par(mfrow=c(1,2))
  plot(data$x, data$y)

  ###### Linear Regression using lm function ######
  model = lm(data$y ~ data$x)

  summary(model)
  #Adjusted R squared 0.9891

  # y = b0 + b1x
  b0 = model$coefficients[1]
  print(b0)

  b1 = model$coefficients[2]
  print(b1)

  #plot line usign abline
  plot(x=data$x, y=data$y)
  abline(model, col='red', lwd=2) 

  #plot using y=b0+b1*x
  plot(x=data$x, y=data$y)

  xhat = seq(min(data$x), max(data$x), length=1000)
  yhat = b0 + b1*xhat

  lines(xhat, yhat, col="deepskyblue", lwd=2)

  ###### Linear Regression from scratch using Gradient Descent ######
  ###y = m * x + b###
  x=data$x
  y=data$y

  m=0 
  b=0

  rate=0.0001 #learning rate
  iter=100000 #iterations

  #Gradient Descent Algorithm
  gd = function(x, y, m, b, rate, iter){
    n=length(x)
    losses=numeric(iter)
    
    #loop
    for (i in 1:iter){
      y_predicted = m*x + b
      
      #Gradients / Derivatives 
      gm = -(2/n)*sum(x*(y - y_predicted)) #derivative of MSE dL/dm
      gb = -(2/n)*sum(y - y_predicted) #derivative of MSE dL/db
      
      #new m & b parameters
      m=m - rate*gm
      b=b - rate*gb
      
      #MSE
      loss = (1/n) * sum((y - y_predicted)^2)
      losses[i] = loss
      #loss per 100 iterations completed
      if (i %% 1000 ==0){
        cat("Iteration:", i, "Cost:", loss, "\n")
      }
    }
    
    return(list(m=m, b=b, losses=losses))
    
  }

  result = gd(x, y, m, b, rate, iter)
  print(result)

  #y = m*x + b
  slope = result$m
  intercept = result$b

  #Compare results
  comparison = data.frame(Coefficient = c("Intercept", "Slope"),
    Gradient_Descent = c(intercept, slope),
    LM = c(b0, b1))

  print(comparison)

  #Compare results using plots
  plot(x, y, pch=20, main="Linear Regression: lm() function vs Gradient Descent")

  xhat = seq(min(x), max(x), length.out = 1000)

  #LM line
  lines(xhat, b0+b1*xhat, col="red", lwd=2)

  #Gradient Descent line
  lines(xhat, intercept+slope*xhat, col="blue", lwd=2, lty=2)

  #Legend
  legend("topleft", legend=c("lm() function", "Gradient Descent"), col=c("red", "blue"), lty=c(1,2), lwd=2)

  par(mfrow=c(1,2))
  #lm() function Plot
  plot(x, y, pch=20, main="Linear Regression: lm() function")
  abline(model, col='red', lwd=2) 

  #gradient descent plot
  plot(x, y, pch=20, main="Linear Regression: Gradient Descent")
  lines(xhat, intercept+slope*xhat, col="blue", lwd=2)
