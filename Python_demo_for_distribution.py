def Normald():
    import numpy as np
    from matplotlib import pyplot as plt
    import operator as op
    from functools import reduce
    import seaborn as sb
    from scipy.stats import norm
    import seaborn as sns
    # generate random numbers from N(0,1)
    def prob(mean,sd):
        pro=norm.cdf(np.arange(60,75,5),mean,sd)
        return pro[-1]-pro[0]

    data_normal = norm.rvs(size=10000,loc=70,scale=10)
    ax = sns.distplot(data_normal,
                  bins=100,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 8,'alpha':1})
    ax.set(xlabel='Normal Distribution', ylabel='Frequency')
    print("The mean weight of 500 college students is 70 kg and the standard deviation is 3 kg. Assuming that the weight is normally distributed, determine how many students weigh:")

    print("1.Between 60 kg and 75 kg.")
    print("answer")
    print(prob(70,3))
print("welcome to Python Demo for Distribution")
print("Enter your choice")
print(" 1 for Binomial Distribution")
print(" 2 for Uniform Distribution")
print(" 3 for Poisson Distribution")
print(" 4 for Normal Distribution")
print(" 5 for Exponential Distribution")
ch = int(input('Enter a number: '))
if(ch==1):
    Binomiald()
elif(ch==2):
    Uniformd()
elif(ch==3):
    Poissond()
elif(ch==4):
    Normald()
elif(ch==5):
    Exponentiald()
else:
    print("wrong option")
    
    
def Binomiald():
    
    import numpy as np
    from matplotlib import pyplot as plt
    import operator as op
    from functools import reduce
    from scipy.stats import binom

    def const(n, r):
            r = min(r, n-r)
            numer = reduce(op.mul, range(n, n-r, -1), 1)
            denom = reduce(op.mul, range(1, r+1), 1)
            return numer / denom

    def binomial(n, p):
            q = 1 - p
            y = [const(n, k) * (p ** k) * (q ** (n-k)) for k in range(n)]
            return y, np.mean(y), np.std(y)



    for ls in [(0.5, 10)]:
        p, n_experiment = ls[0], ls[1]
        x = np.arange(n_experiment)
        y, u, s = binomial(n_experiment, p)
        plt.plot(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f$' % (u, s))
 
    print("suppose that a candy company produces both milk chocolate and dark chocolate candy bars. The product mix is 50 percent of the candy bars are milk chocolate and 50 percent are dark chocolate. Say you choose ten candy bars at random, and choosing milk chocolate is defined as a success. The probability distribution of the number of successes during these ten trials with p = 0.5 is shown here. also find the probability of exactly two of them are milk chocolate")
    plt.legend()
#plt.savefig('graph/binomial.png')
    plt.show() 

    print("probalbitlity that exactly two are milk chocolate:-")
    print(binom.pmf(k=2,n=10,p=0.5))

    
def Uniformd():
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.stats import uniform 
    print("The amount of time, in minutes, that a person must wait for a bus is uniformly distributed between zero and 15 minutes, inclusive.")
    print("What is the probability that a person waits fewer than 12.5 minutes?")
    print("On the average, how long must a person wait? Find the mean, μ, and the standard deviation, σ. also plot a graph")
    def uniform1(x, a, b):

        y = [1 / (b - a) if a <= val and val <= b
                    else 0 for val in x]

        return x, y, np.mean(y), np.std(y)

    x = np.arange(-10, 100) # define range of x
    for ls in [(0, 15)]:
        a, b = ls[0], ls[1]
        x, y, u, s = uniform1(x, a, b)
        plt.plot(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f$' % (u, s))
    
    plt.legend()
    plt.show()
    arr=uniform.cdf(np.arange(0,13,0.5),loc=a,scale=b)
    res=arr[-1]-arr[0]
    print(res)
    
def Exponentiald():
    
    import math
    import numpy as np
    from scipy.stats import expon
    def exponential(x, lamb):
        y = lamb * np.exp(-lamb * x)
        return x, y, np.mean(y), np.std(y)

    def prob(m,n):
        x=-m*n
        ans=math.exp(x)
        return ans
    
    
    for lamb in [0.2]:
        x = np.arange(0, 20, dtype=np.float)
        x, y, u, s = exponential(x, lamb=lamb)
        plt.plot(x, y, label=r'$\mu=%.2f,\ \sigma=%.2f,'
                         r'\ \lambda=%d$' % (u, s, lamb))
    print("On the average, a certain computer part lasts ten years. The length of time the computer part lasts is exponentially , distributed . ")
    print("a. What is the probability that a computer part lasts more than 7 years? ")
    print("b. On the average , how long would five computer  parts last if they are used  one after another?")
    plt.legend()
    plt.show()
    print("The probability that a computer part lasts more than 7 years:-")
    print(prob(0.1,7))
    
def Poissond():
    import numpy as np
    from matplotlib import pyplot as plt
    import operator as op
    from functools import reduce
    from scipy.stats import poisson
    import seaborn as sb

    def prob_in_range(a,b,p):
        cdf=poisson.cdf(np.arange(a,b,1),p)
        prob=cdf[-1]-cdf[0]
        return prob


    data_binom = poisson.rvs(mu=4, size=10)
    ax = sb.distplot(data_binom,
                  kde=True,
                  color='green',
                  hist_kws={"linewidth": 8,'alpha':1})
    ax.set(xlabel='Poisson', ylabel='Frequency')
 
    print("Suppose a fast food restaurant can expect 2 customers every 3 minutes, on average.")
    print("What is the probability that four or fewer patrons will enter the restaurant in a 9 minute period?")
    print("draw a graph for 10 such trials.")
    plt.legend()
    plt.show() 
    print(prob_in_range(0,5,6))
    
    

