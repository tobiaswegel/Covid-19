import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display

def testfunction():
    print("Import successful")
    print(np.pi)
    return
    
def logistic(x,L,k,x0):
    return L/(1+np.exp(-k*(x-x0)))
def logisticcured(x,L,k,x0,recoverytime):
    return logistic(x,L,k,x0)-logistic(x-recoverytime,L,k,x0)
def logisticinverse(x,L,k,x0):
    return (-np.log((L-x)/(x)))/k+x0

def exampleexp(f, afterdays):
    
    x = np.arange(0.0,afterdays,0.1)
    fig, ax = plt.subplots()
    for i in range(len(f)):
        y = np.power((1+f[i-1]),x)
        plt.plot(x,y)
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the first infected person')
    plt.title('growth factors and their effect on exponential growth')
    ax.text(0.1, 0.9,'k = ' + str(f),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    plt.show()
    
    x_pos = np.arange(len(f))
    y = [0.0]*len(f)
    for i in range(len(f)):
        y[i-1] = np.power((1+f[i-1]),afterdays)
    plt.bar(x_pos, y, align='center', alpha=0.5)
    plt.xticks(x_pos, f)
    plt.ylabel('cases after ' + str(afterdays) + ' days')
    plt.xlabel('factor E*p')
    plt.title('cases after ' + str(afterdays) + ' days, by factor')

    plt.show()
    
def examplelogistic(L,k,howlong):
    x = np.arange(0,howlong,0.1)
    x0 = 0
    fig, ax = plt.subplots()
    for i in range(len(k)):
        x0 = 0
        while(logistic(0,L[i],k[i],x0)>1):
            x0 = x0+1
        y = logistic(x,L[i],k[i],x0)
        plt.plot(x,y)
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the first infected person')
    plt.title('effect of growth factor and upper boundary')
    ax.text(0.1, 0.9,'k = ' + str(k),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    ax.text(0.1, 0.8,'L = ' + str(L),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    plt.show()
    
def examplerecovery(L,k,howlong,recoverytime):
    x = np.arange(0,howlong,0.1)
    x0 = 0
    fig, ax = plt.subplots()
    ax.text(0.1, 0.9,'k = ' + str(k),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    for i in range(len(k)):
        x0 = 0
        while(logistic(0,L[i],k[i],x0)>1):
            x0 = x0+1
        y = logisticcured(x,L[i],k[i],x0,recoverytime)
        plt.plot(x,y)
    plt.ylabel(' active cases in absolute numbers')
    plt.xlabel('days since the first infected person')
    plt.title('active cases')
    plt.show()

def examplegrowth(k,L,recoverytime,howlong):
    x = np.arange(0,howlong,0.1)
    x0 = 0
    fig, ax = plt.subplots()
    ax.text(0.1, 0.9,'k = ' + str(k) + '\nL = ' + str(L),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    while(logistic(0,L,k,x0)>1):
        x0 = x0+1
    y1 = logistic(x,L,k,x0)
    y2 = logisticcured(x,L,k,x0,recoverytime)
    y3 = np.exp(k*x)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.ylim(0,1.05*L)
    
    plt.ylabel('(active) cases in absolute numbers')
    plt.xlabel('days since the first infected person')
    plt.title('different models compared')
    plt.show()

def linearRegression(x,y):
    length = np.size(x)
    A = np.zeros((length,2))
    for i in range(length):
        A[i][0] = 1
        A[i][1] = x[i]
    B = np.dot(np.transpose(A), A)
    b = np.dot(np.transpose(A), y)
    solution = np.linalg.solve(B, b)
    t1 = np.arange(x[0], x[length-1], 0.01)
    y1 = f(t1, solution)
    return (t1,y1,solution)
def f(t,solution): return solution[0] + t*solution[1]

def bestexpfit(x,y,solution):
    length = np.size(x)
    t2 = np.arange(x[0], x[length-1], 0.01)
    y2 = g(t2,solution)
    return (t2,y2)
def g(t,solution): return np.exp(f(t,solution))

def bestexpfit2(x,y,solution):
    length = np.size(x)
    y2 = g(x,solution)
    return (x,y2)

def bestlogisticfit(x,y,solution,casemax):
    length = np.size(x)
    t2 = np.arange(x[0], x[length-1], 0.01)
    y2 = h(t2,solution,casemax)
    return (t2,y2)
def h(t,solution, casemax): return logistic(f(t,solution),casemax,1,0)

def prepareglobal(startfrom, endat,data):
    enddiff = len(data[0])-endat
    globalcases = [0]*(len(data[0])-4-startfrom-enddiff)
    for i in range(1,len(data)):
        for j in range(4+startfrom,len(data[0])-enddiff):
            globalcases[j-4-startfrom] = globalcases[j-4-startfrom]+data[i][j]
    x2 = range(0,len(data[0])-4-startfrom-enddiff)
    globalcasesWithoutZero = globalcases
    for i in range(len(globalcasesWithoutZero)):
        if(globalcasesWithoutZero[i]==0.0):
            globalcasesWithoutZero[i] = 1.0
    globalcaseslog = np.log(globalcasesWithoutZero)
    return (globalcases, globalcaseslog, x2)

def plotglobal(period,data):
    (startfrom, endat) = period
    (globalcases, globalcaseslog, x2) = prepareglobal(startfrom,endat,data)
    casestoday = max(globalcases)
    fig, ax = plt.subplots()
    ax.text(0.1, 0.9,'cases today: '+ str(round(casestoday)),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    plt.plot(x2,globalcases)
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the 22nd of January')
    plt.title('global confirmed Covid-19 Cases')
    plt.show()
    return

def expregglobal(period,data,certainty):
    (startfrom, endat) = period
    (globalcases, globalcaseslog, x2) = prepareglobal(startfrom, endat,data)
    (t1,y1,solution) = linearRegression(x2,globalcaseslog)
    (t2,y2) = bestexpfit(x2,globalcases, solution)
    plt.subplot(2,2,1)
    plt.plot(x2,globalcases,'bo')
    plt.plot(t2,y2,)
    plt.subplot(2,2,2)
    plt.plot(x2,globalcaseslog, 'bo')
    plt.plot(t1,y1,)
    plt.subplot(2,2,3)
    plt.plot(x2,globalcases,)
    plt.plot(t2,y2,)
    plt.subplot(2,2,4)
    plt.plot(x2,globalcaseslog,)
    plt.plot(t1,y1,)
    plt.show()
    
    casesmax=[0]*(len(y2))
    casesmin=[0]*(len(y2))
    for i in range(len(t2)):
        casesmax[i] = y2[i]*(1+certainty)
        casesmin[i] = y2[i]*(1-certainty)
    fig, ax = plt.subplots()
    plt.plot(x2,globalcases,'bo')
    plt.plot(x2,globalcases,)
    ax.fill_between(t2, casesmin, casesmax, alpha=0.2)
    plt.plot(t2,y2, '-')
    
    #Average deviation
    (x2,y3) = bestexpfit2(x2,globalcases, solution)
    y4 = abs_diff_list(y3,globalcases)
    avdev = round(np.mean(y4))
    
    
    ax.text(0.1, 0.9,'average deviation from \nbest fit exponential:' + str(avdev),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the 22nd of January')
    plt.title('exponential regression for the global confirmed Covid-19 Cases')
    plt.show()
    
def compare_global_country(period,data, country_index):
    (startfrom, endat) = period
    (globalcases, globalcaseslog, x2) = prepareglobal(startfrom, endat,data)
 
    country_index = country_index-1
    (countrycases, countrycaseslog, x3, x4, newcountrycases) = preparecountry(country_index, startfrom, endat,data,0)
    
    fig, ax = plt.subplots()
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the 22nd of January')
    plt.title('cases in '+indextoname(country_index)+' and the effect on exponential regression of global cases')
    ax.text(0.1, 0.9,str(indextoname(country_index)),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes, color="red")
    ax.text(0.1, 0.8,'global',
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes, color="blue")
    plt.plot(x2,globalcases,'b-')
    plt.plot(x3,countrycases,'r-')
    plt.show()
    
def abs_diff_list(x,y):
    z = [0]*len(x)
    for i in range(len(x)):
        z[i] = np.abs(x[i]-y[i])
    return z
    
def preparecountry(country_index, startfrom, endat,data,verzug):
    
    enddiff = len(data[0])-endat
    countrycases = [0]*(len(data[0])-4-startfrom-enddiff)
    newcountrycases = [0]*(len(data[0])-4-startfrom-enddiff)
    for i in range(4+startfrom,len(data[0])-enddiff):
        countrycases[i-4-startfrom] = data[country_index][i]
        if(i>0):
            newcountrycases[i-4-startfrom] = countrycases[i-4-startfrom]-countrycases[i-4-startfrom-1]
        else:
            newcountrycases[i-4-startfrom] = countrycases[i-4-startfrom]
    x3 = range(0,len(data[0])-4-startfrom-enddiff)
    x4 = range(0,(len(data[0])-4-startfrom-enddiff)+verzug)
    countrycasesWithoutZero = countrycases
    for i in range(len(countrycasesWithoutZero)):
        if(countrycasesWithoutZero[i]==0.0):
            countrycasesWithoutZero[i] = 1.0
    countrycaseslog = np.log(countrycasesWithoutZero)
    return (countrycases, countrycaseslog, x3, x4, newcountrycases)

def plotcountry(country_index, period,data):
    (startfrom, endat) = period
    country_index = country_index-1
    (countrycases, countrycaseslog, x3, x4, newcountrycases) = preparecountry(country_index, startfrom, endat,data,0)
    
    casestoday = max(countrycases)
    fig, ax = plt.subplots()
    ax.text(0.1, 0.9,'cases today: '+ str(round(casestoday)),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    
    plt.plot(x3,countrycases)
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the 22nd of January')
    plt.title('confirmed Covid-19 Cases in '+indextoname(country_index))
    plt.show()
    
    casestoday = max(newcountrycases)
    fig, ax = plt.subplots()
    ax.text(0.1, 0.9,'new cases today: '+ str(round(casestoday)),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    
    plt.plot(x3,newcountrycases)
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the 22nd of January')
    plt.title('daily new confirmed Covid-19 Cases in '+indextoname(country_index))
    plt.show()
    return

def expregcountry(country_index, period,data,certainty):
    (startfrom, endat) = period
    country_index = country_index-1
    (countrycases, countrycaseslog, x3, x4, newcountrycases) = preparecountry(country_index, startfrom, endat,data,0)
    (t1,y1,solution) = linearRegression(x3,countrycaseslog)
    (t2,y2) = bestexpfit(x3,countrycases, solution)
    
    plt.subplot(2,2,1)
    plt.plot(x3,countrycases,'bo')
    plt.plot(t2,y2,)
    plt.subplot(2,2,2)
    plt.plot(x3,countrycaseslog, 'bo')
    plt.plot(t1,y1,)
    plt.subplot(2,2,3)
    plt.plot(x3,countrycases,)
    plt.plot(t2,y2,)
    plt.subplot(2,2,4)
    plt.plot(x3,countrycaseslog,)
    plt.plot(t1,y1,)
    plt.show()
    
    (x3,y3) = bestexpfit2(x3,countrycases, solution)
    y4 = abs_diff_list(y3,countrycases)
    avdev = round(np.mean(y4))
    
    casesmax=[0]*(len(y2))
    casesmin=[0]*(len(y2))
    for i in range(len(t2)):
        casesmax[i] = y2[i]*(1+certainty)
        casesmin[i] = y2[i]*(1-certainty)
    fig, ax = plt.subplots()
    plt.plot(x3,countrycases,'bo')
    plt.plot(x3,countrycases,)
    ax.fill_between(t2, casesmin, casesmax, alpha=0.2)
    plt.plot(t2,y2, '-')
    ax.text(0.1, 0.9,'average deviation from \nbest fit exponential:' + str(avdev),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the 22nd of January + '+str(startfrom)+' days')
    plt.title('exponential regression for the confirmed Covid-19 Cases in '+ str(indextoname(country_index)))
    plt.show()
    
def expprogcountry( country_index,certainty,time, period,data, verzug):
    (startfrom, endat) = period
    country_index = country_index-1
    (countrycases, countrycaseslog, x3, x4, newcountrycases) = preparecountry(country_index, startfrom, endat,data,verzug)
    (t1,y1,solution) = linearRegression(x3,countrycaseslog)
    (t2,y2) = bestexpfit(x4,countrycases, solution)
    prognose = g(time,solution)
    progmin = (1-certainty)*prognose
    progmax = (1+certainty)*prognose
    
    print("prediction: " + str(prognose))
    print("worst case: " + str(progmax))
    print("best case: " + str(progmin))
    
    fig, ax = plt.subplots()
    if(time < max(x4)):
        plt.plot([time],[prognose], 'rs')
        plt.text(time,prognose, str(round(prognose)))
    plt.plot(x3,countrycases,'bo')
    plt.plot(t2,y2, '-')

    t3 = [0]*(len(t2)-verzug)
    y3 = [0]*(len(y2)-verzug)
    for i in range(len(t3)):
        t3[i] = t2[i]-verzug
        y3[i] = y2[i]-verzug
    plt.plot(t3,y3, '-')
    t4 = [max(x3)]*(len(t2)-verzug)
    plt.plot(t4,y3,'--')
    
    print('approximate real case number today: ' + str(max(y3)))  
    
    ax.text(0.1, 0.9,'real case number today: '+ str(round(max(y3)))+'\nprediciton for day ' +str(time)+': ' + str(round(prognose)),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the 22nd of January + '+str(startfrom)+' days')
    plt.title('prediction for the global confirmed Covid-19 Cases')
    plt.show()
    return #g(time,solution)

def preparecountrylog(country_index, startfrom, endat,data,casemaxfactor):
    enddiff = len(data[0])-endat
    countrycases = [0]*(len(data[0])-4-startfrom-enddiff)
    for i in range(4+startfrom,len(data[0])-enddiff):
        countrycases[i-4-startfrom] = data[country_index][i]
    x3 = range(0,len(data[0])-4-startfrom-enddiff)
    x4 = range(0,3*(len(data[0])-4-startfrom-enddiff))
    countrycasesWithoutZero = countrycases
    for i in range(len(countrycasesWithoutZero)):
        if(countrycasesWithoutZero[i]==0.0):
            countrycasesWithoutZero[i] = 1.0
    casemax = casemaxfactor*max(countrycases)
    countrycaseslogisticinv = logisticinverse(countrycasesWithoutZero,casemax,1,0)
    return (countrycases, countrycaseslogisticinv, x3, x4, casemax)

def logregcountry(country_index, period,data,casemaxin,casemaxfactor):
    (startfrom, endat) = period
    country_index = country_index-1
    (countrycases, countrycaseslogisticinv, x3, x4, casemax) = preparecountrylog(country_index, startfrom, endat,data,casemaxfactor)
    if(casemaxin != 1):
        casemax = casemaxin
    (t1,y1,solution) = linearRegression(x3,countrycaseslogisticinv)
    (t2,y2) = bestlogisticfit(x3,countrycases, solution, casemax)
    
    plt.subplot(2,2,1)
    plt.plot(x3,countrycases,'bo')
    plt.plot(t2,y2,)
    plt.subplot(2,2,2)
    plt.plot(x3,countrycaseslogisticinv, 'bo')
    plt.plot(t1,y1,)
    plt.subplot(2,2,3)
    plt.plot(x3,countrycases,)
    plt.plot(t2,y2,)
    plt.subplot(2,2,4)
    plt.plot(x3,countrycaseslogisticinv,)
    plt.plot(t1,y1,)
    plt.show()
    fig, ax = plt.subplots()
    
    (t2,y2) = bestlogisticfit(x4,countrycases, solution, casemax)
    plt.plot(x3,countrycases,'bo')
    plt.plot(t2,y2,)
    
    ax.text(0.1, 0.9,'maximal case number:\n'+ str(casemax),
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the 22nd of January + '+str(startfrom)+' days')
    plt.title('logistic regression of confirmed Covid-19 Cases in '+str(indextoname(country_index)))
    plt.show()
    
    plt.plot(x3,countrycases,'bo')
    for i in range(1,5):
        (countrycases, countrycaseslogisticinv, x3, x4, casemax) = preparecountrylog(country_index, startfrom, endat,data,np.power(2,i))
        (t1,y1,solution) = linearRegression(x3,countrycaseslogisticinv)
        (t2,y2) = bestlogisticfit(x4,countrycases, solution, casemax)
        plt.plot(t2,y2,)
        
    plt.ylabel('cases in absolute numbers')
    plt.xlabel('days since the 22nd of January + '+str(startfrom)+' days')
    plt.title('unpredictability of maximal confirmed Covid-19 Cases in '+str(indextoname(country_index)))
    plt.show()
    return    

def case_death_per_country(country_index,period, cases, deaths):
    (startfrom, endat) = period
    country_index = country_index-1
    enddiff = len(cases[0])-endat
    countrycases = [0]*(len(cases[0])-4-startfrom-enddiff)
    for i in range(4+startfrom,len(cases[0])-enddiff):
        countrycases[i-4-startfrom] = cases[country_index][i]
    countrydeaths = [0]*(len(deaths[0])-4-startfrom-enddiff)
    for i in range(4+startfrom,len(deaths[0])-enddiff):
        countrydeaths[i-4-startfrom] = deaths[country_index][i]
    
    (t1,y1,solution) = linearRegression(countrycases,countrydeaths)
    print('The regressional death rate is: ' +str(round(solution[1]*100,2))+'%')
    plt.plot(countrycases, countrydeaths, 'rx')
    plt.plot(t1,y1, '-')
    plt.ylabel('deaths of confirmed Covid-19 cases')
    plt.xlabel('confirmed Covid-19 cases')
    plt.title('case fatality rate (CFR) in '+str(indextoname(country_index)))
    plt.show()
    
def case_death_globally(cases, deaths, country_index):
    country_index = country_index-2
    countrycases = []
    for i in range(1,len(cases)):
        countrycases.append(cases[i][len(cases[0])-1])
    countrydeaths = []
    for j in range(1,len(deaths)):
        countrydeaths.append(deaths[j][len(cases[0])-1])
    plt.scatter(countrycases, countrydeaths)
    
    plt.text(countrycases[120],countrydeaths[120], indextoname(121))
    plt.text(countrycases[137],countrydeaths[137], indextoname(138))
    plt.text(countrycases[201],countrydeaths[201], indextoname(202))
    plt.text(countrycases[62],countrydeaths[62], indextoname(63))
    plt.text(countrycases[133],countrydeaths[133], indextoname(134))
    plt.text(countrycases[225],countrydeaths[225], indextoname(226))
    
    print('The momentary death rate in '+indextoname(country_index+1)+' is: ' +str(round((countrydeaths[country_index]/countrycases[country_index])*100,2)) + '%')
   
    casemin = min(countrycases)
    casemax = max(countrycases)
    p1 = np.linspace(casemin,casemax)
    for i in range(6):
        p2 = i*0.02*p1
        plt.plot(p1,p2, 'g--')
        plt.text(casemax,max(p2),str(i*2)+'%')
        
    plt.ylabel('deaths of confirmed Covid-19 cases today')
    plt.xlabel('confirmed Covid-19 cases today')
    plt.title('case fatality rate (CFR) globally')
    plt.show()
    
def indextoname(country_index):
    switcher = {
        140: 'Japan',
        121: 'Germany',
        138: 'Italy',
        202: 'Spain',
        63: 'Hubei,China',
        134: 'Iran',
        224: 'UK',
        226: 'US',
    }
    return switcher.get(country_index, "unknown countryname")
    

def nametoindex(country_name):
    switcher = {
        'Japan': 140,
        'Germany': 121,
        'Italy': 138,
        'Spain': 202,
        'Hubei,China': 63,
        'Iran': 134,
        'UK': 224,
        'US': 227,
    }
    return switcher.get(country_name, "unknown countryname")-1

def threeD_case_death_time(country_list, cases, deaths):
    countrycases = []
    print(countrycases)
    countrydeaths = []
    for k in range(len(country_list)):
        countrycases.insert(k, cases[country_list[k]-1][4:])
        countrydeaths.insert(k, deaths[country_list[k]-1][4:])
    time = [0]*len(countrycases[0])
    for i in range(0,len(countrycases[0])):
        time[i] = i
    ax = plt.axes(projection='3d')
    for k in range(len(country_list)):
        ax.plot(countrycases[k],time,countrydeaths[k])
        
    ax.set_xlabel("CCN")
    ax.set_ylabel("days since the 22nd of february")
    ax.set_zlabel("DCCN")
    plt.show()
    
    
    
########-----------------------------------------------Widgets------------------------------------------




