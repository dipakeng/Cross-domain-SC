# -*- coding: utf-8 -*-


import random
import numpy
import math
from solution import solution
import time


    

#def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter):
def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter,trainInput,trainOutput,net):
    print("vIGWO")
    print("Dimentions = " + str(dim))


   
    
    #Max_iter=1000
    #lb=-100
    #ub=100
    #dim=30  
    #SearchAgents_no=5
    
    # initialize alpha, beta, and delta_pos
    Alpha_pos=numpy.zeros(dim)
    Alpha_score=float("inf")
    # print("Alpha_score = " + str(Alpha_score))
    
    
    Beta_pos=numpy.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=numpy.zeros(dim)
    Delta_score=float("inf")

    Worst_pos=numpy.zeros(dim)
    Worst_score=-float("inf")
    
    #Initialize the positions of search agents
    Positions=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(len(Positions))
    Convergence_curve=numpy.zeros(Max_iter)
    s=solution()
    # print(type(s))
     # Loop counter
    print("vIGWO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            # print(len(Positions))
            Positions[i,:]=numpy.clip(Positions[i,:], lb, ub)
            # print(len(Positions))
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print(Positions[-1])
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:],trainInput,trainOutput,net)
            
            # Update Alpha, Beta, and Delta
        
            if fitness<Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness<Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
            
            # Finding worst solution
            if fitness>Worst_score:
                Worst_score=fitness
                Worst_pos=Positions[i,:].copy()
        
        
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        # a = numpy.full((dim), a, dtype=float)
        # two = numpy.full((dim), 2, dtype=int)
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            # Random variable's vectors

            r1 = numpy.random.rand(dim)
            r2 = numpy.random.rand(dim)

            A1 = 2*a*r1-a
            C1 = 2*r2

            D_alpha=abs(C1*Alpha_pos-Positions[i,:]); # Equation (3.5)-part 1
            X1=Alpha_pos-A1*D_alpha; # Equation (3.6)-part 1

            r1 = numpy.random.rand(dim)
            r2 = numpy.random.rand(dim)
            
            A2=2*a*r1-a; # Equation (3.3)
            C2=2*r2; # Equation (3.4)
            
            D_beta=abs(C2*Beta_pos-Positions[i,:]); # Equation (3.5)-part 2
            X2=Beta_pos-A2*D_beta; # Equation (3.6)-part 2

            r1 = numpy.random.rand(dim)
            r2 = numpy.random.rand(dim)

            A3=2*a*r1-a; # Equation (3.3)
            C3=2*r2; # Equation (3.4)
                
            D_delta=abs(C3*Delta_pos-Positions[i,:]); # Equation (3.5)-part 3
            X3=Delta_pos-A3*D_delta; # Equation (3.5)-part 3

            r1 = numpy.random.rand(dim)
            r2 = numpy.random.rand(dim)

            A4=2*a*r1-a; # Equation (3.3)
            C4=2*r2; # Equation (3.4)

            r1 = numpy.random.rand(dim)
            r2 = numpy.random.rand(dim)

            D_worse=abs(r1*Worst_pos-Positions[i,:]); # Equation (3.5)-part 3
            Xk=Worst_pos-A4*D_worse; # Equation (3.5)-part 3

            # Jk = J + (r1+r2)(J+ - J-)
            Xknew = X1 + r1*(X3-Worst_pos) + r2*(X3-Worst_pos)
            Positions[i,:]=(X1+X2+X3+Xknew)/4  # Equation (3.7)
            


            # for j in range (0,dim):     
                           
            #     r1=random.random() # r1 is a random number in [0,1]
            #     r2=random.random() # r2 is a random number in [0,1]
                
            #     A1=2*a*r1-a; # Equation (3.3)
            #     C1=2*r2; # Equation (3.4)
                
                # D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                # X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                # r1=random.random()
                # r2=random.random()
                
                # A2=2*a*r1-a; # Equation (3.3)
                # C2=2*r2; # Equation (3.4)
                
                # D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                # X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
            #     r1=random.random()
            #     r2=random.random() 
                
                # A3=2*a*r1-a; # Equation (3.3)
                # C3=2*r2; # Equation (3.4)
                
                # D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                # X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                # Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                
            
        
        
        Convergence_curve[l]=Alpha_score;

        if (l%1==0):
               print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)]);
    

    print(Positions[-1])
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="vIGWO"
    s.objfname=objf.__name__
    s.bestIndividual=Alpha_pos
    
    
    
    return s
    