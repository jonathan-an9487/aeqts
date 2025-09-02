import math ,random,numpy as np,matplotlib.pyplot as plt

def Knapback_III(K):
    base_weight = np.array([1,2,3,4,5,6,7,8,9,10])
    num_repeat = (K//10)+1
    item_weight = np.tile(base_weight,num_repeat)[:K]
    item_price = item_weight + 5
    Capacity = 0.5 * np.sum(item_weight)

    return item_weight,item_price,Capacity

def weight(solution,item_weight):
    total = 0
    for bit,w in zip(solution,item_weight):
        if bit == 1:
            total += w 
    return total

def Repair (solution,item_weight,item_price,capacity):
    sol = solution[:]

    while weight(sol,item_weight)>capacity:
        ones = [i for i,bit in enumerate(sol) if bit ==1]
        ratios = [(item_price[i]/item_weight[i],i) for i in ones]
        _, worst_index = min(ratios,key=lambda x:x[0])
        sol[worst_index] = 0

    return sol

def measure(qubits):
    solution = []

    for alpha,beta in qubits:
        p_0 = alpha**2

        if random.random() <p_0 :
            solution.append(0)
        else:
            solution.append(1)
    return solution
def update_qubits(K,N,qubits,population,theta,t,ITER):
    population.sort(key=lambda x:x[1],reverse=True)
    num_pairs = N//2

    for pair_idx in range(num_pairs):
        best_sol = population[pair_idx][0]
        worst_sol = population[N-1-pair_idx][0]

        pair_factor = pair_idx+1
        time_factor = 1-t/ITER
        current_theta = theta*time_factor/pair_factor
        
        current_theta = max(min(current_theta,math.pi/4), -math.pi/4)

        for j in range(K):

            alpha,beta = qubits[j]
            
            if best_sol[j]==1 and worst_sol[j]==0:  
                delta = current_theta #theta*(1-t/ITER)
            elif best_sol[j]==0 and worst_sol[j]==1:    
                delta = -current_theta #theta*(1-t/ITER)
            else:   
                delta=0

            alpha_lambda = alpha*math.cos(delta) - beta*math.sin(delta)
            beta_lambda = alpha*math.sin(delta) + beta*math.cos(delta)

            norm = math.sqrt(alpha_lambda**2 + beta_lambda**2)
            if norm>0:
                qubits[j] = [alpha_lambda/norm,beta_lambda/norm]
            else :
                qubits[j] = [alpha,beta]

    return qubits

def Initalize_Qubit(K):
    alpha = beta = 1/math.sqrt(2)
    qubits=[]
    for j in range(0,K):
        qubits.append([alpha,beta])

    return qubits

def Initalize_population(N,qubits,item_weight,item_price,capacity):
    population=[] 
    for i in range(N):
        sol = measure(qubits)#測量
        if  weight(sol,item_weight)> capacity:
            sol = Repair(sol,item_weight,item_price,capacity) 

        fitness = evaluate(sol,item_price)
        population.append((sol,fitness))

    return population

def evaluate(solution,item_price):
    return sum(p for bit,p in zip(solution, item_price) if bit ==1)

def Iteraion(item,p_size,ITER,theta,item_weight,item_price,capacity):
    fitness_history = []
    qubits = Initalize_Qubit(item)
    sb= None
    for t in range(ITER+1):
        population = Initalize_population(p_size,qubits,item_weight,item_price,capacity)    #更新Population
        population.sort(key=lambda x:x[1], reverse=True)

        if sb is None or population[0][1] >sb[1]:
            sb = population[0]

        if t <ITER:
            qubits= update_qubits(item,p_size,qubits,population,theta,t,ITER)

        b = population[0]
        
        if sb[1] > b[1]:
            elite = sb
            population.append(elite)
            population.sort(key=lambda x:x[1],reverse=True)
            population = population[:p_size]

        fitness_history.append(population[0][1])

        print(f"Iteration-{t}:best fitness = {population[0][1]}")
        #return sb
    return fitness_history

def chart(p):
    plt.xlabel("Iteraion")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.plot(range(len(p)),p,marker="o",markersize=0,label="Fitness")
    plt.show()

if __name__ == "__main__":
    item=2000 #物品數 K
    p_size=10 #族群大小(Population size)N
    ITER=1000 #迭帶次數 
    theta=0.01*math.pi #旋轉角度

    item_weight,item_price,capacity = Knapback_III(item)
    chart(Iteraion(item,p_size,ITER,theta,item_weight,item_price,capacity))
