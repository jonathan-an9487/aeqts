import math ,random,numpy as np

# def Knapback_III(K):
#     base_weight = np.array([1,2,3,4,5,6,7,8,9,10])
#     num_repeat = (K//10)+1
#     item_weight = np.tile(base_weight,num_repeat)[:K]
#     item_price = item_weight + 5
#     Capacity = 0.5 * np.sum(item_weight)

#     return item_weight,item_price,Capacity

def Initalize_Qubit():
    alpha = beta = 1/math.sqrt(2)
    qubits=[]
    for j in range(0,):
        qubits.append([alpha,beta])
    return qubits

def Initalize_the_first_population():
    population=[] 
    for i in range(0,):
        population = Initalize_Qubit()
        if Initalize_Qubit > capacity:
            Repair(population) #重量超過容量 就remove
        superbeach = evaluate(population)  #計算profit
    return superbeach

def Iteraion():#1->MAX_ITER
    t=0
    while t <= MAX_ITER:
        superbeach = Initalize_the_first_population()    #更新Population
        Initalize_Qubit()   #更新qubits
        superbeach.sort()
        for i in range(1,N/2):
            best = superbeach[i]
            worst = superbeach[i-(N/2)-1]
            for j in range (1 ,K):
                if best[j]==1 and worst[j] == 0:
                    qubit[j]=qubit[j]-avg_theta
                if best[j] ==0 and worst[j] ==1:
                    qubit[j] = qubit[j]+avg_theta
                else:
                    pass

        best_solution = superbeach
        if best_solution > superbeach:
            superbeach = best_solution

    t=t+1

if __name__ == "__main__":
    K=10 #物品數 
    N=5 #族群大小(Population size)
    MAX_ITER=10  #最大迭帶次數 
    avg_theta=0.01*3.14159 #旋轉角度

    base_weight = np.array([1,2,3,4,5,6,7,8,9,10])
    num_repeat = (K//10)+1
    item_weight = np.tile(base_weight,num_repeat)[:K]

    item_price =item_weight+5
    capacity=0.5*np.sum(item_weight)

    Iteraion()


