import numpy as np
from computeCost import computeCost



def gradientDescent(X, y, theta, alpha, num_iters):
    """
        Функция позволяет выполнить градиентный спуск для поиска 
        параметров модели theta, используя матрицу объекты-признаки X, 
        вектор меток y, параметр сходимости alpha и число итераций 
        алгоритма num_iters
    """
    
    J_history = []
    m = y.shape[0]
    
    
    for i in range(num_iters):

        # ====================== Ваш код здесь ======================
        # Инструкция: выполнить градиентный спуск для num_iters итераций 
        # с целью вычисления вектора параметров theta, минимизирующего 
        # стоимостную функцию
        
        
        
        # ============================================================
        
        J_history.append(computeCost(X, y, theta)) # сохранение значений стоимостной функции
                                                   # на каждой итерации
                                                   
        theta = theta - (alpha/m) * (np.dot(np.transpose(X),(np.dot(X,theta) - y)))
          
      
                        
    return theta, J_history