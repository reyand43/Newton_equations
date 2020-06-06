import numpy as np
import math
import numpy.linalg as linalg
import copy
from time import time
from scipy.linalg import lu

eps = 0.1**10

print('a)')
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 =1, 1, 1, 1, 1, 1, 2, 1, 2, 1
# apx - начальное приближение
apx = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
# k - количество итераций
k = 0
apxnext = copy.deepcopy(apx)
# W * deltax = -F
delta = 1
tic = time()
while (delta >=eps):
	k+=1
	#print('Итерация номер', k)
	x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = apx.item(0), apx.item(1), apx.item(2), apx.item(3), apx.item(4), apx.item(5), apx.item(6), apx.item(7), apx.item(8), apx.item(9)
	J = np.mat([[-x2 * math.sin(x2 * x1), -x1 * math.sin(x2 * x1), 3 * math.exp(-3 * x3), x5 ** 2, 2 * x4 * x5, -1, 0, -2 * math.cosh(2 * x8) * x9, -math.sinh(2 * x8), 2],
				[x2 * math.cos(x2 * x1), x1 * math.cos(x2 * x1), x9 * x7, 0, 6 * x5, -math.exp(-x10 + x6) -x8 - 1, x3 * x9, -x6, x3 * x7, math.exp(-x10 + x6)],
				[1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
				[-x5 / (x3 + x1) ** 2, -2 * x2 * math.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * math.sin(-x9 + x4), 1.0 / (x3 + x1), 0, -2 * math.cos(x7 * x10) * x10 * math.sin(x7 * x10), -1, 2 * math.sin(-x9 + x4), -2 * math.cos(x7 * x10) * x7 * math.sin(x7 * x10)],
				[2 * x8, -2 * math.sin(x2), 2 * x8, 1.0 / 
				(-x9 + x4) ** 2, 
				math.cos(x5), x7 * 
				math.exp(-x7 * (-x10 + x6)), -(x10 - x6) * math.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1, -1.0 / (-x9 + x4) ** 2, -x7 * math.exp(-x7 * (-x10 + x6))],
				[math.exp(x1 - x4 - x9), -1.5 * x10 * math.sin(3 * x10 * x2), -x6,-math.exp(x1 - x4 - x9), 2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -math.exp(x1 - x4 - x9), -1.5 * x2 * math.sin(3 * x10 * x2)],
				[math.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * math.sin(x4), x10 / x5 ** 2 * math.cos(x10 / x5 + x8), -math.cos(x4), x2 ** 3, -math.cos(x10 / x5 + x8), 0, -1.0 / x5 * math.cos(x10 / x5 + x8)],
				[2 * x5 * (x1 - 2 * x6), -x7 * math.exp(x2 * x7 + x10), -2 * math.cos(-x9 + x3), 1.5, (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * math.exp(x2 * x7 + x10), 0, 2 * math.cos(-x9 + x3), -math.exp(x2 * x7 + x10)],
				[-3, -2 * x8 * x10 * x7, 0, math.exp(x5 + x4), math.exp(x5 + x4), -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
				[x10, x9, -x8, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7, math.sin(x4 + x5 + x6), -x3, x2, x1]])

	F = np.mat([
				[math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440],
				[math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994],
				[x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904],
				[2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757],
				[math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862],
				[math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115],
				[x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014],
				[x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040],
				[7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499],
				[x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]])
	
	# x(1)=x(0)+deltax
	deltax = linalg.solve(J, -F)
	apxnext = apx + deltax
	#print('delta=', linalg.norm(abs(deltax)))
	delta = 0
	for i in range(10):
		if(abs(deltax[i])>delta):
			delta = abs(deltax[i])
	
	apx = copy.deepcopy(apxnext)
	
print('Количество итераций:', k)
print('x(k+1)=', apx)
toc = time()
print('Время выполнения функции:', toc-tic)


print('\nb)')
tic = time()
x0 = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
k = 0
apx = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])


delta = 1
apxnext = copy.deepcopy(apx)
y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = x0.item(0), x0.item(1), x0.item(2), x0.item(3), x0.item(4), x0.item(5), x0.item(6), x0.item(7), x0.item(8), x0.item(9)
J = np.mat([[-y2 * math.sin(y2 * y1), -y1 * math.sin(y2 * y1), 3 * math.exp(-3 * y3), y5 ** 2, 2 * y4 * y5, -1, 0, -2 * math.cosh(2 * y8) * y9, -math.sinh(2 * y8), 2],
				[y2 * math.cos(y2 * y1), y1 * math.cos(y2 * y1), y9 * y7, 0, 6 * y5, -math.exp(-y10 + y6) -y8 - 1, y3 * y9, -y6, y3 * y7, math.exp(-y10 + y6)],
				[1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
				[-y5 / (y3 + y1) ** 2, -2 * y2 * math.cos(y2 ** 2), -y5 / (y3 + y1) ** 2, -2 * math.sin(-y9 + y4), 1.0 / (y3 + y1), 0, -2 * math.cos(y7 * y10) * y10 * math.sin(y7 * y10), -1, 2 * math.sin(-y9 + y4), -2 * math.cos(y7 * y10) * y7 * math.sin(y7 * y10)],
				[2 * y8, -2 * math.sin(y2), 2 * y8, 1.0 / (-y9 + y4) ** 2, math.cos(y5), y7 * math.exp(-y7 * (-y10 + y6)), -(y10 - y6) * math.exp(-y7 * (-y10 + y6)), 2 * y3 + 2 * y1, -1.0 / (-y9 + y4) ** 2, -y7 * math.exp(-y7 * (-y10 + y6))],
				[math.exp(y1 - y4 - y9), -1.5 * y10 * math.sin(3 * y10 * y2), -y6,-math.exp(y1 - y4 - y9), 2 * y5 / y8, -y3, 0, -y5 ** 2 / y8 ** 2, -math.exp(y1 - y4 - y9), -1.5 * y2 * math.sin(3 * y10 * y2)],
				[math.cos(y4), 3 * y2 ** 2 * y7, 1, -(y1 - y6) * math.sin(y4), y10 / y5 ** 2 * math.cos(y10 / y5 + y8), -math.cos(y4), y2 ** 3, -math.cos(y10 / y5 + y8), 0, -1.0 / y5 * math.cos(y10 / y5 + y8)],
				[2 * y5 * (y1 - 2 * y6), -y7 * math.exp(y2 * y7 + y10), -2 * math.cos(-y9 + y3), 1.5, (y1 - 2 * y6) ** 2, -4 * y5 * (y1 - 2 * y6), -y2 * math.exp(y2 * y7 + y10), 0, 2 * math.cos(-y9 + y3), -math.exp(y2 * y7 + y10)],
				[-3, -2 * y8 * y10 * y7, 0, math.exp(y5 + y4), math.exp(y5 + y4), -7.0 / y6 ** 2, -2 * y2 * y8 * y10, -2 * y2 * y10 * y7, 3, -2 * y2 * y8 * y7],
				[y10, y9, -y8, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.sin(y4 + y5 + y6), -y3, y2, y1]])
P, L, U= lu(J)

while (delta >=eps):
	k+=1
	#print('Итерация номер', k)
	x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = apx.item(0), apx.item(1), apx.item(2), apx.item(3), apx.item(4), apx.item(5), apx.item(6), apx.item(7), apx.item(8), apx.item(9)
	F = np.mat([
				[math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440],
				[math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994],
				[x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904],
				[2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757],
				[math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862],
				[math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115],
				[x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014],
				[x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040],
				[7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499],
				[x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]])
	z = linalg.solve(L, np.dot(P, np.dot(P, np.dot(P,-F))))
	deltax = linalg.solve(U, z)
	apxnext = apx+deltax
	delta = 0
	for i in range(10):
		if(abs(deltax[i])>delta):
			delta = abs(deltax[i])
	
	#print('delta=', linalg.norm(abs(deltax)))
	apx = copy.deepcopy(apxnext)
print('Количество итераций:', k)
print('x(k+1)=', apx)
toc = time()
print('Время выполнения функции:', toc-tic)




print('c)')
for kshift in range(4, 5):
	print('Обычным методом:', kshift, 'итераций')
	k=0
	x0 = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
	y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = x0.item(0), x0.item(1), x0.item(2), x0.item(3), x0.item(4), x0.item(5), x0.item(6), x0.item(7), x0.item(8), x0.item(9)

	W = np.mat([[-y2 * math.sin(y2 * y1), -y1 * math.sin(y2 * y1), 3 * math.exp(-3 * y3), y5 ** 2, 2 * y4 * y5, -1, 0, -2 * math.cosh(2 * y8) * y9, -math.sinh(2 * y8), 2],
					[y2 * math.cos(y2 * y1), y1 * math.cos(y2 * y1), y9 * y7, 0, 6 * y5, -math.exp(-y10 + y6) -y8 - 1, y3 * y9, -y6, y3 * y7, math.exp(-y10 + y6)],
					[1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
					[-y5 / (y3 + y1) ** 2, -2 * y2 * math.cos(y2 ** 2), -y5 / (y3 + y1) ** 2, -2 * math.sin(-y9 + y4), 1.0 / (y3 + y1), 0, -2 * math.cos(y7 * y10) * y10 * math.sin(y7 * y10), -1, 2 * math.sin(-y9 + y4), -2 * math.cos(y7 * y10) * y7 * math.sin(y7 * y10)],
					[2 * y8, -2 * math.sin(y2), 2 * y8, 1.0 / (-y9 + y4) ** 2, math.cos(y5), y7 * math.exp(-y7 * (-y10 + y6)), -(y10 - y6) * math.exp(-y7 * (-y10 + y6)), 2 * y3 + 2 * y1, -1.0 / (-y9 + y4) ** 2, -y7 * math.exp(-y7 * (-y10 + y6))],
					[math.exp(y1 - y4 - y9), -1.5 * y10 * math.sin(3 * y10 * y2), -y6,-math.exp(y1 - y4 - y9), 2 * y5 / y8, -y3, 0, -y5 ** 2 / y8 ** 2, -math.exp(y1 - y4 - y9), -1.5 * y2 * math.sin(3 * y10 * y2)],
					[math.cos(y4), 3 * y2 ** 2 * y7, 1, -(y1 - y6) * math.sin(y4), y10 / y5 ** 2 * math.cos(y10 / y5 + y8), -math.cos(y4), y2 ** 3, -math.cos(y10 / y5 + y8), 0, -1.0 / y5 * math.cos(y10 / y5 + y8)],
					[2 * y5 * (y1 - 2 * y6), -y7 * math.exp(y2 * y7 + y10), -2 * math.cos(-y9 + y3), 1.5, (y1 - 2 * y6) ** 2, -4 * y5 * (y1 - 2 * y6), -y2 * math.exp(y2 * y7 + y10), 0, 2 * math.cos(-y9 + y3), -math.exp(y2 * y7 + y10)],
					[-3, -2 * y8 * y10 * y7, 0, math.exp(y5 + y4), math.exp(y5 + y4), -7.0 / y6 ** 2, -2 * y2 * y8 * y10, -2 * y2 * y10 * y7, 3, -2 * y2 * y8 * y7],
					[y10, y9, -y8, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.sin(y4 + y5 + y6), -y3, y2, y1]])
	P, L, U= lu(W)
	delta = 1
	apxnext = copy.deepcopy(apx)
	apx = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
	tic = time()

	while (delta >=eps):
		x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = apx.item(0), apx.item(1), apx.item(2), apx.item(3), apx.item(4), apx.item(5), apx.item(6), apx.item(7), apx.item(8), apx.item(9)
		if(k<kshift):
			k+=1
			#print('Итерация номер', k)
			J = np.mat([[-x2 * math.sin(x2 * x1), -x1 * math.sin(x2 * x1), 3 * math.exp(-3 * x3), x5 ** 2, 2 * x4 * x5, -1, 0, -2 * math.cosh(2 * x8) * x9, -math.sinh(2 * x8), 2],
						[x2 * math.cos(x2 * x1), x1 * math.cos(x2 * x1), x9 * x7, 0, 6 * x5, -math.exp(-x10 + x6) -x8 - 1, x3 * x9, -x6, x3 * x7, math.exp(-x10 + x6)],
						[1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
						[-x5 / (x3 + x1) ** 2, -2 * x2 * math.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * math.sin(-x9 + x4), 1.0 / (x3 + x1), 0, -2 * math.cos(x7 * x10) * x10 * math.sin(x7 * x10), -1, 2 * math.sin(-x9 + x4), -2 * math.cos(x7 * x10) * x7 * math.sin(x7 * x10)],
						[2 * x8, -2 * math.sin(x2), 2 * x8, 1.0 / 
						(-x9 + x4) ** 2, 
						math.cos(x5), x7 * 
						math.exp(-x7 * (-x10 + x6)), -(x10 - x6) * math.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1, -1.0 / (-x9 + x4) ** 2, -x7 * math.exp(-x7 * (-x10 + x6))],
						[math.exp(x1 - x4 - x9), -1.5 * x10 * math.sin(3 * x10 * x2), -x6,-math.exp(x1 - x4 - x9), 2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -math.exp(x1 - x4 - x9), -1.5 * x2 * math.sin(3 * x10 * x2)],
						[math.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * math.sin(x4), x10 / x5 ** 2 * math.cos(x10 / x5 + x8), -math.cos(x4), x2 ** 3, -math.cos(x10 / x5 + x8), 0, -1.0 / x5 * math.cos(x10 / x5 + x8)],
						[2 * x5 * (x1 - 2 * x6), -x7 * math.exp(x2 * x7 + x10), -2 * math.cos(-x9 + x3), 1.5, (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * math.exp(x2 * x7 + x10), 0, 2 * math.cos(-x9 + x3), -math.exp(x2 * x7 + x10)],
						[-3, -2 * x8 * x10 * x7, 0, math.exp(x5 + x4), math.exp(x5 + x4), -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
						[x10, x9, -x8, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7, math.sin(x4 + x5 + x6), -x3, x2, x1]])

			F = np.mat([
						[math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440],
						[math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994],
						[x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904],
						[2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757],
						[math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862],
						[math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115],
						[x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014],
						[x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040],
						[7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499],
						[x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]])
			
			# x(1)=x(0)+deltax
			deltax = linalg.solve(J, -F)
			apxnext = apx + deltax
			#print('delta=', linalg.norm(abs(deltax)))
			delta = 0
			for i in range(10):
				if(abs(deltax[i])>delta):
					delta = abs(deltax[i])
			apx = copy.deepcopy(apxnext)
		else:
			k+=1
			#print('Итерация номер', k)
			F = np.mat([
						[math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440],
						[math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994],
						[x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904],
						[2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757],
						[math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862],
						[math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115],
						[x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014],
						[x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040],
						[7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499],
						[x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]])
			z = linalg.solve(L, np.dot(P, np.dot(P, np.dot(P,-F))))
			deltax = linalg.solve(U, z)
			apxnext = apx+deltax
			delta = 0
			for i in range(10):
				if(abs(deltax[i])>delta):
					delta = abs(deltax[i])
			#print('delta=', linalg.norm(abs(deltax)))
			apx = copy.deepcopy(apxnext)
	print('Всего итераций:', k)
	print('x(k+1)=', apx)
	toc = time()
	print('Время выполнения функции:', toc-tic)
	print('\n')


print('d)')

for period in range(5, 6):
	tic = time()
	x0 = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
	apx = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
	delta = 1
	apxnext = copy.deepcopy(apx)
	k = 0
	print('\nМатрица J обновляется каждые', period, 'итераций')
	while (delta >=eps):
		
		if k%period==0:
			x0=copy.deepcopy(apx)
		
		y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = x0.item(0), x0.item(1), x0.item(2), x0.item(3), x0.item(4), x0.item(5), x0.item(6), x0.item(7), x0.item(8), x0.item(9)
		J = np.mat([[-y2 * math.sin(y2 * y1), -y1 * math.sin(y2 * y1), 3 * math.exp(-3 * y3), y5 ** 2, 2 * y4 * y5, -1, 0, -2 * math.cosh(2 * y8) * y9, -math.sinh(2 * y8), 2],
						[y2 * math.cos(y2 * y1), y1 * math.cos(y2 * y1), y9 * y7, 0, 6 * y5, -math.exp(-y10 + y6) -y8 - 1, y3 * y9, -y6, y3 * y7, math.exp(-y10 + y6)],
						[1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
						[-y5 / (y3 + y1) ** 2, -2 * y2 * math.cos(y2 ** 2), -y5 / (y3 + y1) ** 2, -2 * math.sin(-y9 + y4), 1.0 / (y3 + y1), 0, -2 * math.cos(y7 * y10) * y10 * math.sin(y7 * y10), -1, 2 * math.sin(-y9 + y4), -2 * math.cos(y7 * y10) * y7 * math.sin(y7 * y10)],
						[2 * y8, -2 * math.sin(y2), 2 * y8, 1.0 / (-y9 + y4) ** 2, math.cos(y5), y7 * math.exp(-y7 * (-y10 + y6)), -(y10 - y6) * math.exp(-y7 * (-y10 + y6)), 2 * y3 + 2 * y1, -1.0 / (-y9 + y4) ** 2, -y7 * math.exp(-y7 * (-y10 + y6))],
						[math.exp(y1 - y4 - y9), -1.5 * y10 * math.sin(3 * y10 * y2), -y6,-math.exp(y1 - y4 - y9), 2 * y5 / y8, -y3, 0, -y5 ** 2 / y8 ** 2, -math.exp(y1 - y4 - y9), -1.5 * y2 * math.sin(3 * y10 * y2)],
						[math.cos(y4), 3 * y2 ** 2 * y7, 1, -(y1 - y6) * math.sin(y4), y10 / y5 ** 2 * math.cos(y10 / y5 + y8), -math.cos(y4), y2 ** 3, -math.cos(y10 / y5 + y8), 0, -1.0 / y5 * math.cos(y10 / y5 + y8)],
						[2 * y5 * (y1 - 2 * y6), -y7 * math.exp(y2 * y7 + y10), -2 * math.cos(-y9 + y3), 1.5, (y1 - 2 * y6) ** 2, -4 * y5 * (y1 - 2 * y6), -y2 * math.exp(y2 * y7 + y10), 0, 2 * math.cos(-y9 + y3), -math.exp(y2 * y7 + y10)],
						[-3, -2 * y8 * y10 * y7, 0, math.exp(y5 + y4), math.exp(y5 + y4), -7.0 / y6 ** 2, -2 * y2 * y8 * y10, -2 * y2 * y10 * y7, 3, -2 * y2 * y8 * y7],
						[y10, y9, -y8, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.sin(y4 + y5 + y6), -y3, y2, y1]])
		P, L, U= lu(J)

		k+=1
		#print('Итерация номер', k)
		x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = apx.item(0), apx.item(1), apx.item(2), apx.item(3), apx.item(4), apx.item(5), apx.item(6), apx.item(7), apx.item(8), apx.item(9)
		F = np.mat([
					[math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440],
					[math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994],
					[x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904],
					[2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757],
					[math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862],
					[math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115],
					[x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014],
					[x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040],
					[7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499],
					[x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]])
		deltax = linalg.solve(J, -F)
		apxnext = apx + deltax
		#print('delta=', linalg.norm(abs(deltax)))
		delta = 0
		for i in range(10):
			if(abs(deltax[i])>delta):
				delta = abs(deltax[i])
		apx = copy.deepcopy(apxnext)
	print('Количество итераций:', k)
	toc = time()
	print('Время выполнения функции:', toc-tic)

print('e)')

print('\nea)')
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 =1, 1, 1, 1, 1, 1, 2, 1, 2, 1
# apx - начальное приближение
apx = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.2], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
# k - количество итераций
k = 0
apxnext = copy.deepcopy(apx)
# W * deltax = -F
delta = 1
tic = time()
while (delta >=eps):
	k+=1
	#print('Итерация номер', k)
	x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = apx.item(0), apx.item(1), apx.item(2), apx.item(3), apx.item(4), apx.item(5), apx.item(6), apx.item(7), apx.item(8), apx.item(9)
	J = np.mat([[-x2 * math.sin(x2 * x1), -x1 * math.sin(x2 * x1), 3 * math.exp(-3 * x3), x5 ** 2, 2 * x4 * x5, -1, 0, -2 * math.cosh(2 * x8) * x9, -math.sinh(2 * x8), 2],
				[x2 * math.cos(x2 * x1), x1 * math.cos(x2 * x1), x9 * x7, 0, 6 * x5, -math.exp(-x10 + x6) -x8 - 1, x3 * x9, -x6, x3 * x7, math.exp(-x10 + x6)],
				[1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
				[-x5 / (x3 + x1) ** 2, -2 * x2 * math.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * math.sin(-x9 + x4), 1.0 / (x3 + x1), 0, -2 * math.cos(x7 * x10) * x10 * math.sin(x7 * x10), -1, 2 * math.sin(-x9 + x4), -2 * math.cos(x7 * x10) * x7 * math.sin(x7 * x10)],
				[2 * x8, -2 * math.sin(x2), 2 * x8, 1.0 / 
				(-x9 + x4) ** 2, 
				math.cos(x5), x7 * 
				math.exp(-x7 * (-x10 + x6)), -(x10 - x6) * math.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1, -1.0 / (-x9 + x4) ** 2, -x7 * math.exp(-x7 * (-x10 + x6))],
				[math.exp(x1 - x4 - x9), -1.5 * x10 * math.sin(3 * x10 * x2), -x6,-math.exp(x1 - x4 - x9), 2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -math.exp(x1 - x4 - x9), -1.5 * x2 * math.sin(3 * x10 * x2)],
				[math.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * math.sin(x4), x10 / x5 ** 2 * math.cos(x10 / x5 + x8), -math.cos(x4), x2 ** 3, -math.cos(x10 / x5 + x8), 0, -1.0 / x5 * math.cos(x10 / x5 + x8)],
				[2 * x5 * (x1 - 2 * x6), -x7 * math.exp(x2 * x7 + x10), -2 * math.cos(-x9 + x3), 1.5, (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * math.exp(x2 * x7 + x10), 0, 2 * math.cos(-x9 + x3), -math.exp(x2 * x7 + x10)],
				[-3, -2 * x8 * x10 * x7, 0, math.exp(x5 + x4), math.exp(x5 + x4), -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
				[x10, x9, -x8, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7, math.sin(x4 + x5 + x6), -x3, x2, x1]])

	F = np.mat([
				[math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440],
				[math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994],
				[x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904],
				[2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757],
				[math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862],
				[math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115],
				[x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014],
				[x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040],
				[7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499],
				[x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]])
	
	# x(1)=x(0)+deltax
	deltax = linalg.solve(J, -F)
	apxnext = apx + deltax
	#print('delta=', linalg.norm(abs(deltax)))
	delta = 0
	for i in range(10):
		if(abs(deltax[i])>delta):
			delta = abs(deltax[i])
	apx = copy.deepcopy(apxnext)
	
print('Количество итераций:', k)
print('x(k+1)=', apx)
toc = time()
print('Время выполнения функции:', toc-tic)
print('\neb) Для модифицрованного метода не сходится')

print('\nec)')
def mix(kshift):
	print('Обычным методом', kshift)
	
	
	delta = 1
	k = 0
	apx = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.2], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
	while (delta>eps):
		x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = apx.item(0), apx.item(1), apx.item(2), apx.item(3), apx.item(4), apx.item(5), apx.item(6), apx.item(7), apx.item(8), apx.item(9)
		k+=1
		
		#print('apx=', apx)
		F = np.mat([
				[math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440],
				[math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994],
				[x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904],
				[2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757],
				[math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862],
				[math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115],
				[x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014],
				[x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040],
				[7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499],
				[x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]])
		if (k<=kshift):
			J = np.mat([[-x2 * math.sin(x2 * x1), -x1 * math.sin(x2 * x1), 3 * math.exp(-3 * x3), x5 ** 2, 2 * x4 * x5, -1, 0, -2 * math.cosh(2 * x8) * x9, -math.sinh(2 * x8), 2],
				[x2 * math.cos(x2 * x1), x1 * math.cos(x2 * x1), x9 * x7, 0, 6 * x5, -math.exp(-x10 + x6) -x8 - 1, x3 * x9, -x6, x3 * x7, math.exp(-x10 + x6)],
				[1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
				[-x5 / (x3 + x1) ** 2, -2 * x2 * math.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * math.sin(-x9 + x4), 1.0 / (x3 + x1), 0, -2 * math.cos(x7 * x10) * x10 * math.sin(x7 * x10), -1, 2 * math.sin(-x9 + x4), -2 * math.cos(x7 * x10) * x7 * math.sin(x7 * x10)],
				[2 * x8, -2 * math.sin(x2), 2 * x8, 1.0 / 
				(-x9 + x4) ** 2, 
				math.cos(x5), x7 * 
				math.exp(-x7 * (-x10 + x6)), -(x10 - x6) * math.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1, -1.0 / (-x9 + x4) ** 2, -x7 * math.exp(-x7 * (-x10 + x6))],
				[math.exp(x1 - x4 - x9), -1.5 * x10 * math.sin(3 * x10 * x2), -x6,-math.exp(x1 - x4 - x9), 2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -math.exp(x1 - x4 - x9), -1.5 * x2 * math.sin(3 * x10 * x2)],
				[math.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * math.sin(x4), x10 / x5 ** 2 * math.cos(x10 / x5 + x8), -math.cos(x4), x2 ** 3, -math.cos(x10 / x5 + x8), 0, -1.0 / x5 * math.cos(x10 / x5 + x8)],
				[2 * x5 * (x1 - 2 * x6), -x7 * math.exp(x2 * x7 + x10), -2 * math.cos(-x9 + x3), 1.5, (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * math.exp(x2 * x7 + x10), 0, 2 * math.cos(-x9 + x3), -math.exp(x2 * x7 + x10)],
				[-3, -2 * x8 * x10 * x7, 0, math.exp(x5 + x4), math.exp(x5 + x4), -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
				[x10, x9, -x8, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7, math.sin(x4 + x5 + x6), -x3, x2, x1]])

			
			deltax = linalg.solve(J, -F)
			apxnext = apx + deltax
			delta = 0
			for i in range(10):
				if(abs(deltax[i])>delta):
					delta = abs(deltax[i])
			
			apx = copy.deepcopy(apxnext)
			x0 = copy.deepcopy(apxnext)
		

		else:
			
			y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = x0.item(0), x0.item(1), x0.item(2), x0.item(3), x0.item(4), x0.item(5), x0.item(6), x0.item(7), x0.item(8), x0.item(9)
			W= np.mat([[-y2 * math.sin(y2 * y1), -y1 * math.sin(y2 * y1), 3 * math.exp(-3 * y3), y5 ** 2, 2 * y4 * y5, -1, 0, -2 * math.cosh(2 * y8) * y9, -math.sinh(2 * y8), 2],
					[y2 * math.cos(y2 * y1), y1 * math.cos(y2 * y1), y9 * y7, 0, 6 * y5, -math.exp(-y10 + y6) -y8 - 1, y3 * y9, -y6, y3 * y7, math.exp(-y10 + y6)],
					[1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
					[-y5 / (y3 + y1) ** 2, -2 * y2 * math.cos(y2 ** 2), -y5 / (y3 + y1) ** 2, -2 * math.sin(-y9 + y4), 1.0 / (y3 + y1), 0, -2 * math.cos(y7 * y10) * y10 * math.sin(y7 * y10), -1, 2 * math.sin(-y9 + y4), -2 * math.cos(y7 * y10) * y7 * math.sin(y7 * y10)],
					[2 * y8, -2 * math.sin(y2), 2 * y8, 1.0 / (-y9 + y4) ** 2, math.cos(y5), y7 * math.exp(-y7 * (-y10 + y6)), -(y10 - y6) * math.exp(-y7 * (-y10 + y6)), 2 * y3 + 2 * y1, -1.0 / (-y9 + y4) ** 2, -y7 * math.exp(-y7 * (-y10 + y6))],
					[math.exp(y1 - y4 - y9), -1.5 * y10 * math.sin(3 * y10 * y2), -y6,-math.exp(y1 - y4 - y9), 2 * y5 / y8, -y3, 0, -y5 ** 2 / y8 ** 2, -math.exp(y1 - y4 - y9), -1.5 * y2 * math.sin(3 * y10 * y2)],
					[math.cos(y4), 3 * y2 ** 2 * y7, 1, -(y1 - y6) * math.sin(y4), y10 / y5 ** 2 * math.cos(y10 / y5 + y8), -math.cos(y4), y2 ** 3, -math.cos(y10 / y5 + y8), 0, -1.0 / y5 * math.cos(y10 / y5 + y8)],
					[2 * y5 * (y1 - 2 * y6), -y7 * math.exp(y2 * y7 + y10), -2 * math.cos(-y9 + y3), 1.5, (y1 - 2 * y6) ** 2, -4 * y5 * (y1 - 2 * y6), -y2 * math.exp(y2 * y7 + y10), 0, 2 * math.cos(-y9 + y3), -math.exp(y2 * y7 + y10)],
					[-3, -2 * y8 * y10 * y7, 0, math.exp(y5 + y4), math.exp(y5 + y4), -7.0 / y6 ** 2, -2 * y2 * y8 * y10, -2 * y2 * y10 * y7, 3, -2 * y2 * y8 * y7],
					[y10, y9, -y8, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.sin(y4 + y5 + y6), -y3, y2, y1]])
			k+=1
			deltax=linalg.solve(W, -F)
			apxnext = apx + deltax
			delta = 0
			for i in range(10):
				if(abs(deltax[i])>delta):
					delta = abs(deltax[i])
			
			apx = copy.deepcopy(apxnext)
	print('Количество итераций:', k)
	print('x(k+1)=', apx)	
print('При k<6 delta увеличивается => расходится')

print('\nПри переходе на k = 6')
mix(6)
print('\nПри переходе на k = 7')
mix(7)
print('\nПри переходе на k = 8')
mix(8)
print('\nПри переходе на k = 9')
mix(9)
print('\nПри переходе на k = 10')
mix(10)
print('\nПри переходе на k = 11')
mix(11)


print('\ned)')

for period in range(4, 5):
	tic = time()
	x0 = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.2], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
	apx = np.mat([[0.5], [0.5], [1.5], [-1.0], [-0.2], [1.5], [0.5], [-0.5], [1.5], [-1.5]])
	delta = 1
	apxnext = copy.deepcopy(apx)
	k = 0
	print('\nМатрица J обновляется каждые', period, 'итераций')
	while (delta >=eps):
		
		if k%period==0:
			x0=copy.deepcopy(apx)
		
		y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = x0.item(0), x0.item(1), x0.item(2), x0.item(3), x0.item(4), x0.item(5), x0.item(6), x0.item(7), x0.item(8), x0.item(9)
		J = np.mat([[-y2 * math.sin(y2 * y1), -y1 * math.sin(y2 * y1), 3 * math.exp(-3 * y3), y5 ** 2, 2 * y4 * y5, -1, 0, -2 * math.cosh(2 * y8) * y9, -math.sinh(2 * y8), 2],
						[y2 * math.cos(y2 * y1), y1 * math.cos(y2 * y1), y9 * y7, 0, 6 * y5, -math.exp(-y10 + y6) -y8 - 1, y3 * y9, -y6, y3 * y7, math.exp(-y10 + y6)],
						[1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
						[-y5 / (y3 + y1) ** 2, -2 * y2 * math.cos(y2 ** 2), -y5 / (y3 + y1) ** 2, -2 * math.sin(-y9 + y4), 1.0 / (y3 + y1), 0, -2 * math.cos(y7 * y10) * y10 * math.sin(y7 * y10), -1, 2 * math.sin(-y9 + y4), -2 * math.cos(y7 * y10) * y7 * math.sin(y7 * y10)],
						[2 * y8, -2 * math.sin(y2), 2 * y8, 1.0 / (-y9 + y4) ** 2, math.cos(y5), y7 * math.exp(-y7 * (-y10 + y6)), -(y10 - y6) * math.exp(-y7 * (-y10 + y6)), 2 * y3 + 2 * y1, -1.0 / (-y9 + y4) ** 2, -y7 * math.exp(-y7 * (-y10 + y6))],
						[math.exp(y1 - y4 - y9), -1.5 * y10 * math.sin(3 * y10 * y2), -y6,-math.exp(y1 - y4 - y9), 2 * y5 / y8, -y3, 0, -y5 ** 2 / y8 ** 2, -math.exp(y1 - y4 - y9), -1.5 * y2 * math.sin(3 * y10 * y2)],
						[math.cos(y4), 3 * y2 ** 2 * y7, 1, -(y1 - y6) * math.sin(y4), y10 / y5 ** 2 * math.cos(y10 / y5 + y8), -math.cos(y4), y2 ** 3, -math.cos(y10 / y5 + y8), 0, -1.0 / y5 * math.cos(y10 / y5 + y8)],
						[2 * y5 * (y1 - 2 * y6), -y7 * math.exp(y2 * y7 + y10), -2 * math.cos(-y9 + y3), 1.5, (y1 - 2 * y6) ** 2, -4 * y5 * (y1 - 2 * y6), -y2 * math.exp(y2 * y7 + y10), 0, 2 * math.cos(-y9 + y3), -math.exp(y2 * y7 + y10)],
						[-3, -2 * y8 * y10 * y7, 0, math.exp(y5 + y4), math.exp(y5 + y4), -7.0 / y6 ** 2, -2 * y2 * y8 * y10, -2 * y2 * y10 * y7, 3, -2 * y2 * y8 * y7],
						[y10, y9, -y8, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.cos(y4 + y5 + y6) * y7, math.sin(y4 + y5 + y6), -y3, y2, y1]])
		P, L, U= lu(J)

		k+=1
		#print('Итерация номер', k)
		x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = apx.item(0), apx.item(1), apx.item(2), apx.item(3), apx.item(4), apx.item(5), apx.item(6), apx.item(7), apx.item(8), apx.item(9)
		F = np.mat([
					[math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440],
					[math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994],
					[x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904],
					[2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757],
					[math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862],
					[math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115],
					[x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014],
					[x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040],
					[7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499],
					[x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]])
		deltax = linalg.solve(J, -F)
		apxnext = apx + deltax
		#print('delta=', linalg.norm(abs(deltax)))
		delta = 0
		for i in range(10):
			if(abs(deltax[i])>delta):
				delta = abs(deltax[i])
		apx = copy.deepcopy(apxnext)
	print('Всего итераций:', k)
	print('x(k+1)=', apx)
	toc = time()
	print('Время выполнения функции:', toc-tic)
	print('\n')
