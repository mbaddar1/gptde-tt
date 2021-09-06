# hands on for meshgrid
import numpy as np

x = np.linspace(start=0, stop=1, num=3)
y = np.linspace(start=0, stop=1, num=2)

print(f'x= {x}')
print(f'y = {y}')

x_grid,y_grid = np.meshgrid(x,y)

eclipse_ = x_grid**2 + 4*y_grid**2

print(f'eclipse = {eclipse_}')