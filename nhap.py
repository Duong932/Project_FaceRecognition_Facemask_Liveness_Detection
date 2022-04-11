import math
n = input(" nhap:")
try:
    a = float(n)
except ValueError:
    print(" day khong phai la so")
else:
    print(math.ceil(a))
