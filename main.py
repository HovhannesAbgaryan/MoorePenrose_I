import numpy
import sympy
import math


# region Functions

def find_derivatives(matrix, symbol):
    derivatives = list([matrix])
    while not matrix.is_zero_matrix:
        derivative = sympy.diff(matrix, symbol)
        derivatives.append(derivative)
        matrix = derivative
    return derivatives


def compute_discretes(derivatives, symbol, scale_coeff, value):
    discretes = list()
    for k in range(len(derivatives)):
        discrete = (scale_coeff ** k / math.factorial(k) *
                    derivatives[k].subs(symbol, value))
        discretes.append(discrete)
    return discretes


# endregion Functions

# region Main()

# region Matrix sizes

print("Enter mxn matrix sizes (m <= n)")
while True:
    while True:
        m = int(input("m = "))
        if m > 1:
            break

    while True:
        n = int(input("n = "))
        if n > 1:
            break

    if m <= n:
        break
print(f"Matrix sizes are: m = {m}, n = {n}")

# endregion Matrix sizes

# region Matrix elements

t = sympy.symbols('t', real=True)

A_t = sympy.Matrix()
print("Enter elements of A(t) matrix:")
for i in range(m):
    row = [sympy.parse_expr(element, local_dict=dict(t=t))
           for element in input().split()]
    A_t = A_t.row_insert(i, sympy.Matrix([row]))

# endregion Matrix elements

# region A1(t), A2(t)

A1_t, A2_t = A_t.as_real_imag()
print(f"A1(t) = {A1_t}")
print(f"A2(t) = {A2_t}")

# endregion A1(t), A2(t)

# region Analytical Method

print("ANALYTICAL METHOD")

# region B1(t), B2(t), B(t)

B1_t = A1_t * A1_t.T + A2_t * A2_t.T
print(f"B1(t) = {B1_t}")

B2_t = A2_t * A1_t.T - A1_t * A2_t.T
print(f"B2(t) = {B2_t}")

B_t = B1_t + B2_t * sympy.I
print(f"B(t) = {B_t}")

# endregion B1(t), B2(t), B(t)

# region X(t), X1(t), X2(t)

X_t = sympy.expand(B_t.pinv())
print(f"X(t) = {X_t}")

X1_t, X2_t = X_t.as_real_imag()
print(f"X1(t) = {X1_t}")
print(f"X2(t) = {X2_t}")

# endregion X(t), X1(t), X2(t)

# region A+(t)

real = A1_t.T * X1_t + A2_t.T * X2_t
imaginary = sympy.expand(A1_t.T * X2_t - A2_t.T * X1_t)
A_pseudo_inverse = real + imaginary * sympy.I
print(f"A+(t) = {A_pseudo_inverse}")

# endregion A+(t)

# endregion Analytical Method

# region Numerical-analytical method

print("NUMERICAL-ANALYTICAL METHOD")

# region A1(t) and A2(t) derivatives and discretes

A1_derives = find_derivatives(A1_t, t)
A2_derives = find_derivatives(A2_t, t)

H = int(input("Enter scaling coefficient: H = "))
appr_cntr = int(input("Enter approximation center's value: "))
A1_discretes = compute_discretes(A1_derives, t, H, appr_cntr)
A2_discretes = compute_discretes(A2_derives, t, H, appr_cntr)

# endregion A1(t) and A2(t) derivatives and discretes

# region B1(K), B2(K), B(K)

B1_K = list()
B2_K = list()
K = max(len(A1_discretes), len(A2_discretes))
for k in range(K + 1):
    B1_k = numpy.zeros(shape=(m, m), dtype=numpy.int_)
    B2_k = numpy.zeros(shape=(m, m), dtype=numpy.int_)
    for i in range(k + 1):
        index1_1 = i if i < len(A1_discretes) else len(A1_discretes) - 1
        index1_2 = k - i if k - i < len(A1_discretes) else len(A1_discretes) - 1
        index2_1 = i if i < len(A2_discretes) else len(A2_discretes) - 1
        index2_2 = k - i if k - i < len(A2_discretes) else len(A2_discretes) - 1
        B1_k += (A1_discretes[index1_1] * A1_discretes[index1_2].T +
                 A2_discretes[index2_1] * A2_discretes[index2_2].T)
        B2_k += (A2_discretes[index2_1] * A1_discretes[index1_2].T -
                 A1_discretes[index1_1] * A2_discretes[index2_2].T)
    B1_K.append(B1_k)
    B2_K.append(B2_k)
print(f"B1(K) = {B1_K}")
print(f"B2(K) = {B2_K}")

B_K = list(())
for k in range(K + 1):
    B_K.append([
        [B1_K[k], -B2_K[k]],
        [B2_K[k], B1_K[k]]
    ])

# endregion B1(K), B2(K), B(K)

# region X(K), X1(K), X2(K)

X_K = list(([
                [B1_K[0].pinv(), -B2_K[0].pinv()],
                [B2_K[0].pinv(), B1_K[0].pinv()]
            ],
))
for k in range(1, K + 1):
    summa = [
        [numpy.zeros(shape=(m, m), dtype=numpy.int_), numpy.zeros(shape=(m, m), dtype=numpy.int_)],
        [numpy.zeros(shape=(m, m), dtype=numpy.int_), numpy.zeros(shape=(m, m), dtype=numpy.int_)]
    ]
    for i in range(1, k + 1):
        summa[0][0] += B_K[i][0][0] * X_K[k - i][0][0] + B_K[i][0][1] * X_K[k - i][1][0]
        summa[0][1] += B_K[i][0][0] * X_K[k - i][0][1] + B_K[i][0][1] * X_K[k - i][1][1]
        summa[1][0] += B_K[i][1][0] * X_K[k - i][0][0] + B_K[i][1][1] * X_K[k - i][1][0]
        summa[1][1] += B_K[i][1][0] * X_K[k - i][0][1] + B_K[i][1][1] * X_K[k - i][1][1]
    X_K.append(
        [
            [-B1_K[0].pinv() * summa[0][0] + B2_K[0].pinv() * summa[1][0],
             -B1_K[0].pinv() * summa[0][1] + B2_K[0].pinv() * summa[1][1]],
            [-B2_K[0].pinv() * summa[0][0] + -B1_K[0].pinv() * summa[1][0],
             -B2_K[0].pinv() * summa[0][1] + -B1_K[0].pinv() * summa[1][1]]
        ])
# print(f"X(K) = {X_K}")

X1_K = list()
X2_K = list()
for k in range(K + 1):
    X1_K.append(X_K[k][0][0])
    X2_K.append(X_K[k][1][0])
print(f"X1(K) = {X1_K}")
print(f"X2(K) = {X2_K}")

# endregion X(K), X1(K), X2(K)

# region X1(t), X2(t)

X1_t = numpy.zeros(shape=(m, m), dtype=numpy.int_)
X2_t = numpy.zeros(shape=(m, m), dtype=numpy.int_)
for k in range(K + 1):
    X1_t += ((t - appr_cntr) / H) ** k * X1_K[k]
    X2_t += ((t - appr_cntr) / H) ** k * X2_K[k]
print(f"X1(t) = {X1_t}")
print(f"X2(t) = {X2_t}")

# endregion X1(t), X2(t)

# region A+(t)

a = A1_t - sympy.I * A2_t
x = X1_t + sympy.I * X2_t
A_pseudo_inverse = sympy.expand(a.T * x)
print(f"A+(t) = {A_pseudo_inverse}")

# endregion A+(t)

# endregion Numerical-analytical method

# endregion Main()
