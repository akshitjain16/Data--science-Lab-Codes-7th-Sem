# ------------------------------------------------------------
# Experiment 2 : Working with NumPy Arrays
# Aim: To create, manipulate, and perform operations on NumPy arrays
# ------------------------------------------------------------

import numpy as np


# ----------------------- Step 2 -----------------------------
def array_creation():
    print("1️⃣ Array Creation Examples:\n")

    a = np.array([10, 20, 30, 40, 50])
    print("1D Array:\n", a)

    b = np.array([[10, 20, 30], [40, 50, 60]])
    print("\n2D Array:\n", b)

    c = np.arange(0, 100, 20)
    print("\nArray using arange():\n", c)

    d = np.linspace(10, 50, 5)
    print("\nArray using linspace():\n", d)

    print("\nZeros array:\n", np.zeros((3, 2)))
    print("Ones array:\n", np.ones((2, 3)))

    return b  # return b for next step


# ----------------------- Step 3 -----------------------------
def array_attributes(b):
    print("\n2️⃣ Array Attributes:")
    print("Shape of b:", b.shape)
    print("Dimensions of b:", b.ndim)
    print("Size of b:", b.size)
    print("Data type of b:", b.dtype)


# ----------------------- Step 4 -----------------------------
def indexing_slicing():
    print("\n3️⃣ Indexing and Slicing:")

    arr = np.array([15, 25, 35, 45, 55, 65])
    print("Original array:", arr)
    print("First element:", arr[0])
    print("Last element:", arr[-1])
    print("Sliced elements (1:4):", arr[1:4])


# ----------------------- Step 5 -----------------------------
def reshaping_arrays():
    print("\n4️⃣ Reshaping Arrays:")

    arr2 = np.arange(10, 70, 10)
    print("Original array:", arr2)

    reshaped = arr2.reshape(2, 3)
    print("Reshaped 2x3 array:\n", reshaped)


# ----------------------- Step 6 -----------------------------
def mathematical_operations():
    print("\n5️⃣ Mathematical Operations:")

    x = np.array([10, 20, 30])
    y = np.array([40, 50, 60])

    print("x + y =", x + y)
    print("x * y =", x * y)
    print("x squared =", x ** 2)
    print("Square root of y =", np.sqrt(y))
    print("Mean of x =", np.mean(x))
    print("Sum of y =", np.sum(y))


# ----------------------- Step 7 -----------------------------
def aggregation_functions():
    print("\n6️⃣ Aggregation Functions:")

    data = np.array([[10, 20, 30],
                     [40, 50, 60]])
    print("Data:\n", data)
    print("Sum:", np.sum(data))
    print("Maximum:", np.max(data))
    print("Minimum:", np.min(data))
    print("Mean:", np.mean(data))
    print("Standard Deviation:", np.std(data))


# ----------------------- Step 8 -----------------------------
def broadcasting_example():
    print("\n7️⃣ Broadcasting Example:")

    m = np.array([10, 20, 30])
    n = np.array([[100], [200], [300]])

    print("Array m:", m)
    print("Array n:\n", n)
    print("Result of Broadcasting (m + n):\n", m + n)


# ----------------------- MAIN ------------------------------
def main():
    b = array_creation()
    array_attributes(b)
    indexing_slicing()
    reshaping_arrays()
    mathematical_operations()
    aggregation_functions()
    broadcasting_example()
    print("\n✅ Experiment Completed Successfully!")


if __name__ == "__main__":
    main()
