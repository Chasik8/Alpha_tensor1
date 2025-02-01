# import sys
#
# import numpy as np
# print(sys.getsizeof([0]))
# print(sys.getsizeof(np.array([],dtype=np.int8).tobytes()))

from sortedcontainers import SortedList, SortedSet, SortedDict

# initializing a sorted list with parameters
# it takes an iterable as a parameter.
# sorted_list = SortedList([1, 2, 3, 4])
def custom_comparator(a):
    # Например, сравниваем по длине строк
    return a[0]
# initializing a sorted list using default constructor
sorted_list = SortedList(key=custom_comparator)

# inserting values one by one using add()
for i in range(5, 0, -1):
    sorted_list.add([i,-i])

# prints the elements in sorted order
print('list after adding 5 elements: ', sorted_list[-1])